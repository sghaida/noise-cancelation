package noise

import (
	"math"
	"time"
)

var _ Estimator = (*MCRAEstimator)(nil)

// MCRAConfig configures a minimum controlled recursive averaging estimator
type MCRAConfig struct {
	// Smoothing controls temporal smoothing of the observed power
	// this is αs in the mcra equations
	// valid values are in the range [0, 1)
	Smoothing float32

	// SpeechSmoothing controls smoothing of speech presence probability
	// this is αp in the mcra equations
	// valid values are in the range [0, 1)
	SpeechSmoothing float32

	// NoiseSmoothing controls the base smoothing of the noise estimate
	// this is αd in the mcra equations
	// valid values are in the range [0, 1)
	NoiseSmoothing float32

	// RatioThreshold controls when a bin is considered speech dominated
	// this is δ in the mcra equations
	RatioThreshold float32

	// WindowFrames controls the local minimum window size in frames
	WindowFrames int

	// BaselineDuration controls the maximum duration used to collect
	// the initial noise baseline
	//
	// baseline collection may finish earlier by calling FinishBaseline
	// when the known baseline period ends
	BaselineDuration time.Duration

	// SampleRate is the input sample rate in hz
	SampleRate int

	// HopSize is the stft hop size in samples
	//
	// BaselineDuration, SampleRate and HopSize are used to calculate
	// the maximum number of baseline frames
	HopSize int

	// MinProbabilityFrequency is the minimum frequency included when
	// calculating global speech and noise probabilities
	MinProbabilityFrequency float32

	// Floor is the minimum allowed power value
	Floor float32
}

// DefaultMCRAConfig returns the default mcra estimator configuration
func DefaultMCRAConfig() MCRAConfig {
	// refer to https://israelcohen.com/wp-content/uploads/2018/05/SPL_Jan2002.pdf
	return MCRAConfig{
		Smoothing:               0.8,
		SpeechSmoothing:         0.2,
		NoiseSmoothing:          0.95,
		RatioThreshold:          5,
		WindowFrames:            50,
		BaselineDuration:        5 * time.Second,
		SampleRate:              8000,
		HopSize:                 128,
		MinProbabilityFrequency: 80,
		Floor:                   1e-12,
	}
}

// MCRAEstimator estimates noise using minimum controlled recursive averaging
// before normal mcra processing starts, the estimator may collect a known
// noise only baseline
//
// during baseline collection, the initial noise psd is estimated as
//
//	N0[k] = (1 / T) * Σ P[t,k]
//
// where: P[t,k] = |X[t,k]|²
//
// once the baseline finishes, normal mcra processing begins
// the observed power is smoothed
//
//	S[t,k] = αs * S[t-1,k] + (1-αs) * P[t,k]
//
// a local minimum of the smoothed power is tracked
//
//	M[t,k] = min(S[t,k])
//
// the ratio between smoothed power and the local minimum is
//
//	R[t,k] = S[t,k] / max(M[t,k], ε)
//
// a speech indicator is derived from this ratio
//
//	I[t,k] = 1, if R[t,k] > δ
//	I[t,k] = 0, otherwise
//
// the speech presence probability is recursively smoothed
//
//	p[t,k] = αp * p[t-1,k] + (1-αp) * I[t,k]
//
// the adaptive noise smoothing factor is
//
//	αd[t,k] = αd + (1-αd) * p[t,k]
//
// the noise power spectral density is updated
//
//	N[t,k] = αd[t,k] * N[t-1,k] + (1-αd[t,k]) * P[t,k]
//
// when speech probability is high, αd[t,k] approaches 1 and the noise
// estimate changes slowly
// when speech probability is low, αd[t,k] stays closer to its base value
// and the noise estimate adapts more quickly
// the current and previous minimum windows are both considered to avoid
// unstable estimates when a new minimum-search window begins
// symbols
//
//	t   frame index
//	k   frequency bin index
//	αs  spectral smoothing factor
//	αp  speech probability smoothing factor
//	αd  base noise smoothing factor
//	δ   speech to minimum ratio threshold
//	ε   small positive value used for numerical stability
//	T   number of baseline frames
type MCRAEstimator struct {
	cfg MCRAConfig

	smoothed    []float32
	currentMin  []float32
	previousMin []float32
	speechProb  []float32
	noise       []float32

	// baselineSum contain the accumulated power for every frequency bin
	// while the known noise baseline is being captured
	baselineSum []float32

	// baselineFrames contain the actual number of baseline frames collected
	baselineFrames int

	// baselineFrameLimit contain the maximum number of frames allowed by
	// BaselineDuration
	baselineFrameLimit int

	// baselineActive indicate that incoming frames belong to the known
	// noise baseline period
	baselineActive bool

	framesInWindow int
	initialized    bool
}

// NewMCRAEstimator creates a minimum controlled recursive averaging estimator
func NewMCRAEstimator(cfg MCRAConfig) *MCRAEstimator {
	cfg = normalizeMCRAConfig(cfg)

	return &MCRAEstimator{
		cfg:                cfg,
		baselineFrameLimit: calculateBaselineFrames(cfg.BaselineDuration, cfg.SampleRate, cfg.HopSize),
	}
}

func normalizeMCRAConfig(cfg MCRAConfig) MCRAConfig {
	defaults := DefaultMCRAConfig()

	cfg = normalizeMCRASmoothing(cfg, defaults)
	cfg = normalizeMCRAWindow(cfg, defaults)
	cfg = normalizeMCRABaseline(cfg, defaults)
	cfg = normalizeMCRAProbability(cfg, defaults)

	return cfg
}

func normalizeMCRASmoothing(cfg, defaults MCRAConfig) MCRAConfig {
	if cfg.Smoothing < 0 || cfg.Smoothing >= 1 {
		cfg.Smoothing = defaults.Smoothing
	}

	if cfg.SpeechSmoothing < 0 || cfg.SpeechSmoothing >= 1 {
		cfg.SpeechSmoothing = defaults.SpeechSmoothing
	}

	if cfg.NoiseSmoothing < 0 || cfg.NoiseSmoothing >= 1 {
		cfg.NoiseSmoothing = defaults.NoiseSmoothing
	}

	return cfg
}

func normalizeMCRAWindow(cfg, defaults MCRAConfig) MCRAConfig {
	if cfg.RatioThreshold <= 1 {
		cfg.RatioThreshold = defaults.RatioThreshold
	}

	if cfg.WindowFrames <= 0 {
		cfg.WindowFrames = defaults.WindowFrames
	}

	return cfg
}

func normalizeMCRABaseline(cfg, defaults MCRAConfig) MCRAConfig {
	if cfg.BaselineDuration <= 0 {
		cfg.BaselineDuration = defaults.BaselineDuration
	}

	if cfg.SampleRate <= 0 {
		cfg.SampleRate = defaults.SampleRate
	}

	if cfg.HopSize <= 0 {
		cfg.HopSize = defaults.HopSize
	}

	return cfg
}

func normalizeMCRAProbability(cfg, defaults MCRAConfig) MCRAConfig {
	if cfg.MinProbabilityFrequency <= 0 {
		cfg.MinProbabilityFrequency = defaults.MinProbabilityFrequency
	}

	if cfg.Floor <= 0 {
		cfg.Floor = defaults.Floor
	}

	return cfg
}

// StartBaseline starts collecting a known noise only baseline
// call this when the application knows that incoming audio should
// primarily represent environmental background noise
//
// for cti call this when the recorder audio starts playing to capture the sound from the receiver
func (e *MCRAEstimator) StartBaseline() {
	e.baselineFrames = 0
	e.baselineActive = true
	e.initialized = false
	e.framesInWindow = 0

	for k := range e.baselineSum {
		e.baselineSum[k] = 0
	}

	for k := range e.noise {
		e.noise[k] = 0
	}

	for k := range e.smoothed {
		e.smoothed[k] = 0
	}

	for k := range e.currentMin {
		e.currentMin[k] = 0
	}

	for k := range e.previousMin {
		e.previousMin[k] = 0
	}

	for k := range e.speechProb {
		e.speechProb[k] = 0
	}
}

// FinishBaseline finishes baseline collection and initializes mcra
// this can be called before BaselineDuration expires
// like when the service starts receiving audio that needs to be processed
func (e *MCRAEstimator) FinishBaseline() {
	if !e.baselineActive {
		return
	}

	e.baselineActive = false

	if e.baselineFrames == 0 {
		return
	}

	e.initializeFromBaseline()
}

// Process estimates the noise power spectral density for one frame
func (e *MCRAEstimator) Process(power []float32) []float32 {
	if len(power) == 0 {
		return nil
	}

	if len(e.noise) != len(power) {
		e.resize(len(power))
	}

	// during the known baseline period, accumulate the input PSD instead
	// of running the normal MCRA adaptation
	if e.baselineActive {
		e.processBaseline(power)

		return e.noise
	}

	// MCRA requires an initial noise PSD
	// Until a baseline has been collected, there is no reliable noise
	// estimate to adapt
	if !e.initialized {
		return e.noise
	}

	for k, value := range power {
		e.updateBin(k, sanitizeMCRAPower(value, e.cfg.Floor))
	}

	e.framesInWindow++

	if e.framesInWindow >= e.cfg.WindowFrames {
		e.rotateWindow()
	}

	return e.noise
}

func (e *MCRAEstimator) processBaseline(power []float32) {
	e.updateBaseline(power)

	if e.baselineFrames >= e.baselineFrameLimit {
		e.FinishBaseline()
	}
}

func sanitizeMCRAPower(value, floor float32) float32 {
	if value < 0 {
		value = 0
	}

	return maxMCRAPower(value, floor)
}

// Reset clears the estimator state
func (e *MCRAEstimator) Reset() {
	e.smoothed = nil
	e.currentMin = nil
	e.previousMin = nil
	e.speechProb = nil
	e.noise = nil
	e.baselineSum = nil

	e.baselineFrames = 0
	e.baselineActive = false
	e.framesInWindow = 0
	e.initialized = false
}

// updateBaseline accumulates the known noise only baseline
// because the baseline interval is expected to contain environmental noise,
// the initial noise PSD is estimated using the mean observed power:
//
//	N0[k] = (1 / T) * Σ P[t,k]
//
// using the average instead of the minimum produces a representative noise
// spectral power density (PSD) when several seconds of known background noise are available
func (e *MCRAEstimator) updateBaseline(power []float32) {
	nextFrameCount := float32(e.baselineFrames + 1)

	for k, value := range power {
		if value < 0 {
			value = 0
		}

		value = maxMCRAPower(value, e.cfg.Floor)
		e.baselineSum[k] += value

		// Expose the running baseline estimate while collection is active
		e.noise[k] = e.baselineSum[k] / nextFrameCount
	}

	e.baselineFrames++
}

// initializeFromBaseline initializes normal mcra state from the captured
// baseline noise psd
func (e *MCRAEstimator) initializeFromBaseline() {
	if e.baselineFrames == 0 {
		return
	}

	count := float32(e.baselineFrames)

	for k := range e.baselineSum {
		noise := e.baselineSum[k] / count

		noise = maxMCRAPower(noise, e.cfg.Floor)

		e.noise[k] = noise
		e.smoothed[k] = noise
		e.currentMin[k] = noise
		e.previousMin[k] = noise
		e.speechProb[k] = 0
	}

	e.framesInWindow = 0
	e.initialized = true
}

// updateBin updates mcra state for one frequency bin
func (e *MCRAEstimator) updateBin(k int, power float32) {
	e.updateSmoothedPower(k, power)
	e.updateMinimum(k)
	e.updateSpeechProbability(k)
	e.updateNoise(k, power)
}

// updateSmoothedPower updates the temporally smoothed power
// the smoothing equation is:
//
//	S[t,k] = αs * S[t-1,k] + (1-αs) * P[t,k]
func (e *MCRAEstimator) updateSmoothedPower(k int, power float32) {
	alpha := e.cfg.Smoothing
	e.smoothed[k] = alpha*e.smoothed[k] + (1-alpha)*power
}

// updateMinimum updates the current local minimum
func (e *MCRAEstimator) updateMinimum(k int) {
	if e.smoothed[k] < e.currentMin[k] {
		e.currentMin[k] = e.smoothed[k]
	}
}

// updateSpeechProbability update the estimated speech probability
// the power to minimum ratio is
//
//	R[t,k] = S[t,k] / max(M[t,k], ε)
//
// the speech indicator is
//
//	I[t,k] = 1, if R[t,k] > δ
//	I[t,k] = 0, otherwise
//
// the probability estimate is
//
//	p[t,k] = αp * p[t-1,k] + (1-αp) * I[t,k]
func (e *MCRAEstimator) updateSpeechProbability(k int) {
	minimum := minMCRAPower(e.previousMin[k], e.currentMin[k])

	minimum = maxMCRAPower(minimum, e.cfg.Floor)

	ratio := e.smoothed[k] / minimum

	indicator := float32(0)

	if ratio > e.cfg.RatioThreshold {
		indicator = 1
	}

	alpha := e.cfg.SpeechSmoothing

	e.speechProb[k] = alpha*e.speechProb[k] + (1-alpha)*indicator
}

// SpeechProbability returns the power weighted average speech presence
// probability across all frequency bins at or above MinProbabilityFrequency
//
// each bin speech probability is weighted by its observed power so bins
// carrying more signal energy contribute more strongly to the result:
//
//	Pspeech = Σ(p[k] * P[k]) / ΣP[k]
//
// only bins whose center frequency is greater than or equal to
// MinProbabilityFrequency are included in the calculation
// where:
//
//	p[k] speech presence probability for frequency bin k
//	P[k] observed power for frequency bin k
//
// the returned value is in the range [0, 1], where values closer to 1
// indicate that more of the observed signal energy in the configured
// frequency range is associated with bins classified as speech
func (e *MCRAEstimator) SpeechProbability(power []float32) float32 {
	if len(e.speechProb) == 0 || len(power) != len(e.speechProb) || len(power) < 2 {
		return 0
	}

	binWidth := float32(e.cfg.SampleRate) / float32((len(power)-1)*2)

	var weightedSpeech float32
	var totalPower float32

	for k := 0; k < len(e.speechProb) && k < len(power); k++ {
		probability := e.speechProb[k]
		frequency := float32(k) * binWidth

		if frequency < e.cfg.MinProbabilityFrequency {
			continue
		}

		p := maxMCRAPower(power[k], e.cfg.Floor)

		weightedSpeech += probability * p
		totalPower += p
	}

	if totalPower <= e.cfg.Floor {
		return 0
	}

	return weightedSpeech / totalPower
}

// NoiseProbability returns the average noise presence probability
// across all frequency bins
func (e *MCRAEstimator) NoiseProbability(power []float32) float32 {
	return 1 - e.SpeechProbability(power)
}

// updateNoise update the noise power spectral density
//
// the adaptive smoothing factor is
//
//	αd[t,k] = αd + (1-αd) * p[t,k]
//
// the noise estimate is
//
//	N[t,k] = αd[t,k] * N[t-1,k] + (1-αd[t,k]) * P[t,k]
func (e *MCRAEstimator) updateNoise(k int, power float32) {
	alpha := e.cfg.NoiseSmoothing

	adaptiveAlpha := alpha + (1-alpha)*e.speechProb[k]

	noise := adaptiveAlpha*e.noise[k] + (1-adaptiveAlpha)*power

	e.noise[k] = maxMCRAPower(noise, e.cfg.Floor)
}

// resize allocate state for the requested number of frequency bins
func (e *MCRAEstimator) resize(size int) {
	e.smoothed = make([]float32, size)
	e.currentMin = make([]float32, size)
	e.previousMin = make([]float32, size)
	e.speechProb = make([]float32, size)
	e.noise = make([]float32, size)
	e.baselineSum = make([]float32, size)

	e.baselineFrames = 0
	e.framesInWindow = 0
	e.initialized = false
}

// rotateWindow start a new local minimum window
func (e *MCRAEstimator) rotateWindow() {
	copy(e.previousMin, e.currentMin)

	inf := float32(math.Inf(1))

	for k := range e.currentMin {
		e.currentMin[k] = inf
	}

	e.framesInWindow = 0
}

// calculateBaselineFrames convert a baseline duration to the number of
// STFT frames required for the configured sample rate and hop size
//
// for example
//
//	duration   = 5 seconds
//	sampleRate = 8000 Hz
//	hopSize    = 128 samples
//	10 * 8000 / 128 = 313 frames
func calculateBaselineFrames(duration time.Duration, sampleRate int, hopSize int) int {
	if duration <= 0 || sampleRate <= 0 || hopSize <= 0 {
		return 0
	}

	frames := math.Ceil(
		duration.Seconds() * float64(sampleRate) / float64(hopSize),
	)

	return int(frames)
}

// minMCRAPower return the smaller power value
func minMCRAPower(a, b float32) float32 {
	if a < b {
		return a
	}

	return b
}

// maxMCRAPower return the larger power value
func maxMCRAPower(a, b float32) float32 {
	if a > b {
		return a
	}

	return b
}
