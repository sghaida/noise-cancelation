package noise

import "math"

const defaultSPPFixedPriorSNR = float32(31.622776601683793)

// SPPMMSEConfig configures the Gerkmann Hendriks spp mmse noise psd estimator
type SPPMMSEConfig struct {
	// NoiseSmoothing controls recursive smoothing of the estimated noise psd
	//	N[t,k] = αN * N[t-1,k] + (1-αN) * Nmmse[t,k]
	// values must be in the range (0, 1)
	NoiseSmoothing float32

	// SPPSmoothing controls temporal smoothing of the speech presence probability
	//	pbar[t,k] = αP * pbar[t-1,k] + (1-αP) * p[t,k]
	// values must be in the range (0, 1)
	SPPSmoothing float32

	// SpeechPrior is the fixed prior probability P(H1) that speech is present
	// corresponding speech absence probability is
	//	P(H0) = 1 - P(H1)
	// values must be in the range (0, 1)
	SpeechPrior float32

	// FixedPriorSNR is the fixed linear snr used by the speech presence model
	// the paper uses 15 dB
	//	ξH1 = 10^(15/10) ≈ 31.6228
	// the value is fixed and is different from the adaptive decision directed ξ
	// used later by the log mmse suppressor
	FixedPriorSNR float32

	// StagnationThreshold controls when persistent high spp is considered a
	// possible indication that the noise PSD estimate has stagnated
	StagnationThreshold float32

	// MaxSpeechProbability limits the current spp when stagnation is detected
	MaxSpeechProbability float32

	// Floor prevents zero or negative power values from causing numerical instability
	Floor float32
}

// DefaultSPPMMSEConfig returns the default Gerkmann Hendriks spp mmse configuration
func DefaultSPPMMSEConfig() SPPMMSEConfig {
	return SPPMMSEConfig{
		NoiseSmoothing:       0.8,
		SPPSmoothing:         0.9,
		SpeechPrior:          0.5,
		FixedPriorSNR:        defaultSPPFixedPriorSNR,
		StagnationThreshold:  0.99,
		MaxSpeechProbability: 0.99,
		Floor:                1e-12,
	}
}

// SPPMMSEEstimator estimates the spectral noise psd using the Gerkmann Hendriks
// speech presence probability mmse method
// for every time frame t and frequency bin k the a posteriori snr used by the
// speech presence model is:
//
//	γ[t,k] = P[t,k] / N[t-1,k]
//
// where:
//
//	P[t,k]     observed noisy spectral power
//	N[t-1,k]   previous spectral noise PSD estimate
//
// speech presence probability is
//
//	p[t,k] = {1 + ((1-q)/q) * (1+ξH1) * exp(-γ[t,k] * ξH1/(1+ξH1)) }^-1
//
// where:
//
//	q       fixed prior probability of speech presence
//	ξH1     fixed SNR assumed by the speech presence model
//
// speech presence probability is smoothed over time
//
//	pbar[t,k] = αP * pbar[t-1,k] + (1-αP) * p[t,k]
//
// if pbar exceeds the stagnation threshold the current probability is limited
//
//	p[t,k] = min(p[t,k], pmax)
//
// conditional mmse noise periodogram estimate is
//
//	Nmmse[t,k] = (1-p[t,k]) * P[t,k] + p[t,k] * N[t-1,k]
//
// final noise psd is recursively smoothed
//
//	N[t,k] = αN * N[t-1,k] + (1-αN) * Nmmse[t,k]
type SPPMMSEEstimator struct {
	cfg SPPMMSEConfig

	noisePSD           []float32
	speechProbability  []float32
	smoothedSpeechProb []float32

	baselineSum    []float64
	baselineFrames int
	baselineActive bool

	initialized bool
}

// NewSPPMMSEEstimator creates a Gerkmann Hendriks SPP MMSE noise PSD estimator
func NewSPPMMSEEstimator(cfg SPPMMSEConfig) *SPPMMSEEstimator {
	cfg = normalizeSPPMMSEConfig(cfg)

	return &SPPMMSEEstimator{
		cfg: cfg,
	}
}

func normalizeSPPMMSEConfig(cfg SPPMMSEConfig) SPPMMSEConfig {
	defaults := DefaultSPPMMSEConfig()

	cfg = normalizeSPPMMSESmoothing(cfg, defaults)
	cfg = normalizeSPPMMSEPrior(cfg, defaults)
	cfg = normalizeSPPMMSEProbability(cfg, defaults)

	return cfg
}

func normalizeSPPMMSESmoothing(cfg, defaults SPPMMSEConfig) SPPMMSEConfig {
	if cfg.NoiseSmoothing <= 0 || cfg.NoiseSmoothing >= 1 {
		cfg.NoiseSmoothing = defaults.NoiseSmoothing
	}

	if cfg.SPPSmoothing <= 0 || cfg.SPPSmoothing >= 1 {
		cfg.SPPSmoothing = defaults.SPPSmoothing
	}

	return cfg
}

func normalizeSPPMMSEPrior(cfg, defaults SPPMMSEConfig) SPPMMSEConfig {
	if cfg.SpeechPrior <= 0 || cfg.SpeechPrior >= 1 {
		cfg.SpeechPrior = defaults.SpeechPrior
	}

	if cfg.FixedPriorSNR <= 0 {
		cfg.FixedPriorSNR = defaults.FixedPriorSNR
	}

	return cfg
}

func normalizeSPPMMSEProbability(cfg, defaults SPPMMSEConfig) SPPMMSEConfig {
	if cfg.StagnationThreshold <= 0 || cfg.StagnationThreshold > 1 {
		cfg.StagnationThreshold = defaults.StagnationThreshold
	}

	if cfg.MaxSpeechProbability <= 0 || cfg.MaxSpeechProbability > 1 {
		cfg.MaxSpeechProbability = defaults.MaxSpeechProbability
	}

	if cfg.Floor <= 0 {
		cfg.Floor = defaults.Floor
	}

	return cfg
}

// Process estimates the noise psd for one spectral power frame
//
// returned slice is owned by the estimator and is reused by subsequent Process calls
// when baseline collection is active the observed power is accumulated into
// a mean baseline noise psd and the spp update is temporarily disabled
func (e *SPPMMSEEstimator) Process(power []float32) []float32 {
	if len(power) == 0 {
		return nil
	}

	if len(e.noisePSD) != len(power) {
		e.resize(len(power))
	}

	if e.baselineActive {
		e.updateBaseline(power)

		return e.noisePSD
	}

	if !e.initialized {
		e.initializeNoise(power)

		return e.noisePSD
	}

	for k, observedPower := range power {
		observedPower = sanitizeSPPPower(observedPower, e.cfg.Floor)
		previousNoise := sanitizeSPPPower(e.noisePSD[k], e.cfg.Floor)
		probability := e.calculateSpeechPresenceProbability(observedPower, previousNoise)

		smoothedProbability :=
			e.cfg.SPPSmoothing*e.smoothedSpeechProb[k] + (1-e.cfg.SPPSmoothing)*probability

		e.smoothedSpeechProb[k] = smoothedProbability

		// Persistent probabilities close to one can prevent the noise PSD
		// from adapting when the current noise estimate is too small
		if smoothedProbability > e.cfg.StagnationThreshold && probability > e.cfg.MaxSpeechProbability {
			probability = e.cfg.MaxSpeechProbability
		}

		e.speechProbability[k] = probability

		conditionalNoise := (1-probability)*observedPower + probability*previousNoise

		updatedNoise :=
			e.cfg.NoiseSmoothing*previousNoise + (1-e.cfg.NoiseSmoothing)*conditionalNoise

		e.noisePSD[k] = sanitizeSPPPower(updatedNoise, e.cfg.Floor)
	}

	return e.noisePSD
}

// SpeechPresenceProbability returns the current spp value for every frequency bin
//
// returned slice is owned by the estimator and is reused by subsequent
// process calls
func (e *SPPMMSEEstimator) SpeechPresenceProbability() []float32 {
	return e.speechProbability
}

// StartBaseline resets the estimator and begins noise only baseline collection
// during baseline collection the initial noise psd is estimated using
//
//	Nbaseline[k] = (1/L) * Σ P[t,k]
//
// where L is the number of collected baseline frames
func (e *SPPMMSEEstimator) StartBaseline() {
	e.noisePSD = nil
	e.speechProbability = nil
	e.smoothedSpeechProb = nil
	e.baselineSum = nil

	e.baselineFrames = 0
	e.baselineActive = true
	e.initialized = false
}

// FinishBaseline finishes baseline collection and enables normal spp mmse updates
//
// if no baseline frames were collected the estimator remains uninitialized and
// the next Process call initializes the noise psd from that frame
func (e *SPPMMSEEstimator) FinishBaseline() {
	e.baselineActive = false

	if e.baselineFrames > 0 {
		e.initialized = true
	}
}

// Reset clears all spp mmse estimator state
func (e *SPPMMSEEstimator) Reset() {
	e.noisePSD = nil
	e.speechProbability = nil
	e.smoothedSpeechProb = nil
	e.baselineSum = nil

	e.baselineFrames = 0
	e.baselineActive = false
	e.initialized = false
}

// initializeNoise initializes the noise PSD from the first observed power frame
func (e *SPPMMSEEstimator) initializeNoise(power []float32) {
	for k, observedPower := range power {
		e.noisePSD[k] = sanitizeSPPPower(observedPower, e.cfg.Floor)
		e.speechProbability[k] = 0
		e.smoothedSpeechProb[k] = 0
	}

	e.initialized = true
}

// updateBaseline accumulates one noise only frame into the initial mean noise PSD
func (e *SPPMMSEEstimator) updateBaseline(power []float32) {
	nextFrameCount := e.baselineFrames + 1

	for k, observedPower := range power {
		observedPower = sanitizeSPPPower(observedPower, e.cfg.Floor)

		e.baselineSum[k] += float64(observedPower)

		mean := float32(e.baselineSum[k] / float64(nextFrameCount))

		e.noisePSD[k] = sanitizeSPPPower(mean, e.cfg.Floor)
		e.speechProbability[k] = 0
		e.smoothedSpeechProb[k] = 0
	}

	e.baselineFrames = nextFrameCount
}

// calculateSpeechPresenceProbability calculates the posterior probability that
// speech is present in one time frequency bin
//
//	p = {1 + ((1-q)/q) * (1+ξH1) * exp(-γ * ξH1/(1+ξH1)) }^-1
//
// with:
//
//	γ = P / N
func (e *SPPMMSEEstimator) calculateSpeechPresenceProbability(
	observedPower float32, noisePower float32,
) float32 {
	observedPower = sanitizeSPPPower(observedPower, e.cfg.Floor)
	noisePower = sanitizeSPPPower(noisePower, e.cfg.Floor)
	gamma := float64(observedPower / noisePower)
	speechPrior := float64(e.cfg.SpeechPrior)
	absencePrior := 1 - speechPrior
	fixedPriorSNR := float64(e.cfg.FixedPriorSNR)
	exponent := -gamma * fixedPriorSNR / (1 + fixedPriorSNR)
	likelihoodRatio := (absencePrior / speechPrior) * (1 + fixedPriorSNR) * math.Exp(exponent)
	probability := 1 / (1 + likelihoodRatio)

	if math.IsNaN(probability) || math.IsInf(probability, 0) {
		return 0
	}

	if probability < 0 {
		return 0
	}

	if probability > 1 {
		return 1
	}

	return float32(probability)
}

// resize allocate estimator state for the requested number of frequency bins
//
// Changing the number of frequency bins invalidates all previously learned
// spectral state
func (e *SPPMMSEEstimator) resize(size int) {
	e.noisePSD = make([]float32, size)
	e.speechProbability = make([]float32, size)
	e.smoothedSpeechProb = make([]float32, size)
	e.baselineSum = make([]float64, size)

	e.baselineFrames = 0
	e.initialized = false
}

// sanitizeSPPPower replace invalid or very small power values with the
// configured numerical floor
func sanitizeSPPPower(value float32, floor float32) float32 {
	if value < floor || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		return floor
	}

	return value
}
