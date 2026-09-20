package interference

import "math"

// TonalTransientConfig configures tonal and transient foreground interference detection
type TonalTransientConfig struct {
	// LocalRadius control how many neighboring bins are used for local tonal prominence
	LocalRadius int

	// TonalStartDB prominence where tonal evidence begins
	TonalStartDB float32

	// TonalFullDB prominence that produces full tonal evidence
	TonalFullDB float32

	// FluxStartDB positive spectral rise where transient evidence begins
	FluxStartDB float32

	// FluxFullDB positive spectral rise that produces full transient evidence
	FluxFullDB float32

	// MovementSearchRadius control how far the detector searches for the previous spectral peak
	MovementSearchRadius int

	// MovementStartBins bin movement where movement evidence begins
	MovementStartBins int

	// MovementFullBins bin movement that produces full movement evidence
	MovementFullBins int

	// MinFrequencyBin prevents interference detection below this FFT bin
	MinFrequencyBin int

	// MovementMinRelativePower require previous peak to contain enough energy to be considered related
	MovementMinRelativePower float32

	// HarmonicToleranceBins control how many bins around a harmonic relation are searched
	HarmonicToleranceBins int

	// HarmonicRelativePower control how much related harmonic energy is required for full support
	HarmonicRelativePower float32

	// Strength control maximum amount of extra interference attenuation
	//	targetGain = 1 - Strength * interferenceScore
	Strength float32

	// MinGain limit maximum attenuation
	// A value of 0.5 corresponds to approximately 6 dB attenuation
	MinGain float32

	// Attack control how quickly gain decreases when interference appears
	//	G[t,k] = Attack * G[t-1,k] + (1-Attack) * Gtarget[t,k]
	Attack float32

	// Release control how quickly gain returns toward one
	//	G[t,k] = Release * G[t-1,k] + (1-Release) * Gtarget[t,k]
	Release float32

	// SpreadRadius control how many neighboring bins receive partial attenuation
	SpreadRadius int

	// Floor prevent numerical instability when spectral power approaches zero
	Floor float32
}

// DefaultTonalTransientConfig returns the default tonal transient detector configuration
func DefaultTonalTransientConfig() TonalTransientConfig {
	return TonalTransientConfig{
		LocalRadius:              2,
		TonalStartDB:             5,
		TonalFullDB:              14,
		FluxStartDB:              3,
		FluxFullDB:               18,
		MovementSearchRadius:     6,
		MovementStartBins:        1,
		MovementFullBins:         4,
		MinFrequencyBin:          64,
		MovementMinRelativePower: 0.1,
		HarmonicToleranceBins:    1,
		HarmonicRelativePower:    0.15,
		Strength:                 0.5,
		MinGain:                  0.5,
		Attack:                   0.3,
		Release:                  0.85,
		SpreadRadius:             2,
		Floor:                    1e-12,
	}
}

// TonalTransientDetector detects narrow tonal foreground events
// that appear suddenly or move between frequency bins
// tonal prominence compare one bin against its local spectral neighborhood:
//
//	T[t,k] = 10 * log10(P[t,k] / (Plocal[t,k] + ε))
//
// where:
//
//	P[t,k]       current spectral power
//	Plocal[t,k]  mean neighboring power excluding bin k
//
// positive spectral flux measures a sudden increase relative to the previous frame
//
//	F[t,k] = max(10 * log10(P[t,k] / (P[t-1,k] + ε)), 0)
//
// movement compares the current peak location with the strongest nearby peak from the previous frame
//
//	M[t,k] = normalize(|k - kprev|)
//
// harmonic support searches related frequencies around
//
//	k/2, k/3, k/4, 2k, 3k, 4k
//
// for each valid relation r
//
//	Hr[t,k] = min(Pr[t] / (ρ * P[t,k]), 1)
//
// final harmonic support is
//
//	H[t,k] = max(Hr[t,k])
//
// interference score is
//
//	S[t,k] = Tscore[t,k] * max(Fscore[t,k], Mscore[t,k]) * (1-H[t,k])
//
// target gain is
//
//	Gtarget[t,k] = max(MinGain, 1-Strength*S[t,k])
//
// final gain uses separate attack and release smoothing to reduce musical artifacts
type TonalTransientDetector struct {
	cfg TonalTransientConfig

	previousPower []float32

	score             []float32
	gain              []float32
	targetGain        []float32
	tonalProminenceDB []float32
	spectralFluxDB    []float32
	harmonicSupport   []float32
	movementScore     []float32

	initialized bool
}

// NewTonalTransientDetector creates a tonal transient foreground interference detector
func NewTonalTransientDetector(cfg TonalTransientConfig) *TonalTransientDetector {
	defaults := DefaultTonalTransientConfig()

	cfg = normalizeTonalTransientConfig(cfg, defaults)

	return &TonalTransientDetector{
		cfg: cfg,
	}
}

func normalizeTonalTransientConfig(
	cfg TonalTransientConfig,
	defaults TonalTransientConfig,
) TonalTransientConfig {
	cfg = normalizeTonalBands(cfg, defaults)
	cfg = normalizeTonalMovement(cfg, defaults)
	cfg = normalizeTonalHarmonics(cfg, defaults)
	cfg = normalizeTonalGain(cfg, defaults)

	if cfg.SpreadRadius < 0 {
		cfg.SpreadRadius = defaults.SpreadRadius
	}

	if cfg.Floor <= 0 {
		cfg.Floor = defaults.Floor
	}

	return cfg
}

func normalizeTonalBands(
	cfg TonalTransientConfig,
	defaults TonalTransientConfig,
) TonalTransientConfig {
	if cfg.LocalRadius <= 0 {
		cfg.LocalRadius = defaults.LocalRadius
	}

	if cfg.TonalStartDB < 0 || cfg.TonalFullDB <= cfg.TonalStartDB {
		cfg.TonalStartDB = defaults.TonalStartDB
		cfg.TonalFullDB = defaults.TonalFullDB
	}

	if cfg.FluxStartDB < 0 || cfg.FluxFullDB <= cfg.FluxStartDB {
		cfg.FluxStartDB = defaults.FluxStartDB
		cfg.FluxFullDB = defaults.FluxFullDB
	}

	return cfg
}

func normalizeTonalMovement(
	cfg TonalTransientConfig,
	defaults TonalTransientConfig,
) TonalTransientConfig {
	if cfg.MovementSearchRadius <= 0 {
		cfg.MovementSearchRadius = defaults.MovementSearchRadius
	}

	if cfg.MovementStartBins <= 0 || cfg.MovementFullBins <= cfg.MovementStartBins {
		cfg.MovementStartBins = defaults.MovementStartBins
		cfg.MovementFullBins = defaults.MovementFullBins
	}

	if cfg.MovementMinRelativePower <= 0 || cfg.MovementMinRelativePower > 1 {
		cfg.MovementMinRelativePower = defaults.MovementMinRelativePower
	}

	return cfg
}

func normalizeTonalHarmonics(cfg TonalTransientConfig, defaults TonalTransientConfig) TonalTransientConfig {
	if cfg.HarmonicToleranceBins < 0 {
		cfg.HarmonicToleranceBins = defaults.HarmonicToleranceBins
	}

	if cfg.MinFrequencyBin <= 0 {
		cfg.MinFrequencyBin = defaults.MinFrequencyBin
	}

	if cfg.HarmonicRelativePower <= 0 || cfg.HarmonicRelativePower > 1 {
		cfg.HarmonicRelativePower = defaults.HarmonicRelativePower
	}

	return cfg
}

func normalizeTonalGain(cfg TonalTransientConfig, defaults TonalTransientConfig) TonalTransientConfig {
	cfg.Strength = normalizeUnitInterval(cfg.Strength, defaults.Strength)
	cfg.MinGain = normalizeUnitInterval(cfg.MinGain, defaults.MinGain)
	cfg.Attack = normalizeSmoothing(cfg.Attack, defaults.Attack)
	cfg.Release = normalizeSmoothing(cfg.Release, defaults.Release)

	return cfg
}

func normalizeUnitInterval(value, fallback float32) float32 {
	if value < 0 || value > 1 {
		return fallback
	}

	return value
}

func normalizeSmoothing(value, fallback float32) float32 {
	if value < 0 || value >= 1 {
		return fallback
	}

	return value
}

func (d *TonalTransientDetector) processBin(power []float32, k int) {
	centerPower := sanitizePower(power[k], d.cfg.Floor)

	if !d.isLocalPeak(power, k, centerPower) {
		return
	}

	totalScore := d.calculateBinScore(power, k, centerPower)

	if totalScore <= 0 {
		return
	}

	d.score[k] = totalScore

	targetGain := 1 - d.cfg.Strength*totalScore

	if targetGain < d.cfg.MinGain {
		targetGain = d.cfg.MinGain
	}

	d.spreadGain(k, targetGain)
}

func (d *TonalTransientDetector) calculateBinScore(
	power []float32,
	k int,
	centerPower float32,
) float32 {
	totalDB := d.tonalProminence(power, k, centerPower)
	totalScore := normalizeScore(totalDB, d.cfg.TonalStartDB, d.cfg.TonalFullDB)

	d.tonalProminenceDB[k] = totalDB

	if totalScore <= 0 {
		return 0
	}

	fluxDB := positiveSpectralFlux(centerPower, d.previousPower[k], d.cfg.Floor)
	fluxScore := normalizeScore(fluxDB, d.cfg.FluxStartDB, d.cfg.FluxFullDB)
	movement := d.calculateMovementScore(k, centerPower)
	harmonic := d.calculateHarmonicSupport(power, k, centerPower)

	d.spectralFluxDB[k] = fluxDB
	d.movementScore[k] = movement
	d.harmonicSupport[k] = harmonic

	persistentScore := float32(0.35) * totalScore
	temporalScore := maxFloat32(maxFloat32(fluxScore, movement), persistentScore)

	return clampFloat32(totalScore*temporalScore*(1-harmonic), 0, 1)
}

func (d *TonalTransientDetector) updateGains() {
	for k := range d.gain {
		target := d.targetGain[k]
		previous := d.gain[k]

		if target < previous {
			d.gain[k] = d.cfg.Attack*previous + (1-d.cfg.Attack)*target
		} else {
			d.gain[k] = d.cfg.Release*previous + (1-d.cfg.Release)*target
		}

		d.gain[k] = clampFloat32(d.gain[k], d.cfg.MinGain, 1)
	}
}

func (d *TonalTransientDetector) processFrame(power []float32) {
	start := maxInt(d.cfg.LocalRadius, d.cfg.MinFrequencyBin)
	end := len(power) - d.cfg.LocalRadius

	for k := start; k < end; k++ {
		d.processBin(power, k)
	}

	d.updateGains()
	d.storePreviousPower(power)
}

// Process detects tonal transient interference for one spectral power frame
// first frame initializes temporal state and returns gain one for every bin
// returned slices are owned by the detector and are reused by subsequent calls
func (d *TonalTransientDetector) Process(power []float32) Result {
	if len(power) == 0 {
		return Result{}
	}

	if len(d.previousPower) != len(power) {
		d.resize(len(power))
	}

	if !d.initialized {
		d.initialize(power)

		return d.result()
	}

	clear(d.score)
	clear(d.tonalProminenceDB)
	clear(d.spectralFluxDB)
	clear(d.harmonicSupport)
	clear(d.movementScore)

	for k := range d.targetGain {
		d.targetGain[k] = 1
	}

	d.processFrame(power)

	return d.result()
}

// Reset clear all tonal transient detector state
func (d *TonalTransientDetector) Reset() {
	d.previousPower = nil

	d.score = nil
	d.gain = nil
	d.targetGain = nil
	d.tonalProminenceDB = nil
	d.spectralFluxDB = nil
	d.harmonicSupport = nil
	d.movementScore = nil

	d.initialized = false
}

// result return the reusable detector result
func (d *TonalTransientDetector) result() Result {
	return Result{
		Score:             d.score,
		Gain:              d.gain,
		TonalProminenceDB: d.tonalProminenceDB,
		SpectralFluxDB:    d.spectralFluxDB,
		HarmonicSupport:   d.harmonicSupport,
		MovementScore:     d.movementScore,
	}
}

// initialize store the first spectral frame without applying interference attenuation
func (d *TonalTransientDetector) initialize(power []float32) {
	for k, value := range power {
		d.previousPower[k] = sanitizePower(value, d.cfg.Floor)
		d.gain[k] = 1
		d.targetGain[k] = 1
	}

	d.initialized = true
}

// resize allocate detector state for the requested number of frequency bins
func (d *TonalTransientDetector) resize(size int) {
	d.previousPower = make([]float32, size)

	d.score = make([]float32, size)
	d.gain = make([]float32, size)
	d.targetGain = make([]float32, size)
	d.tonalProminenceDB = make([]float32, size)
	d.spectralFluxDB = make([]float32, size)
	d.harmonicSupport = make([]float32, size)
	d.movementScore = make([]float32, size)

	for k := range d.gain {
		d.gain[k] = 1
		d.targetGain[k] = 1
	}

	d.initialized = false
}

// isLocalPeak report whether the center bin is greater than or equal to all local neighbors
func (d *TonalTransientDetector) isLocalPeak(power []float32, k int, centerPower float32) bool {
	for offset := -d.cfg.LocalRadius; offset <= d.cfg.LocalRadius; offset++ {
		if offset == 0 {
			continue
		}

		neighbor := sanitizePower(power[k+offset], d.cfg.Floor)

		if neighbor > centerPower {
			return false
		}
	}

	return true
}

// tonalProminence calculate local spectral prominence in dB while ignoring bins
// close to the detected peak that may contain FFT main lobe energy
//
//	T[t,k] = 10 * log10(P[t,k] / Pbackground[t,k])
func (d *TonalTransientDetector) tonalProminence(power []float32, k int, centerPower float32) float32 {
	const (
		guardBins  = 2
		searchBins = 6
	)
	var backgroundPower float64
	var count int

	for offset := -searchBins; offset <= searchBins; offset++ {
		// ignore the center and immediate neighbors because the Hann main lobe
		// spreads a narrow tone across these bins
		if absInt(offset) <= guardBins {
			continue
		}

		index := k + offset

		if index < 0 || index >= len(power) {
			continue
		}

		backgroundPower += float64(
			sanitizePower(power[index], d.cfg.Floor),
		)

		count++
	}

	if count == 0 {
		return 0
	}

	backgroundMean := backgroundPower / float64(count)

	ratio :=
		float64(centerPower) / math.Max(backgroundMean, float64(d.cfg.Floor))

	return float32(10 * math.Log10(ratio))
}

// calculateMovementScore compare current peak with the strongest nearby peak in the previous frame
func (d *TonalTransientDetector) calculateMovementScore(k int, centerPower float32) float32 {
	start := maxInt(0, k-d.cfg.MovementSearchRadius)
	end := minInt(len(d.previousPower)-1, k+d.cfg.MovementSearchRadius)

	previousPeakBin := k
	previousPeakPower := float32(0)

	for candidate := start; candidate <= end; candidate++ {
		value := sanitizePower(d.previousPower[candidate], d.cfg.Floor)

		if value > previousPeakPower {
			previousPeakPower = value
			previousPeakBin = candidate
		}
	}

	if previousPeakPower < centerPower*d.cfg.MovementMinRelativePower {
		return 0
	}

	distance := absInt(k - previousPeakBin)

	return normalizeScore(
		float32(distance), float32(d.cfg.MovementStartBins), float32(d.cfg.MovementFullBins),
	)
}

// calculateHarmonicSupport calculate protection from harmonically related spectral energy
//
//	H[k] = max min(Pr / (ρ * P[k]), 1)
//
// where r represents valid subharmonic and harmonic relations
func (d *TonalTransientDetector) calculateHarmonicSupport(
	power []float32, k int, centerPower float32,
) float32 {
	if centerPower <= d.cfg.Floor {
		return 0
	}

	targets := [6]int{
		int(math.Round(float64(k) / 2)),
		int(math.Round(float64(k) / 3)),
		int(math.Round(float64(k) / 4)),
		k * 2,
		k * 3,
		k * 4,
	}

	seen := [6]int{}
	seenCount := 0

	var support float32

	for _, target := range targets {
		if target <= 0 || target >= len(power) {
			continue
		}

		if !appendUniqueHarmonicTarget(&seen, &seenCount, target) {
			continue
		}

		relatedPower := localMaximum(
			power, target, d.cfg.HarmonicToleranceBins, d.cfg.Floor,
		)

		relation := relatedPower / (d.cfg.HarmonicRelativePower * centerPower)
		relation = clampFloat32(relation, 0, 1)

		support = maxFloat32(support, relation)
	}

	return support
}

func appendUniqueHarmonicTarget(seen *[6]int, seenCount *int, target int) bool {
	for i := 0; i < *seenCount; i++ {
		if seen[i] == target {
			return false
		}
	}

	seen[*seenCount] = target
	*seenCount++

	return true
}

// spreadGain apply progressively smaller attenuation to bins surrounding a detected peak
func (d *TonalTransientDetector) spreadGain(center int, centerGain float32) {
	radius := d.cfg.SpreadRadius

	for offset := -radius; offset <= radius; offset++ {
		k := center + offset

		if k < 0 || k >= len(d.targetGain) {
			continue
		}

		distance := absInt(offset)

		weight := float32(radius-distance+1) / float32(radius+1)

		neighborGain := 1 - (1-centerGain)*weight

		if neighborGain < d.targetGain[k] {
			d.targetGain[k] = neighborGain
		}
	}
}

// storePreviousPower store sanitized spectral power for the next frame
func (d *TonalTransientDetector) storePreviousPower(power []float32) {
	for k, value := range power {
		d.previousPower[k] = sanitizePower(value, d.cfg.Floor)
	}
}

// positiveSpectralFlux calculate only positive spectral changes in dB
//
//	F[t,k] = max(10 * log10(P[t,k] / P[t-1,k]), 0)
func positiveSpectralFlux(current float32, previous float32, floor float32) float32 {
	current = sanitizePower(current, floor)
	previous = sanitizePower(previous, floor)

	value := float32(10 * math.Log10(float64(current/previous)))

	if value < 0 {
		return 0
	}

	return value
}

// localMaximum return the largest power around one frequency bin
func localMaximum(power []float32, center int, radius int, floor float32) float32 {
	start := maxInt(0, center-radius)
	end := minInt(len(power)-1, center+radius)

	maximum := floor

	for k := start; k <= end; k++ {
		value := sanitizePower(power[k], floor)

		if value > maximum {
			maximum = value
		}
	}

	return maximum
}

// normalizeScore map a value to the range [0, 1]
func normalizeScore(value float32, start float32, full float32) float32 {
	if value <= start {
		return 0
	}

	if value >= full {
		return 1
	}

	return (value - start) / (full - start)
}

// sanitizePower replace invalid or very small spectral power with the configured floor
func sanitizePower(value float32, floor float32) float32 {
	if value < floor || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		return floor
	}

	return value
}

func clampFloat32(value float32, minimum float32, maximum float32) float32 {
	if value < minimum {
		return minimum
	}

	if value > maximum {
		return maximum
	}

	return value
}

func maxFloat32(a float32, b float32) float32 {
	if a > b {
		return a
	}

	return b
}

func absInt(value int) int {
	if value < 0 {
		return -value
	}

	return value
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}

	return b
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}

	return b
}
