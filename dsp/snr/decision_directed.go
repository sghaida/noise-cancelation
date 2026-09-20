package snr

import "fmt"

// DecisionDirectedConfig configures decision directed priori snr estimation
type DecisionDirectedConfig struct {
	// Alpha controls how strongly the previous clean estimate contributes
	// to the current a priori snr estimate
	// valid values are in the range [0, 1)
	Alpha float32

	// Floor prevents division by zero when the noise psd is very small
	Floor float32
}

// DefaultDecisionDirectedConfig returns the default snr estimator configuration
func DefaultDecisionDirectedConfig() DecisionDirectedConfig {
	return DecisionDirectedConfig{Alpha: 0.98, Floor: 1e-12}
}

// DecisionDirected estimates posteriori and priori snr values
// the a posteriori snr is
//
//		γ[t,k] = P[t,k] / N[t,k]
//
//		P[t,k] observed noisy power
//		N[t,k] estimated noise power (this is coming from mcra)
//		γ posteriori snr gives something like current observed energy against the estimated noise energy
//
//	 decision directed priori snr is
//
// ξ[t,k] = α * ξclean[t-1,k] + (1-α) * max(γ[t,k]-1, 0)
// where: ξclean[t-1,k] = Ŝ[t-1,k] / N[t-1,k]
//
//	P[t,k]    observed noisy power
//	N[t,k]    estimated noise PSD
//	Ŝ[t-1,k]  previous estimated clean power
//	ξclean    previous clean speech to noise ratio
//	α         decision directed smoothing factor
type DecisionDirected struct {
	cfg DecisionDirectedConfig

	gamma []float32
	xi    []float32

	// previousCleanPower store the clean power estimated for the previous frame
	// it is retained for diagnostics while previousCleanSNR is used by the
	// decision-directed estimator
	previousCleanPower []float32

	// previousCleanSNR store clean speech/noise ratio estimated for the previous frame
	// previousCleanSNR[k] = previousCleanPower[k] / previousNoisePSD[k]
	// This is equivalent to:
	//
	//	previousCleanSNR[k] = previousGain[k]² * previousGamma[k]
	previousCleanSNR []float32

	initialized bool
}

// NewDecisionDirected creates a decision directed snr estimator
func NewDecisionDirected(cfg DecisionDirectedConfig) *DecisionDirected {
	defaults := DefaultDecisionDirectedConfig()

	if cfg.Alpha < 0 || cfg.Alpha >= 1 {
		cfg.Alpha = defaults.Alpha
	}

	if cfg.Floor <= 0 {
		cfg.Floor = defaults.Floor
	}

	return &DecisionDirected{
		cfg: cfg,
	}
}

// Process estimates posteriori and priori snr for one spectral frame
// the returned slices are owned by the estimator and are reused on
// subsequent calls
func (d *DecisionDirected) Process(power []float32, noisePSD []float32) ([]float32, []float32, error) {
	if len(power) == 0 {
		return nil, nil, nil
	}

	if len(power) != len(noisePSD) {
		return nil, nil, fmt.Errorf(
			"power contains %d bins, noise PSD contains %d", len(power), len(noisePSD),
		)
	}

	if len(d.gamma) != len(power) {
		d.resize(len(power))
	}

	for k, observedPower := range power {
		gamma, instantaneousXi := decisionDirectedValues(
			observedPower,
			noisePSD[k],
			d.cfg.Floor,
		)
		xi := d.calculateXi(k, instantaneousXi)

		d.gamma[k] = gamma
		d.xi[k] = xi
	}

	return d.gamma, d.xi, nil
}

func decisionDirectedValues(observedPower float32, noisePower float32, floor float32) (float32, float32) {
	if observedPower < 0 {
		observedPower = 0
	}

	if noisePower < floor {
		noisePower = floor
	}

	gamma := observedPower / noisePower
	instantaneousXi := gamma - 1
	if instantaneousXi < 0 {
		instantaneousXi = 0
	}

	return gamma, instantaneousXi
}

func (d *DecisionDirected) calculateXi(k int, instantaneousXi float32) float32 {
	if !d.initialized {
		return instantaneousXi
	}

	return d.cfg.Alpha*d.previousCleanSNR[k] + (1-d.cfg.Alpha)*instantaneousXi
}

// Update stores the estimated clean power for the next snr frame
func (d *DecisionDirected) Update(cleanPower, noisePSD []float32) error {
	if len(cleanPower) != len(d.previousCleanPower) {
		return fmt.Errorf(
			"clean power contains %d bins, want %d", len(cleanPower), len(d.previousCleanPower),
		)
	}

	if len(noisePSD) != len(d.previousCleanSNR) {
		return fmt.Errorf(
			"noise PSD contains %d bins, want %d", len(noisePSD), len(d.previousCleanSNR),
		)
	}

	for k, value := range cleanPower {
		if value < 0 {
			value = 0
		}

		noisePower := noisePSD[k]

		if noisePower < d.cfg.Floor {
			noisePower = d.cfg.Floor
		}

		d.previousCleanPower[k] = value

		d.previousCleanSNR[k] = value / noisePower
	}

	d.initialized = true

	return nil
}

// Reset clears all decision directed state
func (d *DecisionDirected) Reset() {
	d.gamma = nil
	d.xi = nil
	d.previousCleanPower = nil
	d.previousCleanSNR = nil
	d.initialized = false
}

// resize allocate state for the requested number of frequency bins
func (d *DecisionDirected) resize(size int) {
	d.gamma = make([]float32, size)
	d.xi = make([]float32, size)
	d.previousCleanPower = make([]float32, size)
	d.previousCleanSNR = make([]float32, size)
	d.initialized = false
}
