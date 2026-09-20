package suppressor

import (
	"fmt"
	"math"

	"github.com/sghaida/noise-cancelation/audio"
	"github.com/sghaida/noise-cancelation/dsp/snr"
)

var _ Suppressor = (*LogMMSE)(nil)

// LogMMSEConfig configures the log mmse noise suppressor
type LogMMSEConfig struct {
	// MinGain limits maximum attenuation
	// value of 0.05 corresponds to approximately -26 dB amplitude gain
	MinGain float32

	// MaxGain prevents the suppressor from amplifying frequency bins
	MaxGain float32

	// Floor prevents numerical instability in log mmse calculations
	Floor float32

	// NoiseOverestimation scales estimated noise psd before snr
	// values greater than one intentionally increase the assumed noise power
	// N(effective[k]) = NoiseOverestimation * N(mcra[k])
	// calculation to control suppression strength
	// increasing the value lowers the estimated SNR and produces stronger suppression
	NoiseOverestimation float32
}

// DefaultLogMMSEConfig returns the default log mmse configuration
func DefaultLogMMSEConfig() LogMMSEConfig {
	return LogMMSEConfig{
		MinGain:             0.05,
		MaxGain:             1,
		Floor:               1e-12,
		NoiseOverestimation: 1.25,
	}
}

// LogMMSE suppresses noise using the Ephraim Malah log spectral amplitude estimator
// reference
// https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Seltzer_Tashev_iwaenc08.pdf
//
// effective noise psd is
// N(effective[k]) = β * N[k]
// where β is NoiseOverestimation
// posteriori and priori snr values are calculated using the effective
// noise PSD:
//
//	γ[k] = P[k] / Neffective[k]
//
// log mmse auxiliary
//
//	v[k] = γ[k]ξ[k] / (1 + ξ[k])
//
// log mmse gain
//
//	G[k] = ξ[k] / (1 + ξ[k]) * exp(0.5 * E1(v[k]))
//
// clean spectrum
//
//	Y[k] = G[k] * X[k]
//
// where:
//
//	P[k] observed noisy power
//	N[k] estimated noise PSD from MCRA
//	γ[k] a-posteriori SNR
//	ξ[k] a-priori SNR
//	E1 exponential integral
//	G[k] suppression gain
type LogMMSE struct {
	cfg               LogMMSEConfig
	snrEstimator      snr.Processor
	cleanPower        []float32
	effectiveNoisePSD []float32
}

// NewLogMMSE creates a log mmse noise suppressor
func NewLogMMSE(cfg LogMMSEConfig, snrCfg snr.DecisionDirectedConfig) *LogMMSE {
	cfg = normalizeLogMMSEConfig(cfg)

	return &LogMMSE{
		cfg:          cfg,
		snrEstimator: snr.NewDecisionDirected(snrCfg),
	}
}

func normalizeLogMMSEConfig(cfg LogMMSEConfig) LogMMSEConfig {
	defaults := DefaultLogMMSEConfig()

	if cfg.MinGain < 0 || cfg.MinGain > 1 {
		cfg.MinGain = defaults.MinGain
	}

	if cfg.MaxGain <= 0 || cfg.MaxGain > 1 {
		cfg.MaxGain = defaults.MaxGain
	}

	if cfg.MinGain > cfg.MaxGain {
		cfg.MinGain = defaults.MinGain
	}

	if cfg.Floor <= 0 {
		cfg.Floor = defaults.Floor
	}

	if cfg.NoiseOverestimation <= 0 {
		cfg.NoiseOverestimation = defaults.NoiseOverestimation
	}

	return cfg
}

// Process applies log mmse suppression to one frequency domain frame
func (l *LogMMSE) Process(spectrum audio.Spectrum, noisePSD []float32) (audio.Spectrum, error) {
	if len(spectrum.Bins) == 0 {
		return audio.Spectrum{}, nil
	}

	if len(spectrum.Bins) != len(noisePSD) {
		return audio.Spectrum{}, fmt.Errorf(
			"spectrum contains %d bins, noise PSD contains %d",
			len(spectrum.Bins), len(noisePSD),
		)
	}

	power := spectrum.EnsurePower()
	l.prepareEffectiveNoisePSD(noisePSD)

	gamma, xi, err := l.snrEstimator.Process(power, l.effectiveNoisePSD)
	if err != nil {
		return audio.Spectrum{}, fmt.Errorf("estimate SNR: %w", err)
	}

	bins := l.applyGains(spectrum.Bins, power, gamma, xi)

	if err := l.snrEstimator.Update(l.cleanPower, l.effectiveNoisePSD); err != nil {
		return audio.Spectrum{}, fmt.Errorf("update SNR state: %w", err)
	}

	// copy clean power because the internal buffer is reused on the
	// next Process call
	outputPower := append([]float32(nil), l.cleanPower...)

	return audio.Spectrum{
		Bins:  bins,
		Power: outputPower,
	}, nil
}

func (l *LogMMSE) prepareEffectiveNoisePSD(noisePSD []float32) {
	if len(l.effectiveNoisePSD) != len(noisePSD) {
		l.effectiveNoisePSD = make([]float32, len(noisePSD))
	}

	for k, noisePower := range noisePSD {
		if noisePower < 0 || math.IsNaN(float64(noisePower)) || math.IsInf(float64(noisePower), 0) {
			noisePower = 0
		}

		l.effectiveNoisePSD[k] = noisePower * l.cfg.NoiseOverestimation
	}
}

func (l *LogMMSE) applyGains(
	bins []complex64,
	power, gamma, xi []float32,
) []complex64 {
	output := make([]complex64, len(bins))

	if len(l.cleanPower) != len(bins) {
		l.cleanPower = make([]float32, len(bins))
	}

	for k, bin := range bins {
		gain := logMMSEGain(gamma[k], xi[k], l.cfg.Floor)
		gain = clampLogMMSEGain(gain, l.cfg.MinGain, l.cfg.MaxGain)
		output[k] = bin * complex(gain, 0)
		l.cleanPower[k] = gain * gain * power[k]
	}

	return output
}

// Reset clears all log mmse and snr estimator state
func (l *LogMMSE) Reset() {
	l.cleanPower = nil
	l.effectiveNoisePSD = nil
	l.snrEstimator.Reset()
}

// logMMSEGain calculates the Log-MMSE gain for one frequency bin
func logMMSEGain(gamma float32, xi float32, floor float32) float32 {
	if gamma < 0 {
		gamma = 0
	}

	if xi < 0 {
		xi = 0
	}

	onePlusXi := 1 + xi

	v := gamma * xi / onePlusXi

	if v < floor {
		v = floor
	}

	e1 := expIntegralE1(float64(v))

	wienerGain := float64(xi / onePlusXi)

	gain := wienerGain * math.Exp(0.5*e1)

	if math.IsNaN(gain) || math.IsInf(gain, 0) {
		return 0
	}

	return float32(gain)
}

// clampLogMMSEGain limit gain to the configured attenuation range
func clampLogMMSEGain(gain float32, minGain float32, maxGain float32) float32 {
	if gain < minGain {
		return minGain
	}

	if gain > maxGain {
		return maxGain
	}

	return gain
}

// expIntegralE1 calculate the exponential integral E1(x)
//
// For 0 < x <= 1 a convergent power series is used
// For x > 1 a continued fraction is used
func expIntegralE1(x float64) float64 {
	const (
		eulerGamma = 0.5772156649015328606
		epsilon    = 1e-14
		fpMin      = 1e-300
		maxIter    = 100
	)

	if x <= 0 {
		return math.Inf(1)
	}

	if x <= 1 {
		result := -math.Log(x) - eulerGamma

		factor := 1.0

		for i := 1; i <= maxIter; i++ {
			factor *= -x / float64(i)

			delta := -factor / float64(i)

			result += delta

			if math.Abs(delta) < math.Abs(result)*epsilon {
				break
			}
		}

		return result
	}

	return expIntegralE1ContinuedFraction(
		x, epsilon, fpMin, maxIter,
	)
}

// expIntegralE1ContinuedFraction evaluate E1(x) for x greater than one
// using a continued fraction representation
// epsilon controls convergence while fpMin protects intermediate denominators
// from becoming too close to zero
func expIntegralE1ContinuedFraction(x float64, epsilon float64, fpMin float64, maxIter int) float64 {
	b := x + 1
	c := 1 / fpMin
	d := 1 / b
	h := d

	for i := 1; i <= maxIter; i++ {
		fi := float64(i)
		a := -(fi * fi)
		b += 2
		d = a*d + b

		if math.Abs(d) < fpMin {
			d = fpMin
		}

		c = b + a/c

		if math.Abs(c) < fpMin {
			c = fpMin
		}

		d = 1 / d

		delta := c * d

		h *= delta

		if math.Abs(delta-1) < epsilon {
			break
		}
	}

	return h * math.Exp(-x)
}
