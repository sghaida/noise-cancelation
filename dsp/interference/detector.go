package interference

import (
	"fmt"

	"github.com/sghaida/noise-cancelation/audio"
)

// Result contains the per bin output of an interference detector
// returned slices may be reused by the detector on subsequent Process calls
type Result struct {
	// Score contains the estimated foreground interference probability
	// values are in the range [0, 1]
	Score []float32

	// Gain contains smoothed gain that should be applied to each frequency bin
	// values are in the range [MinGain, 1]
	Gain []float32

	// TonalProminenceDB contains local peak prominence in dB
	TonalProminenceDB []float32

	// SpectralFluxDB contains positive frame to frame spectral rise in dB
	SpectralFluxDB []float32

	// HarmonicSupport contains the amount of speech like harmonic support
	// values are in the range [0, 1]
	HarmonicSupport []float32

	// MovementScore contains the normalized movement of a spectral peak
	// values are in the range [0, 1]
	MovementScore []float32
}

// Detector identifies foreground interference from spectral power
type Detector interface {
	Process(power []float32) Result
	Reset()
}

// ApplyGain applies an interference gain mask to a spectrum
// complex spectrum is scaled by:
//
//	Y[k] = G[k] * X[k]
//
// power spectrum is scaled by:
//
//	Pout[k] = G[k]² * Pin[k]
func ApplyGain(spectrum *audio.Spectrum, gain []float32) error {
	if spectrum == nil {
		return nil
	}

	if len(spectrum.Bins) != len(gain) {
		return fmt.Errorf("spectrum contains %d bins, gain contains %d", len(spectrum.Bins), len(gain))
	}

	if len(spectrum.Power) != 0 && len(spectrum.Power) != len(spectrum.Bins) {
		return fmt.Errorf(
			"spectrum contains %d bins, power contains %d bins",
			len(spectrum.Bins), len(spectrum.Power),
		)
	}

	hasPower := len(spectrum.Power) == len(spectrum.Bins)

	for k, value := range gain {
		value = clampFloat32(value, 0, 1)

		spectrum.Bins[k] *= complex(value, 0)

		if hasPower {
			spectrum.Power[k] *= value * value
		}
	}

	return nil
}
