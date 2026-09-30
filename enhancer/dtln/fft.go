package dtln

import (
	"fmt"
	"math"

	stft "github.com/sghaida/noise-cancelation/dsp/stft"
)

// forwardSpectrum calculates the DTLN input magnitude spectrum
//
//	X[k] = FFT{x[n]}
//
//	|X[k]| = sqrt(Re(X[k])² + Im(X[k])²)
func forwardSpectrum(samples []float32, spectrum []complex64, magnitude []float32) error {
	if len(samples) != BlockLength {
		return fmt.Errorf("input contains %d samples, want %d", len(samples), BlockLength)
	}

	if len(spectrum) != BlockLength {
		return fmt.Errorf("spectrum contains %d bins, want %d", len(spectrum), BlockLength)
	}

	if len(magnitude) != SpectrumBins {
		return fmt.Errorf("magnitude contains %d bins, want %d", len(magnitude), SpectrumBins)
	}

	for i, sample := range samples {
		spectrum[i] = complex(sample, 0)
	}

	if err := stft.FFT(spectrum); err != nil {
		return fmt.Errorf("DTLN FFT: %w", err)
	}

	for k := range SpectrumBins {
		re := real(spectrum[k])
		im := imag(spectrum[k])

		magnitude[k] = float32(math.Sqrt(float64(re*re + im*im)))
	}

	return nil
}

// applyMask applies the model 1 magnitude mask while preserving phase
//
//	Y[k] = M[k] X[k]
func applyMask(spectrum []complex64, mask []float32) error {
	if len(spectrum) != BlockLength {
		return fmt.Errorf("spectrum contains %d bins, want %d", len(spectrum), BlockLength)
	}

	if len(mask) != SpectrumBins {
		return fmt.Errorf("mask contains %d bins, want %d", len(mask), SpectrumBins)
	}

	for k, gain := range mask {
		if math.IsNaN(float64(gain)) || math.IsInf(float64(gain), 0) {
			return fmt.Errorf("mask value %d is not finite", k)
		}

		spectrum[k] *= complex(gain, 0)
	}

	return nil
}

// inverseSpectrum reconstructs the real time domain frame
//
// # Upper frequency bins are complex conjugates of matching lower bins
//
// The reconstructed frame is
//
//	x[n] = IFFT{X[k]}
func inverseSpectrum(spectrum []complex64, output []float32) error {
	if len(spectrum) != BlockLength {
		return fmt.Errorf("spectrum contains %d bins, want %d", len(spectrum), BlockLength)
	}

	if len(output) != BlockLength {
		return fmt.Errorf("output contains %d samples, want %d", len(output), BlockLength)
	}

	spectrum[0] = complex(real(spectrum[0]), 0)
	spectrum[BlockLength/2] = complex(real(spectrum[BlockLength/2]), 0)

	for k := 1; k < BlockLength/2; k++ {
		value := spectrum[k]
		spectrum[BlockLength-k] = complex(real(value), -imag(value))
	}

	if err := stft.IFFT(spectrum); err != nil {
		return fmt.Errorf("DTLN IFFT: %w", err)
	}

	for i := range BlockLength {
		output[i] = real(spectrum[i])
	}

	return nil
}
