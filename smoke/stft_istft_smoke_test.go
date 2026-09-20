//go:build smoke
// +build smoke

package smoke

import (
	"math"
	"testing"

	"github.com/sghaida/noise-cancelation/dsp/stft"
)

const (
	stftOutputPath = "output/stft_istft.out.wav"

	stftFFTSize = 256
	stftHopSize = 128

	minimumRoundTripSNR = 60.0
	maximumSampleError  = 1e-3
)

// TestSTFTISTFTSmoke processes the real 30-second fixture through
// STFT followed immediately by ISTFT without modifying the spectrum
func TestSTFTISTFTSmoke(t *testing.T) {
	sampleRate, samples, durationSeconds := readTestFixture(t)

	original := append([]float32(nil), samples...)

	analyzer, err := stft.New(stftFFTSize, stftHopSize)
	if err != nil {
		t.Fatalf("create STFT analyzer: %v", err)
	}

	synthesizer, err := stft.NewSynthesizer(stftFFTSize, stftHopSize)
	if err != nil {
		t.Fatalf("create ISTFT synthesizer: %v", err)
	}

	frameSize := sampleRate * frameDuration / 1000

	var (
		output        []float32
		spectrumCount int
	)

	for start := 0; start < len(samples); start += frameSize {
		end := start + frameSize

		if end > len(samples) {
			end = len(samples)
		}

		spectra, err := analyzer.Process(samples[start:end])
		if err != nil {
			t.Fatalf("STFT frame starting at sample %d: %v", start, err)
		}

		for _, spectrum := range spectra {
			assertSpectrumFinite(t, spectrum.Bins)

			frame, err := synthesizer.Process(spectrum)
			if err != nil {
				t.Fatalf("ISTFT spectrum %d: %v", spectrumCount, err)
			}

			output = append(output, frame...)
			spectrumCount++
		}
	}

	finalSpectra, err := analyzer.Flush()
	if err != nil {
		t.Fatalf("flush STFT analyzer: %v", err)
	}

	for _, spectrum := range finalSpectra {
		assertSpectrumFinite(t, spectrum.Bins)

		frame, err := synthesizer.Process(spectrum)
		if err != nil {
			t.Fatalf("ISTFT flushed spectrum %d: %v", spectrumCount, err)
		}

		output = append(output, frame...)
		spectrumCount++
	}

	output = append(output, synthesizer.Flush()...)

	if spectrumCount == 0 {
		t.Fatal("STFT produced no spectra")
	}

	assertFiniteSamples(t, output)

	if len(output) < len(original) {
		t.Fatalf(
			"reconstructed sample count = %d, want at least %d", len(output), len(original),
		)
	}

	// Flush can produce samples beyond the original duration because
	// the final incomplete STFT frame is zero-padded
	output = output[:len(original)]

	snrDB, maxError := roundTripQuality(original, output, stftHopSize)

	if snrDB < minimumRoundTripSNR {
		t.Fatalf(
			"round-trip SNR = %.2f dB, want at least %.2f dB",
			snrDB, minimumRoundTripSNR,
		)
	}

	if maxError > maximumSampleError {
		t.Fatalf(
			"maximum error = %.8f, want <= %.8f",
			maxError, maximumSampleError,
		)
	}

	if err := writePCM16MonoWAV(stftOutputPath, sampleRate, output); err != nil {
		t.Fatalf("write STFT/ISTFT output: %v", err)
	}

	t.Logf(
		"processed %.2f seconds in %d-sample input frames",
		durationSeconds, frameSize,
	)

	t.Logf(
		"STFT size = %d, hop = %d, spectra = %d",
		stftFFTSize, stftHopSize, spectrumCount,
	)

	t.Logf(
		"round-trip SNR = %.2f dB, max error = %.8f", snrDB, maxError,
	)

	t.Logf("STFT/ISTFT output written to %s", stftOutputPath)

	analyzer.Reset()
	synthesizer.Reset()
}

// assertSpectrumFinite verifies that FFT output contains no NaN or
// infinite real or imaginary values
func assertSpectrumFinite(t *testing.T, bins []complex64) {
	t.Helper()

	for i, bin := range bins {
		realPart := float64(real(bin))
		imaginaryPart := float64(imag(bin))

		if math.IsNaN(realPart) || math.IsNaN(imaginaryPart) {
			t.Fatalf("spectrum bin %d contains NaN", i)
		}

		if math.IsInf(realPart, 0) || math.IsInf(imaginaryPart, 0) {
			t.Fatalf("spectrum bin %d contains infinity", i)
		}
	}
}

// roundTripQuality measures reconstruction quality in the stable
// overlap region while ignoring Hann boundary effects
func roundTripQuality(
	before []float32, after []float32, boundary int,
) (snrDB float64, maxError float64) {
	start := boundary
	end := len(before) - boundary

	var (
		signalPower float64
		errorPower  float64
	)

	for i := start; i < end; i++ {
		expected := float64(before[i])
		actual := float64(after[i])
		difference := actual - expected

		signalPower += expected * expected
		errorPower += difference * difference

		absoluteError := math.Abs(difference)

		if absoluteError > maxError {
			maxError = absoluteError
		}
	}

	if errorPower == 0 {
		return math.Inf(1), maxError
	}

	if signalPower == 0 {
		return math.Inf(-1), maxError
	}

	snrDB = 10 * math.Log10(signalPower/errorPower)

	return snrDB, maxError
}
