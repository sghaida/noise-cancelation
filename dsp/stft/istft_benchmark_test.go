package stft

import (
	"math"
	"testing"

	"github.com/sghaida/noise-cancelation/audio"
)

const (
	istftBenchmarkFFTSize = 256
	istftBenchmarkHopSize = 128
)

// BenchmarkISTFTProcess measures steady-state inverse STFT processing
// for one 256-point spectrum with 50 percent overlap
func BenchmarkISTFTProcess(b *testing.B) {
	synthesizer, err := NewSynthesizer(istftBenchmarkFFTSize, istftBenchmarkHopSize)
	if err != nil {
		b.Fatalf("create ISTFT synthesizer: %v", err)
	}

	spectrum := makeBenchmarkSpectrum(istftBenchmarkFFTSize)

	// Warm up the synthesizer before benchmark measurements begin
	if _, err := synthesizer.Process(spectrum); err != nil {
		b.Fatalf("warm up ISTFT synthesizer: %v", err)
	}

	synthesizer.Reset()

	b.ReportAllocs()

	// The input contains one complex64 value for each one-sided FFT bin
	b.SetBytes(int64(len(spectrum.Bins) * 8))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if _, err := synthesizer.Process(spectrum); err != nil {
			b.Fatalf("process ISTFT spectrum: %v", err)
		}
	}
}

// BenchmarkISTFTInitialization measures synthesizer construction
func BenchmarkISTFTInitialization(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		synthesizer, err := NewSynthesizer(istftBenchmarkFFTSize, istftBenchmarkHopSize)
		if err != nil {
			b.Fatalf("create ISTFT synthesizer: %v", err)
		}

		if synthesizer == nil {
			b.Fatal("expected ISTFT synthesizer")
		}
	}
}

// makeBenchmarkSpectrum creates a deterministic one-sided frequency spectrum
// containing several frequency components
func makeBenchmarkSpectrum(fftSize int) audio.Spectrum {
	binCount := fftSize/2 + 1
	bins := make([]complex64, binCount)

	for k := range bins {
		phase := 2 * math.Pi * float64(k) / float64(binCount)

		bins[k] = complex(float32(math.Cos(phase)), float32(math.Sin(phase)))
	}

	return audio.Spectrum{
		Bins: bins,
	}
}
