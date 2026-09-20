package audio_test

import (
	"testing"

	"github.com/sghaida/noise-cancelation/audio"
)

func BenchmarkRMS160Samples(b *testing.B) {
	samples :=
		make([]float32, 160)

	for i := range samples {
		samples[i] = 0.5
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		audio.RMS(samples)
	}
}

func BenchmarkRMS960Samples(b *testing.B) {
	samples :=
		make([]float32, 960)

	for i := range samples {
		samples[i] = 0.5
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		audio.RMS(samples)
	}
}

func BenchmarkEnsurePower256FFT(b *testing.B) {
	spectrum := &audio.Spectrum{
		Bins: make([]complex64, 129),
	}

	for i := range spectrum.Bins {
		spectrum.Bins[i] =
			complex(
				float32(i),
				float32(i)/2,
			)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		spectrum.EnsurePower()
	}
}

func BenchmarkEnsurePower2048FFT(b *testing.B) {
	spectrum := &audio.Spectrum{
		Bins: make([]complex64, 1025),
	}

	for i := range spectrum.Bins {
		spectrum.Bins[i] =
			complex(
				float32(i),
				float32(i)/2,
			)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		spectrum.EnsurePower()
	}
}
