package interference

import (
	"testing"

	"github.com/sghaida/noise-cancelation/audio"
)

const tonalTransientBenchmarkBins = 129

func BenchmarkTonalTransientProcess(b *testing.B) {
	detector := NewTonalTransientDetector(
		DefaultTonalTransientConfig(),
	)

	previous := makeTonalTransientBenchmarkPower(
		tonalTransientBenchmarkBins,
		1,
	)

	current := makeTonalTransientBenchmarkPower(
		tonalTransientBenchmarkBins,
		1,
	)

	// Add a deterministic narrow foreground tone
	current[96] = 100

	// Initialize temporal state before benchmarking the hot path
	detector.Process(previous)

	b.ReportAllocs()
	b.SetBytes(int64(len(current) * 4))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		detector.Process(current)
	}
}

func BenchmarkTonalTransientProcessMovingTone(b *testing.B) {
	detector := NewTonalTransientDetector(
		DefaultTonalTransientConfig(),
	)

	previous := makeTonalTransientBenchmarkPower(
		tonalTransientBenchmarkBins,
		1,
	)

	current := makeTonalTransientBenchmarkPower(
		tonalTransientBenchmarkBins,
		1,
	)

	// Simulate a tonal peak moving by three FFT bins
	previous[93] = 100
	current[96] = 100

	detector.Process(previous)

	b.ReportAllocs()
	b.SetBytes(int64(len(current) * 4))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		detector.previousPower[93] = 100
		detector.previousPower[96] = 1

		detector.Process(current)
	}
}

func BenchmarkTonalTransientProcessFlatSpectrum(b *testing.B) {
	detector := NewTonalTransientDetector(
		DefaultTonalTransientConfig(),
	)

	power := makeTonalTransientBenchmarkPower(
		tonalTransientBenchmarkBins,
		1,
	)

	detector.Process(power)

	b.ReportAllocs()
	b.SetBytes(int64(len(power) * 4))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		detector.Process(power)
	}
}

func BenchmarkTonalTransientInitialization(b *testing.B) {
	power := makeTonalTransientBenchmarkPower(
		tonalTransientBenchmarkBins,
		1,
	)

	cfg := DefaultTonalTransientConfig()

	b.ReportAllocs()
	b.SetBytes(int64(len(power) * 4))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		detector := NewTonalTransientDetector(cfg)
		detector.Process(power)
	}
}

func BenchmarkTonalTransientApplyGain(b *testing.B) {
	spectrum := audio.Spectrum{
		Bins: make(
			[]complex64,
			tonalTransientBenchmarkBins,
		),
		Power: make(
			[]float32,
			tonalTransientBenchmarkBins,
		),
	}

	gain := make(
		[]float32,
		tonalTransientBenchmarkBins,
	)

	for k := range spectrum.Bins {
		spectrum.Bins[k] =
			complex(
				float32(k+1)*0.01,
				float32(k%7)*0.005,
			)

		spectrum.Power[k] =
			float32(k+1) *
				0.01

		gain[k] = 0.8
	}

	b.ReportAllocs()
	b.SetBytes(
		int64(
			len(spectrum.Bins) * 4,
		),
	)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := ApplyGain(
			&spectrum,
			gain,
		); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTonalProminence(b *testing.B) {
	detector := NewTonalTransientDetector(
		DefaultTonalTransientConfig(),
	)

	power := makeTonalTransientBenchmarkPower(
		tonalTransientBenchmarkBins,
		1,
	)

	power[96] = 100

	const centerBin = 96

	centerPower := power[centerBin]

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = detector.tonalProminence(
			power,
			centerBin,
			centerPower,
		)
	}
}

func BenchmarkHarmonicSupport(b *testing.B) {
	detector := NewTonalTransientDetector(
		DefaultTonalTransientConfig(),
	)

	power := makeTonalTransientBenchmarkPower(
		tonalTransientBenchmarkBins,
		1,
	)

	power[40] = 100
	power[20] = 30
	power[80] = 30

	const centerBin = 40

	centerPower := power[centerBin]

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = detector.calculateHarmonicSupport(
			power,
			centerBin,
			centerPower,
		)
	}
}

func BenchmarkMovementScore(b *testing.B) {
	detector := NewTonalTransientDetector(
		DefaultTonalTransientConfig(),
	)

	previous := makeTonalTransientBenchmarkPower(
		tonalTransientBenchmarkBins,
		1,
	)

	previous[93] = 100

	detector.Process(previous)

	const (
		currentBin   = 96
		currentPower = float32(100)
	)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = detector.calculateMovementScore(
			currentBin,
			currentPower,
		)
	}
}

// makeTonalTransientBenchmarkPower creates deterministic spectral power
// values without introducing random number generation into benchmarks
func makeTonalTransientBenchmarkPower(
	size int,
	base float32,
) []float32 {
	power := make(
		[]float32,
		size,
	)

	for k := range power {
		power[k] =
			base +
				float32(k%9)*0.01
	}

	return power
}
