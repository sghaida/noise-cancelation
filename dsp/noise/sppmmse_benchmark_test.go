package noise

import "testing"

const sppMMSEBenchmarkBins = 129

func BenchmarkSPPMMSEProcess(b *testing.B) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	initialPower := makeSPPMMSEBenchmarkPower(sppMMSEBenchmarkBins, 0.01)
	power := makeSPPMMSEBenchmarkPower(sppMMSEBenchmarkBins, 0.02)

	estimator.Process(initialPower)

	b.ReportAllocs()
	b.SetBytes(int64(len(power) * 4))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		estimator.Process(power)
	}
}

func BenchmarkSPPMMSEBaselineUpdate(b *testing.B) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	power := makeSPPMMSEBenchmarkPower(sppMMSEBenchmarkBins, 0.01)

	estimator.StartBaseline()

	// Allocate all baseline state before entering the measured section
	estimator.Process(power)

	b.ReportAllocs()
	b.SetBytes(int64(len(power) * 4))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		estimator.Process(power)
	}
}

func BenchmarkSPPMMSESpeechPresenceProbability(b *testing.B) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	const (
		observedPower = float32(0.1)
		noisePower    = float32(0.01)
	)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = estimator.calculateSpeechPresenceProbability(observedPower, noisePower)
	}
}

func BenchmarkSPPMMSEInitialization(b *testing.B) {
	power := makeSPPMMSEBenchmarkPower(sppMMSEBenchmarkBins, 0.01)
	cfg := DefaultSPPMMSEConfig()

	b.ReportAllocs()
	b.SetBytes(int64(len(power) * 4))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		estimator := NewSPPMMSEEstimator(cfg)
		estimator.Process(power)
	}
}

func makeSPPMMSEBenchmarkPower(size int, base float32) []float32 {
	power := make([]float32, size)

	for k := range power {
		// Produce deterministic spectral variation without random number generation
		power[k] = base + float32(k%11)*0.001
	}

	return power
}
