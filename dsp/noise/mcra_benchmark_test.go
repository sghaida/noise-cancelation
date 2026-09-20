package noise

import (
	"testing"
	"time"
)

const (
	mcraBenchmarkSampleRate = 8000
	mcraBenchmarkFFTSize    = 256
	mcraBenchmarkHopSize    = 128
	mcraBenchmarkBins       = mcraBenchmarkFFTSize/2 + 1
)

// BenchmarkMCRAProcess measures steady-state MCRA processing after
// the initial noise baseline has been collected
func BenchmarkMCRAProcess(b *testing.B) {
	cfg := MCRAConfig{
		Smoothing:               0.8,
		SpeechSmoothing:         0.2,
		NoiseSmoothing:          0.95,
		RatioThreshold:          5,
		WindowFrames:            50,
		BaselineDuration:        5 * time.Second,
		SampleRate:              mcraBenchmarkSampleRate,
		HopSize:                 mcraBenchmarkHopSize,
		MinProbabilityFrequency: 80,
		Floor:                   1e-12,
	}

	estimator := NewMCRAEstimator(cfg)

	baseline := makeBenchmarkPowerFrame(mcraBenchmarkBins, 0.01)

	// Initialize MCRA using a realistic five-second baseline
	estimator.StartBaseline()

	for i := 0; i < estimator.baselineFrameLimit; i++ {
		estimator.Process(baseline)
	}

	// Create several frames so the benchmark does not repeatedly process
	// exactly the same spectrum
	frames := make([][]float32, 64)

	for i := range frames {
		frames[i] = makeBenchmarkPowerFrame(mcraBenchmarkBins, float32(i+1)*0.002)
	}

	b.ReportAllocs()

	// Each MCRA frame contains one float32 power value per frequency bin
	b.SetBytes(int64(mcraBenchmarkBins * 4))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		frame := frames[i%len(frames)]
		estimator.Process(frame)
	}
}

// BenchmarkMCRASpeechProbability measures the cost of calculating the
// power-weighted speech probability for one spectrum
func BenchmarkMCRASpeechProbability(b *testing.B) {
	cfg := DefaultMCRAConfig()

	cfg.SampleRate = mcraBenchmarkSampleRate
	cfg.HopSize = mcraBenchmarkHopSize
	cfg.MinProbabilityFrequency = 80

	estimator := NewMCRAEstimator(cfg)

	power := makeBenchmarkPowerFrame(mcraBenchmarkBins, 0.02)

	estimator.resize(mcraBenchmarkBins)

	for i := range estimator.speechProb {
		estimator.speechProb[i] = float32(i%10) / 10
	}

	b.ReportAllocs()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		estimator.SpeechProbability(power)
	}
}

// BenchmarkMCRABaseline measures baseline PSD accumulation
func BenchmarkMCRAInitialization(b *testing.B) {
	cfg := DefaultMCRAConfig()

	cfg.SampleRate = mcraBenchmarkSampleRate
	cfg.HopSize = mcraBenchmarkHopSize
	cfg.MinProbabilityFrequency = 80

	power := makeBenchmarkPowerFrame(mcraBenchmarkBins, 0.01)

	b.ReportAllocs()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		estimator := NewMCRAEstimator(cfg)
		estimator.StartBaseline()
		estimator.Process(power)
	}
}

func BenchmarkMCRABaselineProcess(b *testing.B) {
	cfg := DefaultMCRAConfig()

	cfg.SampleRate = mcraBenchmarkSampleRate
	cfg.HopSize = mcraBenchmarkHopSize
	cfg.BaselineDuration = 5 * time.Second
	cfg.MinProbabilityFrequency = 80

	estimator := NewMCRAEstimator(cfg)

	power := makeBenchmarkPowerFrame(mcraBenchmarkBins, 0.01)
	estimator.StartBaseline()

	// Allocate internal buffers before benchmarking
	estimator.Process(power)

	b.ReportAllocs()
	b.SetBytes(int64(mcraBenchmarkBins * 4))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		estimator.Process(power)
	}
}

func BenchmarkMCRAUpdateBaseline(b *testing.B) {
	cfg := DefaultMCRAConfig()

	estimator := NewMCRAEstimator(cfg)
	power := makeBenchmarkPowerFrame(mcraBenchmarkBins, 0.01)
	estimator.resize(mcraBenchmarkBins)

	b.ReportAllocs()
	b.SetBytes(int64(mcraBenchmarkBins * 4))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		estimator.updateBaseline(power)
	}
}

// makeBenchmarkPowerFrame creates deterministic power values for one
// one-sided FFT spectrum
func makeBenchmarkPowerFrame(bins int, base float32) []float32 {
	power := make([]float32, bins)

	for i := range power {
		power[i] = base + float32((i%17)+1)*0.001
	}

	return power
}
