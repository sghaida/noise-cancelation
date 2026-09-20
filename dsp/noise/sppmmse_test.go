package noise

import (
	"math"
	"testing"
)

const sppMMSETestTolerance = 1e-5

// TestDefaultSPPMMSEConfig verify the default SPP MMSE configuration
func TestDefaultSPPMMSEConfig(t *testing.T) {
	cfg := DefaultSPPMMSEConfig()

	assertSPPMMSEClose(t, cfg.NoiseSmoothing, 0.8)
	assertSPPMMSEClose(t, cfg.SPPSmoothing, 0.9)
	assertSPPMMSEClose(t, cfg.SpeechPrior, 0.5)
	assertSPPMMSEClose(t, cfg.FixedPriorSNR, defaultSPPFixedPriorSNR)
	assertSPPMMSEClose(t, cfg.StagnationThreshold, 0.99)
	assertSPPMMSEClose(t, cfg.MaxSpeechProbability, 0.99)
	assertSPPMMSEClose(t, cfg.Floor, 1e-12)
}

// TestNewSPPMMSEEstimatorDefaults verify invalid configuration values use defaults
func TestNewSPPMMSEEstimatorDefaults(t *testing.T) {
	defaults := DefaultSPPMMSEConfig()

	estimator := NewSPPMMSEEstimator(
		SPPMMSEConfig{
			NoiseSmoothing:       0,
			SPPSmoothing:         1,
			SpeechPrior:          0,
			FixedPriorSNR:        0,
			StagnationThreshold:  2,
			MaxSpeechProbability: 0,
			Floor:                0,
		},
	)

	if estimator.cfg != defaults {
		t.Fatalf("expected config %+v, got %+v", defaults, estimator.cfg)
	}
}

// TestNewSPPMMSEEstimatorCustomConfig verify valid configuration values are preserved
func TestNewSPPMMSEEstimatorCustomConfig(t *testing.T) {
	cfg := SPPMMSEConfig{
		NoiseSmoothing:       0.7,
		SPPSmoothing:         0.8,
		SpeechPrior:          0.4,
		FixedPriorSNR:        20,
		StagnationThreshold:  0.95,
		MaxSpeechProbability: 0.9,
		Floor:                1e-10,
	}

	estimator := NewSPPMMSEEstimator(cfg)

	if estimator.cfg != cfg {
		t.Fatalf("expected config %+v, got %+v", cfg, estimator.cfg)
	}
}

// TestSPPMMSEFirstFrameInitializesNoise verify the first frame initializes
// the spectral noise PSD when no explicit baseline is used
func TestSPPMMSEFirstFrameInitializesNoise(t *testing.T) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	got := estimator.Process([]float32{2, 4, 8})

	assertSPPMMSESliceClose(t, got, []float32{2, 4, 8})

	if !estimator.initialized {
		t.Fatal("expected estimator to be initialized")
	}

	assertSPPMMSESliceClose(t, estimator.SpeechPresenceProbability(), []float32{0, 0, 0})
}

// TestSPPMMSESpeechPresenceProbability verifies the Gerkmann Hendriks
// posterior speech presence probability equation
func TestSPPMMSESpeechPresenceProbability(t *testing.T) {
	estimator := NewSPPMMSEEstimator(
		SPPMMSEConfig{
			NoiseSmoothing:       0.8,
			SPPSmoothing:         0.9,
			SpeechPrior:          0.5,
			FixedPriorSNR:        1,
			StagnationThreshold:  0.99,
			MaxSpeechProbability: 0.99,
			Floor:                1e-12,
		},
	)

	got := estimator.calculateSpeechPresenceProbability(10, 10)

	// γ = P / N = 10 / 10 = 1
	// q = 0.5
	// ξH1 = 1
	// p = 1 / (1 + 2 * exp(-0.5))
	//   = 0.4518627619
	want := float32(0.45186276187760605)

	assertSPPMMSEClose(t, got, want)
}

// TestSPPMMSESpeechProbabilityIncreasesWithObservedPower verify stronger
// observations relative to the noise PSD produce greater speech probability
func TestSPPMMSESpeechProbabilityIncreasesWithObservedPower(t *testing.T) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	low := estimator.calculateSpeechPresenceProbability(1, 10)

	high := estimator.calculateSpeechPresenceProbability(100, 10)

	if high <= low {
		t.Fatalf("expected high power probability %v to exceed low power probability %v", high, low)
	}
}

// TestSPPMMSENoiseUpdate verifies the conditional MMSE periodogram and
// recursive noise PSD smoothing
func TestSPPMMSENoiseUpdate(t *testing.T) {
	estimator := NewSPPMMSEEstimator(
		SPPMMSEConfig{
			NoiseSmoothing:       0.8,
			SPPSmoothing:         0.9,
			SpeechPrior:          0.5,
			FixedPriorSNR:        1,
			StagnationThreshold:  0.99,
			MaxSpeechProbability: 0.99,
			Floor:                1e-12,
		},
	)

	// initialize previous noise PSD to 10
	estimator.Process([]float32{10})

	got := estimator.Process([]float32{5})

	// γ = 5 / 10 = 0.5
	// p = 1 / (1 + 2 * exp(-0.25)) = 0.3909913152
	// Nmmse = (1-p) * 5 + p * 10= 6.954956576
	// N = 0.8 * 10 + 0.2 * Nmmse = 9.390991315
	want := float32(9.390991315159432)

	assertSPPMMSEClose(t, got[0], want)
}

// TestSPPMMSEStagnationLimit verify persistent high SPP activates the
// configured stagnation probability limit
func TestSPPMMSEStagnationLimit(t *testing.T) {
	estimator := NewSPPMMSEEstimator(
		SPPMMSEConfig{
			NoiseSmoothing:       0.8,
			SPPSmoothing:         0.5,
			SpeechPrior:          0.5,
			FixedPriorSNR:        1,
			StagnationThreshold:  0.6,
			MaxSpeechProbability: 0.7,
			Floor:                1e-12,
		},
	)

	estimator.Process([]float32{1})

	// Simulate a frequency bin whose smoothed SPP has remained high
	estimator.smoothedSpeechProb[0] = 1

	estimator.Process([]float32{100})

	assertSPPMMSEClose(t, estimator.speechProbability[0], 0.7)
}

// TestSPPMMSEBaselineAverage verifies baseline frames are averaged into the
// initial spectral noise PSD
func TestSPPMMSEBaselineAverage(t *testing.T) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	estimator.StartBaseline()

	first := estimator.Process([]float32{2, 4})

	assertSPPMMSESliceClose(t, first, []float32{2, 4})

	second := estimator.Process([]float32{4, 8})

	assertSPPMMSESliceClose(t, second, []float32{3, 6})

	if estimator.baselineFrames != 2 {
		t.Fatalf("expected 2 baseline frames, got %d", estimator.baselineFrames)
	}

	if estimator.initialized {
		t.Fatal("expected estimator to remain uninitialized during baseline")
	}

	estimator.FinishBaseline()

	if !estimator.initialized {
		t.Fatal("expected estimator to be initialized after baseline")
	}

	if estimator.baselineActive {
		t.Fatal("expected baseline collection to be inactive")
	}
}

// TestSPPMMSEBaselineKeepsSpeechProbabilityZero verify baseline collection
// does not perform speech presence estimation
func TestSPPMMSEBaselineKeepsSpeechProbabilityZero(t *testing.T) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	estimator.StartBaseline()

	estimator.Process([]float32{10, 20})

	assertSPPMMSESliceClose(t, estimator.SpeechPresenceProbability(), []float32{0, 0})
}

// TestSPPMMSEFinishBaselineWithoutFrames verify an empty baseline does not
// initialize the estimator
func TestSPPMMSEFinishBaselineWithoutFrames(t *testing.T) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	estimator.StartBaseline()
	estimator.FinishBaseline()

	if estimator.initialized {
		t.Fatal("expected estimator to remain uninitialized")
	}

	got := estimator.Process([]float32{7})

	assertSPPMMSEClose(t, got[0], 7)
}

// TestSPPMMSEFloor verifies negative and zero power values are limited by the
// configured numerical floor
func TestSPPMMSEFloor(t *testing.T) {
	cfg := DefaultSPPMMSEConfig()
	cfg.Floor = 0.1

	estimator := NewSPPMMSEEstimator(cfg)

	got := estimator.Process([]float32{-1, 0})

	assertSPPMMSEClose(t, got[0], 0.1)
	assertSPPMMSEClose(t, got[1], 0.1)
}

// TestSPPMMSEInvalidPower verify invalid floating point power values are
// replaced with the configured floor
func TestSPPMMSEInvalidPower(t *testing.T) {
	cfg := DefaultSPPMMSEConfig()
	cfg.Floor = 0.1

	estimator := NewSPPMMSEEstimator(cfg)

	got := estimator.Process(
		[]float32{float32(math.NaN()), float32(math.Inf(1))},
	)

	assertSPPMMSEClose(t, got[0], 0.1)
	assertSPPMMSEClose(t, got[1], 0.1)
}

// TestSPPMMSEEmptyPower verify empty input produces no spectral estimate
func TestSPPMMSEEmptyPower(t *testing.T) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	got := estimator.Process(nil)

	if got != nil {
		t.Fatalf("expected nil noise PSD, got %v", got)
	}
}

// TestSPPMMSEResize verify changing the number of bins resets learned
// spectral state and initializes from the new frame
func TestSPPMMSEResize(t *testing.T) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	estimator.Process([]float32{10})
	estimator.Process([]float32{20})

	got := estimator.Process([]float32{2, 4})

	assertSPPMMSESliceClose(t, got, []float32{2, 4})

	assertSPPMMSESliceClose(t, estimator.SpeechPresenceProbability(), []float32{0, 0})

	if len(estimator.smoothedSpeechProb) != 2 {
		t.Fatalf("expected 2 smoothed SPP bins, got %d", len(estimator.smoothedSpeechProb))
	}

	if len(estimator.baselineSum) != 2 {
		t.Fatalf("expected 2 baseline bins, got %d", len(estimator.baselineSum))
	}
}

// TestSPPMMSEReset verify all estimator state is cleared
func TestSPPMMSEReset(t *testing.T) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	estimator.StartBaseline()

	estimator.Process([]float32{10, 20})

	estimator.FinishBaseline()

	estimator.Process([]float32{12, 24})

	estimator.Reset()

	if estimator.noisePSD != nil {
		t.Fatalf("expected nil noise PSD, got %v", estimator.noisePSD)
	}

	if estimator.speechProbability != nil {
		t.Fatalf("expected nil speech probability, got %v", estimator.speechProbability)
	}

	if estimator.smoothedSpeechProb != nil {
		t.Fatalf("expected nil smoothed speech probability, got %v", estimator.smoothedSpeechProb)
	}

	if estimator.baselineSum != nil {
		t.Fatalf("expected nil baseline sum, got %v", estimator.baselineSum)
	}

	if estimator.baselineFrames != 0 {
		t.Fatalf("expected zero baseline frames, got %d", estimator.baselineFrames)
	}

	if estimator.baselineActive {
		t.Fatal("expected baseline collection to be inactive")
	}

	if estimator.initialized {
		t.Fatal("expected estimator to be uninitialized")
	}
}

// TestSPPMMSEProbabilityRange verify SPP remains within the probability range
func TestSPPMMSEProbabilityRange(t *testing.T) {
	estimator := NewSPPMMSEEstimator(DefaultSPPMMSEConfig())

	estimator.Process([]float32{1, 1, 1})

	estimator.Process([]float32{0.001, 1, 1000})

	for k, probability := range estimator.SpeechPresenceProbability() {
		if probability < 0 || probability > 1 {
			t.Fatalf("speech probability bin %d = %v, want value in [0, 1]", k, probability)
		}
	}
}

// assertSPPMMSEClose verify two float32 values are approximately equal
func assertSPPMMSEClose(t *testing.T, got float32, want float32) {
	t.Helper()

	diff := math.Abs(float64(got - want))

	if diff > sppMMSETestTolerance {
		t.Fatalf("expected %v, got %v", want, got)
	}
}

// assertSPPMMSESliceClose verify two float32 slices are approximately equal
func assertSPPMMSESliceClose(t *testing.T, got []float32, want []float32) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("expected %d values, got %d", len(want), len(got))
	}

	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))

		if diff > sppMMSETestTolerance {
			t.Fatalf("index %d expected %v, got %v", i, want[i], got[i])
		}
	}
}
