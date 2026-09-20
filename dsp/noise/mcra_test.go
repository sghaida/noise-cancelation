package noise

import (
	"math"
	"testing"
	"time"
)

const mcraTestTolerance = 1e-5

// TestDefaultMCRAConfig verify the default MCRA configuration
func TestDefaultMCRAConfig(t *testing.T) {
	cfg := DefaultMCRAConfig()

	assertMCRAClose(t, cfg.Smoothing, 0.8)
	assertMCRAClose(t, cfg.SpeechSmoothing, 0.2)
	assertMCRAClose(t, cfg.NoiseSmoothing, 0.95)
	assertMCRAClose(t, cfg.RatioThreshold, 5)

	if cfg.WindowFrames != 50 {
		t.Fatalf("expected 50 window frames, got %d", cfg.WindowFrames)
	}

	if cfg.BaselineDuration != 5*time.Second {
		t.Fatalf(
			"expected baseline duration %v, got %v", 5*time.Second, cfg.BaselineDuration,
		)
	}

	if cfg.SampleRate != 8000 {
		t.Fatalf("expected sample rate 8000, got %d", cfg.SampleRate)
	}

	if cfg.HopSize != 128 {
		t.Fatalf("expected hop size 128, got %d", cfg.HopSize)
	}

	assertMCRAClose(t, cfg.MinProbabilityFrequency, 80)
	assertMCRAClose(t, cfg.Floor, 1e-12)
}

// TestNewMCRAEstimator verify estimator construction
func TestNewMCRAEstimator(t *testing.T) {
	cfg := MCRAConfig{
		Smoothing:               0.7,
		SpeechSmoothing:         0.3,
		NoiseSmoothing:          0.9,
		RatioThreshold:          4,
		WindowFrames:            25,
		BaselineDuration:        5 * time.Second,
		SampleRate:              8000,
		HopSize:                 128,
		MinProbabilityFrequency: 100,
		Floor:                   1e-10,
	}

	estimator := NewMCRAEstimator(cfg)

	if estimator == nil {
		t.Fatal("expected estimator")
	}

	if estimator.cfg != cfg {
		t.Fatalf("expected config %+v, got %+v", cfg, estimator.cfg)
	}

	if estimator.baselineFrameLimit != 313 {
		t.Fatalf(
			"expected 313 baseline frames, got %d", estimator.baselineFrameLimit,
		)
	}
}

// TestNewMCRAEstimatorDefaults verify invalid values use defaults
func TestNewMCRAEstimatorDefaults(t *testing.T) {
	defaults := DefaultMCRAConfig()

	cfg := MCRAConfig{
		Smoothing:               1,
		SpeechSmoothing:         1,
		NoiseSmoothing:          1,
		RatioThreshold:          1,
		WindowFrames:            0,
		BaselineDuration:        0,
		SampleRate:              0,
		HopSize:                 0,
		MinProbabilityFrequency: 0,
		Floor:                   0,
	}

	estimator := NewMCRAEstimator(cfg)

	if estimator.cfg != defaults {
		t.Fatalf(
			"expected defaults %+v, got %+v", defaults, estimator.cfg,
		)
	}
}

// TestCalculateBaselineFrames verify duration to STFT frame conversion
func TestCalculateBaselineFrames(t *testing.T) {
	tests := []struct {
		name       string
		duration   time.Duration
		sampleRate int
		hopSize    int
		want       int
	}{
		{
			name:       "three seconds",
			duration:   3 * time.Second,
			sampleRate: 8000,
			hopSize:    128,
			want:       188,
		},
		{
			name:       "five seconds",
			duration:   5 * time.Second,
			sampleRate: 8000,
			hopSize:    128,
			want:       313,
		},
		{
			name:       "ten seconds",
			duration:   10 * time.Second,
			sampleRate: 8000,
			hopSize:    128,
			want:       625,
		},
		{
			name:       "invalid duration",
			duration:   0,
			sampleRate: 8000,
			hopSize:    128,
			want:       0,
		},
		{
			name:       "invalid sample rate",
			duration:   5 * time.Second,
			sampleRate: 0,
			hopSize:    128,
			want:       0,
		},
		{
			name:       "invalid hop size",
			duration:   5 * time.Second,
			sampleRate: 8000,
			hopSize:    0,
			want:       0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateBaselineFrames(tt.duration, tt.sampleRate, tt.hopSize)

			if got != tt.want {
				t.Fatalf("expected %d frames, got %d", tt.want, got)
			}
		})
	}
}

// TestMCRAEstimatorEmptyInput verify empty input returns nil
func TestMCRAEstimatorEmptyInput(t *testing.T) {
	estimator := newTestMCRAEstimator()

	got := estimator.Process(nil)

	if got != nil {
		t.Fatalf("expected nil, got %v", got)
	}
}

// TestMCRAEstimatorRequiresBaseline verify processing does not initialize
// the estimator before baseline collection starts
func TestMCRAEstimatorRequiresBaseline(t *testing.T) {
	estimator := newTestMCRAEstimator()

	got := estimator.Process([]float32{10, 20, 30})

	assertMCRASliceClose(t, got, []float32{0, 0, 0})

	if estimator.initialized {
		t.Fatal("expected estimator to remain uninitialized")
	}
}

// TestMCRAEstimatorBaselineInitialization verify explicit baseline capture
func TestMCRAEstimatorBaselineInitialization(t *testing.T) {
	estimator := newTestMCRAEstimator()

	power := []float32{0.1, 0.2, 0.3}

	estimator.StartBaseline()

	got := estimator.Process(power)

	assertMCRASliceClose(t, got, power)

	if estimator.initialized {
		t.Fatal("expected baseline collection to remain active")
	}

	if estimator.baselineFrames != 1 {
		t.Fatalf(
			"expected 1 baseline frame, got %d", estimator.baselineFrames,
		)
	}

	estimator.FinishBaseline()

	assertMCRASliceClose(t, estimator.noise, power)
	assertMCRASliceClose(t, estimator.smoothed, power)
	assertMCRASliceClose(t, estimator.currentMin, power)
	assertMCRASliceClose(t, estimator.previousMin, power)

	assertMCRASliceClose(t, estimator.speechProb, []float32{0, 0, 0})

	if !estimator.initialized {
		t.Fatal("expected estimator to be initialized")
	}

	if estimator.baselineActive {
		t.Fatal("expected baseline to be inactive")
	}

	if estimator.framesInWindow != 0 {
		t.Fatalf(
			"expected 0 frames in window, got %d", estimator.framesInWindow,
		)
	}
}

// TestMCRAEstimatorBaselineAverage verify baseline PSD uses mean power
func TestMCRAEstimatorBaselineAverage(t *testing.T) {
	estimator := newTestMCRAEstimator()

	estimator.StartBaseline()

	estimator.Process([]float32{2, 4})

	got := estimator.Process([]float32{4, 8})

	assertMCRASliceClose(t, got, []float32{3, 6})

	estimator.FinishBaseline()

	assertMCRASliceClose(t, estimator.noise, []float32{3, 6})
}

// TestMCRAEstimatorEarlyBaselineFinish verify baseline collection can finish
// before the configured maximum duration
func TestMCRAEstimatorEarlyBaselineFinish(t *testing.T) {
	estimator := newTestMCRAEstimator()

	estimator.StartBaseline()

	estimator.Process([]float32{10})
	estimator.Process([]float32{20})
	estimator.Process([]float32{30})

	if estimator.baselineFrames != 3 {
		t.Fatalf("expected 3 baseline frames, got %d", estimator.baselineFrames)
	}

	if estimator.initialized {
		t.Fatal("expected estimator to still be collecting baseline")
	}

	estimator.FinishBaseline()

	assertMCRAClose(t, estimator.noise[0], 20)

	if !estimator.initialized {
		t.Fatal("expected estimator to be initialized")
	}

	if estimator.baselineActive {
		t.Fatal("expected baseline to be inactive")
	}
}

// TestMCRAEstimatorAutomaticBaselineFinish verify baseline collection stops
// automatically when the configured duration is reached
func TestMCRAEstimatorAutomaticBaselineFinish(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0.8,
			SpeechSmoothing:  0.2,
			NoiseSmoothing:   0.95,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 32 * time.Millisecond,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	if estimator.baselineFrameLimit != 2 {
		t.Fatalf("expected 2 baseline frames, got %d", estimator.baselineFrameLimit)
	}

	estimator.StartBaseline()

	estimator.Process([]float32{10})

	if estimator.initialized {
		t.Fatal("expected estimator to still be collecting baseline")
	}

	got := estimator.Process([]float32{20})

	assertMCRAClose(t, got[0], 15)

	if !estimator.initialized {
		t.Fatal("expected estimator to be initialized")
	}

	if estimator.baselineActive {
		t.Fatal("expected baseline to stop automatically")
	}
}

// TestMCRAEstimatorFinishBaselineWithoutFrames verify an empty baseline does
// not initialize MCRA
func TestMCRAEstimatorFinishBaselineWithoutFrames(t *testing.T) {
	estimator := newTestMCRAEstimator()

	estimator.StartBaseline()
	estimator.FinishBaseline()

	if estimator.initialized {
		t.Fatal("expected estimator to remain uninitialized")
	}

	if estimator.baselineActive {
		t.Fatal("expected baseline to be inactive")
	}
}

// TestMCRAEstimatorSmoothing verify spectral power smoothing
func TestMCRAEstimatorSmoothing(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0.8,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0.95,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Process([]float32{30})

	assertMCRAClose(t, estimator.smoothed[0], 14)
}

// TestMCRAEstimatorUpdatesMinimum verify lower power updates the minimum
func TestMCRAEstimatorUpdatesMinimum(t *testing.T) {
	estimator := newSimpleMCRAEstimator()

	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Process([]float32{5})

	assertMCRAClose(t, estimator.currentMin[0], 5)
}

// TestMCRAEstimatorKeepsMinimum verify higher power keeps the minimum
func TestMCRAEstimatorKeepsMinimum(t *testing.T) {
	estimator := newSimpleMCRAEstimator()

	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Process([]float32{20})

	assertMCRAClose(t, estimator.currentMin[0], 10)
}

// TestMCRAEstimatorNoSpeech verify ratio below threshold
func TestMCRAEstimatorNoSpeech(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0.5,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Process([]float32{40})

	assertMCRAClose(t, estimator.speechProb[0], 0)
}

// TestMCRAEstimatorDetectsSpeech verify ratio above threshold
func TestMCRAEstimatorDetectsSpeech(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0.5,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Process([]float32{100})

	assertMCRAClose(t, estimator.speechProb[0], 1)
}

// TestMCRAEstimatorSpeechProbability verify probability smoothing
func TestMCRAEstimatorSpeechProbability(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0.2,
			NoiseSmoothing:   0.95,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Process([]float32{100})

	assertMCRAClose(t, estimator.speechProb[0], 0.8)
}

// TestMCRAEstimatorNoiseUpdate verify normal noise adaptation
func TestMCRAEstimatorNoiseUpdate(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0.5,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	got := estimator.Process([]float32{20})

	assertMCRAClose(t, got[0], 15)
}

// TestMCRAEstimatorFreezesNoiseDuringSpeech verify speech protection
func TestMCRAEstimatorFreezesNoiseDuringSpeech(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0.5,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	got := estimator.Process([]float32{100})

	assertMCRAClose(t, estimator.speechProb[0], 1)

	assertMCRAClose(t, got[0], 10)
}

// TestMCRAEstimatorAdaptiveNoise verify partial speech protection
func TestMCRAEstimatorAdaptiveNoise(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0.2,
			NoiseSmoothing:   0.95,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	got := estimator.Process([]float32{100})

	assertMCRAClose(t, estimator.speechProb[0], 0.8)

	assertMCRAClose(t, got[0], 10.9)
}

// TestMCRAEstimatorTracksBinsIndependently verify per bin state
func TestMCRAEstimatorTracksBinsIndependently(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0.5,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10, 20, 30})

	got := estimator.Process([]float32{20, 10, 300})

	assertMCRAClose(t, got[0], 15)
	assertMCRAClose(t, got[1], 15)

	// The third bin is speech dominated and remains frozen
	assertMCRAClose(t, got[2], 30)
}

// TestMCRAEstimatorClampsNegativePower verify negative power handling
func TestMCRAEstimatorClampsNegativePower(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0.5,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0.5,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Process([]float32{-10})

	assertMCRAClose(t, estimator.smoothed[0], 5)
}

// TestMCRAEstimatorClampsNegativeBaselinePower verify baseline clamp
func TestMCRAEstimatorClampsNegativeBaselinePower(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0.8,
			SpeechSmoothing:  0.2,
			NoiseSmoothing:   0.95,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            0.01,
		},
	)

	estimator.StartBaseline()

	got := estimator.Process([]float32{-10})

	assertMCRAClose(t, got[0], 0.01)

	estimator.FinishBaseline()

	assertMCRAClose(t, estimator.noise[0], 0.01)
}

// TestMCRAEstimatorAppliesFloor verify the minimum power floor
func TestMCRAEstimatorAppliesFloor(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            0.01,
		},
	)

	estimator.StartBaseline()

	got := estimator.Process([]float32{0, 0.001, 0.1})

	assertMCRASliceClose(t, got, []float32{0.01, 0.01, 0.1})
}

// TestMCRAEstimatorRotatesAfterFirstMCRAFrame verify window rotation
func TestMCRAEstimatorRotatesAfterFirstMCRAFrame(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0.8,
			SpeechSmoothing:  0.2,
			NoiseSmoothing:   0.95,
			RatioThreshold:   5,
			WindowFrames:     1,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Process([]float32{10})

	assertMCRAClose(t, estimator.previousMin[0], 10)

	if !mcraIsPositiveInf(estimator.currentMin[0]) {
		t.Fatalf(
			"expected current minimum to be infinity, got %v", estimator.currentMin[0],
		)
	}

	if estimator.framesInWindow != 0 {
		t.Fatalf("expected 0 frames in window, got %d", estimator.framesInWindow)
	}
}

// TestMCRAEstimatorRotatesWindow verify normal window rotation
func TestMCRAEstimatorRotatesWindow(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0.95,
			RatioThreshold:   5,
			WindowFrames:     2,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Process([]float32{5})
	estimator.Process([]float32{6})

	assertMCRAClose(t, estimator.previousMin[0], 5)

	if !mcraIsPositiveInf(estimator.currentMin[0]) {
		t.Fatalf("expected current minimum to be infinity, got %v", estimator.currentMin[0])
	}

	if estimator.framesInWindow != 0 {
		t.Fatalf("expected 0 frames in window, got %d", estimator.framesInWindow)
	}
}

// TestMCRAEstimatorUsesPreviousMinimum verify boundary protection
func TestMCRAEstimatorUsesPreviousMinimum(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0.5,
			RatioThreshold:   5,
			WindowFrames:     2,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)

	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Process([]float32{10})
	estimator.Process([]float32{10})

	got := estimator.Process([]float32{100})

	assertMCRAClose(t, estimator.speechProb[0], 1)

	assertMCRAClose(t, got[0], 10)
}

// TestMCRAEstimatorReset verify all estimator state is cleared
func TestMCRAEstimatorReset(t *testing.T) {
	estimator := newTestMCRAEstimator()

	initializeTestMCRA(t, estimator, []float32{0.1, 0.2})

	estimator.Reset()
	assertMCRAResetSlices(t, estimator)
	assertMCRAResetCounters(t, estimator)
}

func assertMCRAResetSlices(t *testing.T, estimator *MCRAEstimator) {
	t.Helper()

	if estimator.smoothed != nil {
		t.Fatalf("expected nil smoothed state, got %v", estimator.smoothed)
	}

	if estimator.currentMin != nil {
		t.Fatalf("expected nil current minimum, got %v", estimator.currentMin)
	}

	if estimator.previousMin != nil {
		t.Fatalf("expected nil previous minimum, got %v", estimator.previousMin)
	}

	if estimator.speechProb != nil {
		t.Fatalf("expected nil speech probability, got %v", estimator.speechProb)
	}

	if estimator.noise != nil {
		t.Fatalf("expected nil noise state, got %v", estimator.noise)
	}

	if estimator.baselineSum != nil {
		t.Fatalf("expected nil baseline state, got %v", estimator.baselineSum)
	}
}

func assertMCRAResetCounters(t *testing.T, estimator *MCRAEstimator) {
	t.Helper()

	if estimator.baselineFrames != 0 {
		t.Fatalf("expected 0 baseline frames, got %d", estimator.baselineFrames)
	}

	if estimator.baselineActive {
		t.Fatal("expected baseline to be inactive")
	}

	if estimator.framesInWindow != 0 {
		t.Fatalf("expected 0 frames, got %d", estimator.framesInWindow)
	}

	if estimator.initialized {
		t.Fatal("expected estimator to be uninitialized")
	}
}

// TestMCRAEstimatorResetReinitializes verify baseline can be collected again
// after reset
func TestMCRAEstimatorResetReinitializes(t *testing.T) {
	estimator := newTestMCRAEstimator()
	initializeTestMCRA(t, estimator, []float32{10})

	estimator.Reset()

	initializeTestMCRA(t, estimator, []float32{20})

	assertMCRAClose(t, estimator.noise[0], 20)

	if !estimator.initialized {
		t.Fatal("expected estimator to be initialized")
	}
}

// TestMCRAEstimatorResizes verify bin count changes invalidate previous state
func TestMCRAEstimatorResizes(t *testing.T) {
	estimator := newTestMCRAEstimator()

	initializeTestMCRA(t, estimator, []float32{1, 2})
	got := estimator.Process([]float32{3, 4, 5})

	assertMCRASliceClose(t, got, []float32{0, 0, 0})

	if estimator.initialized {
		t.Fatal("expected estimator to require a new baseline after resize")
	}

	initializeTestMCRA(t, estimator, []float32{3, 4, 5})

	assertMCRASliceClose(t, estimator.noise, []float32{3, 4, 5})

	if len(estimator.smoothed) != 3 {
		t.Fatalf("expected 3 smoothed bins, got %d", len(estimator.smoothed))
	}

	if len(estimator.speechProb) != 3 {
		t.Fatalf("expected 3 speech probability bins, got %d", len(estimator.speechProb))
	}

	if len(estimator.noise) != 3 {
		t.Fatalf("expected 3 noise bins, got %d", len(estimator.noise))
	}
}

// TestMCRAEstimatorReusesOutput verify the output buffer is reused
func TestMCRAEstimatorReusesOutput(t *testing.T) {
	estimator := newTestMCRAEstimator()
	estimator.StartBaseline()
	first := estimator.Process([]float32{1, 2})
	second := estimator.Process([]float32{1, 2})

	if &first[0] != &second[0] {
		t.Fatal("expected estimator to reuse the output buffer")
	}
}

// TestMinMCRAPower verify the smaller power value is returned
func TestMinMCRAPower(t *testing.T) {
	assertMCRAClose(t, minMCRAPower(10, 20), 10)
	assertMCRAClose(t, minMCRAPower(20, 10), 10)
}

// TestMaxMCRAPower verify the larger power value is returned
func TestMaxMCRAPower(t *testing.T) {
	assertMCRAClose(t, maxMCRAPower(10, 20), 20)
	assertMCRAClose(t, maxMCRAPower(20, 10), 20)
}

// TestMCRAEstimatorPowerWeightedSpeechProbability verify speech probability
// is power weighted and excludes frequencies below the configured minimum
func TestMCRAEstimatorPowerWeightedSpeechProbability(t *testing.T) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:               0.8,
			SpeechSmoothing:         0.2,
			NoiseSmoothing:          0.95,
			RatioThreshold:          5,
			WindowFrames:            50,
			BaselineDuration:        5 * time.Second,
			SampleRate:              8000,
			HopSize:                 128,
			MinProbabilityFrequency: 80,
			Floor:                   1e-12,
		},
	)

	const bins = 129

	estimator.resize(bins)

	power := make([]float32, bins)

	// 31.25 Hz
	//
	// this bin has a large amount of energy but is below the configured
	// 80 Hz minimum and must therefore be ignored
	power[1] = 1000
	estimator.speechProb[1] = 1

	// 93.75 Hz
	//
	// speech probability = 1
	// power = 10
	power[3] = 10
	estimator.speechProb[3] = 1

	// 125 Hz
	//
	// speech probability = 0
	// power = 30
	power[4] = 30
	estimator.speechProb[4] = 0

	// only bins 3 and 4 contribute:
	//
	// Pspeech =
	//     (1 * 10) + (0 * 30)
	//     -------------------
	//          10 + 30
	//
	// = 0.25
	got := estimator.SpeechProbability(power)

	assertMCRAClose(t, got, 0.25)

	assertMCRAClose(t, estimator.NoiseProbability(power), 0.75)
}

// TestMCRAEstimatorSpeechProbabilityEmpty verify empty estimator state
func TestMCRAEstimatorSpeechProbabilityEmpty(t *testing.T) {
	estimator := newTestMCRAEstimator()

	got := estimator.SpeechProbability([]float32{1, 2})

	assertMCRAClose(t, got, 0)
}

// TestMCRAEstimatorSpeechProbabilitySizeMismatch verify mismatched spectra
func TestMCRAEstimatorSpeechProbabilitySizeMismatch(t *testing.T) {
	estimator := newTestMCRAEstimator()
	estimator.resize(129)

	got := estimator.SpeechProbability(make([]float32, 128))
	assertMCRAClose(t, got, 0)
}

// TestMCRAEstimatorSpeechProbabilitySingleBin verify invalid fft spectra
func TestMCRAEstimatorSpeechProbabilitySingleBin(t *testing.T) {
	estimator := newTestMCRAEstimator()
	estimator.resize(1)

	got := estimator.SpeechProbability([]float32{10})
	assertMCRAClose(t, got, 0)
}

// TestMCRAEstimatorSpeechProbabilityNoIncludedBins verify zero probability
// when no frequency bins are inside the configured range
func TestMCRAEstimatorSpeechProbabilityNoIncludedBins(
	t *testing.T,
) {
	estimator := NewMCRAEstimator(
		MCRAConfig{
			Smoothing:               0.8,
			SpeechSmoothing:         0.2,
			NoiseSmoothing:          0.95,
			RatioThreshold:          5,
			WindowFrames:            50,
			BaselineDuration:        5 * time.Second,
			SampleRate:              8000,
			HopSize:                 128,
			MinProbabilityFrequency: 5000,
			Floor:                   1e-12,
		},
	)

	const bins = 129
	estimator.resize(bins)

	power := make([]float32, bins)
	power[100] = 100
	estimator.speechProb[100] = 1

	got := estimator.SpeechProbability(power)
	assertMCRAClose(t, got, 0)
	assertMCRAClose(t, estimator.NoiseProbability(power), 1)
}

// newTestMCRAEstimator creates an estimator suitable for general tests
func newTestMCRAEstimator() *MCRAEstimator {
	return NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0.8,
			SpeechSmoothing:  0.2,
			NoiseSmoothing:   0.95,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)
}

// newSimpleMCRAEstimator creates an estimator without spectral smoothing
func newSimpleMCRAEstimator() *MCRAEstimator {
	return NewMCRAEstimator(
		MCRAConfig{
			Smoothing:        0,
			SpeechSmoothing:  0,
			NoiseSmoothing:   0.95,
			RatioThreshold:   5,
			WindowFrames:     50,
			BaselineDuration: 5 * time.Second,
			SampleRate:       8000,
			HopSize:          128,
			Floor:            1e-12,
		},
	)
}

// initializeTestMCRA captures a single-frame baseline and initializes MCRA
//
// Normal algorithm tests deliberately use one baseline frame so they test
// only the MCRA behavior under investigation rather than baseline averaging
func initializeTestMCRA(t *testing.T, estimator *MCRAEstimator, power []float32) {
	t.Helper()

	estimator.StartBaseline()
	got := estimator.Process(power)
	assertMCRASliceClose(t, got, power)
	estimator.FinishBaseline()

	if !estimator.initialized {
		t.Fatal("expected estimator to be initialized")
	}
}

// assertMCRASliceClose verifies two slices are approximately equal
func assertMCRASliceClose(t *testing.T, got []float32, want []float32) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("expected length %d, got %d", len(want), len(got))
	}

	for i := range want {
		if !mcraClose(got[i], want[i]) {
			t.Fatalf("index %d: expected %v, got %v", i, want[i], got[i])
		}
	}
}

// assertMCRAClose verifies two values are approximately equal
func assertMCRAClose(t *testing.T, got float32, want float32) {
	t.Helper()

	if !mcraClose(got, want) {
		t.Fatalf("expected %v, got %v", want, got)
	}
}

// mcraClose reports whether two float32 values are approximately equal
func mcraClose(a float32, b float32) bool {
	diff := math.Abs(float64(a - b))
	return diff <= mcraTestTolerance
}

// mcraIsPositiveInf reports whether a value is positive infinity
func mcraIsPositiveInf(value float32) bool {
	return math.IsInf(float64(value), 1)
}
