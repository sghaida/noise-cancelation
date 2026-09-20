package snr

import (
	"math"
	"testing"
)

const snrTestTolerance = 1e-5

// TestDefaultDecisionDirectedConfig verify the default configuration
func TestDefaultDecisionDirectedConfig(t *testing.T) {
	cfg := DefaultDecisionDirectedConfig()

	assertSNRClose(t, cfg.Alpha, 0.98)
	assertSNRClose(t, cfg.Floor, 1e-12)
}

// TestNewDecisionDirectedDefaults verify invalid configuration uses defaults
func TestNewDecisionDirectedDefaults(t *testing.T) {
	defaults := DefaultDecisionDirectedConfig()
	estimator := NewDecisionDirected(DecisionDirectedConfig{Alpha: 1, Floor: 0})

	if estimator.cfg != defaults {
		t.Fatalf("expected config %+v, got %+v", defaults, estimator.cfg)
	}
}

// TestDecisionDirectedFirstFrame verify instantaneous SNR initialization
func TestDecisionDirectedFirstFrame(t *testing.T) {
	estimator := NewDecisionDirected(DefaultDecisionDirectedConfig())

	gamma, xi, err := estimator.Process([]float32{20}, []float32{10})
	if err != nil {
		t.Fatalf("process SNR: %v", err)
	}

	// γ = 20 / 10 = 2
	assertSNRClose(t, gamma[0], 2)

	// first frame has no previous clean power
	//
	// ξ = max(γ - 1, 0)
	//   = 1
	assertSNRClose(t, xi[0], 1)
}

// TestDecisionDirectedNoiseDominated verify negative instantaneous SNR
// is clamped to zero
func TestDecisionDirectedNoiseDominated(t *testing.T) {
	estimator := NewDecisionDirected(DefaultDecisionDirectedConfig())

	gamma, xi, err := estimator.Process([]float32{5}, []float32{10})
	if err != nil {
		t.Fatalf("process SNR: %v", err)
	}

	assertSNRClose(t, gamma[0], 0.5)
	assertSNRClose(t, xi[0], 0)
}

// TestDecisionDirectedUpdate verify previous clean power contributes to
// the next a-priori SNR estimate
func TestDecisionDirectedUpdate(t *testing.T) {
	estimator := NewDecisionDirected(DecisionDirectedConfig{Alpha: 0.98, Floor: 1e-12})

	_, _, err := estimator.Process([]float32{20}, []float32{10})
	if err != nil {
		t.Fatalf("process first frame: %v", err)
	}

	err = estimator.Update([]float32{5}, []float32{10})
	if err != nil {
		t.Fatalf("update clean power: %v", err)
	}

	gamma, xi, err := estimator.Process([]float32{30}, []float32{10})
	if err != nil {
		t.Fatalf("process second frame: %v", err)
	}

	// γ = 30 / 10 = 3
	assertSNRClose(t, gamma[0], 3)

	// previous clean SNR:
	// 5 / 10 = 0.5
	// instantaneous:
	// max(3 - 1, 0) = 2
	// ξ = 0.98 * 0.5 + 0.02 * 2 = 0.53
	assertSNRClose(t, xi[0], 0.53)
}

// TestDecisionDirectedUsesPreviousFrameNoise verifies historical clean SNR
// is not recalculated using the current frame noise PSD
func TestDecisionDirectedUsesPreviousFrameNoise(
	t *testing.T,
) {
	estimator := NewDecisionDirected(
		DecisionDirectedConfig{
			Alpha: 0.98,
			Floor: 1e-12,
		},
	)

	// Frame 1
	//
	// clean power = 10
	// noise PSD   = 5
	//
	// previous clean SNR = 10 / 5 = 2
	_, _, err := estimator.Process(
		[]float32{20},
		[]float32{5},
	)
	if err != nil {
		t.Fatalf(
			"process first frame: %v",
			err,
		)
	}

	err = estimator.Update(
		[]float32{10},
		[]float32{5},
	)
	if err != nil {
		t.Fatalf(
			"update first frame: %v",
			err,
		)
	}

	// Frame 2 has a very different noise PSD
	// γ = 2 / 1 = 2
	// instantaneous ξ = max(2 - 1, 0) = 1
	// previous clean SNR must remain: 10 / 5 = 2
	// ξ = 0.98 * 2 + 0.02 * 1 = 1.98
	_, xi, err := estimator.Process([]float32{2}, []float32{1})
	if err != nil {
		t.Fatalf("process second frame: %v", err)
	}

	assertSNRClose(t, xi[0], 1.98)
}

// TestDecisionDirectedFloor verify zero noise power remains numerically safe
func TestDecisionDirectedFloor(t *testing.T) {
	estimator := NewDecisionDirected(DecisionDirectedConfig{Alpha: 0.98, Floor: 0.1})

	gamma, _, err := estimator.Process([]float32{1}, []float32{0})
	if err != nil {
		t.Fatalf("process SNR: %v", err)
	}

	assertSNRClose(t, gamma[0], 10)
}

// TestDecisionDirectedSizeMismatch verify mismatched inputs return an error
func TestDecisionDirectedSizeMismatch(t *testing.T) {
	estimator := NewDecisionDirected(DefaultDecisionDirectedConfig())

	_, _, err := estimator.Process([]float32{1, 2}, []float32{1})

	if err == nil {
		t.Fatal("expected size mismatch error")
	}
}

// TestDecisionDirectedUpdateSizeMismatch verify invalid clean power size
func TestDecisionDirectedUpdateSizeMismatch(t *testing.T) {
	estimator := NewDecisionDirected(DefaultDecisionDirectedConfig())

	_, _, err := estimator.Process([]float32{1, 2}, []float32{1, 1})
	if err != nil {
		t.Fatalf("process SNR: %v", err)
	}

	err = estimator.Update([]float32{1}, []float32{1, 1})

	if err == nil {
		t.Fatal("expected update size mismatch error")
	}
}

func TestDecisionDirectedUpdateNoisePSDSizeMismatch(t *testing.T) {
	estimator := NewDecisionDirected(DefaultDecisionDirectedConfig())

	_, _, err := estimator.Process([]float32{1, 2}, []float32{1, 1})
	if err != nil {
		t.Fatalf("process SNR: %v", err)
	}

	err = estimator.Update([]float32{1, 2}, []float32{1})

	if err == nil {
		t.Fatal("expected noise PSD size mismatch error")
	}
}

// TestDecisionDirectedResize verify bin count changes reset previous state
func TestDecisionDirectedResize(t *testing.T) {
	estimator := NewDecisionDirected(DefaultDecisionDirectedConfig())

	_, _, err := estimator.Process([]float32{10}, []float32{5})
	if err != nil {
		t.Fatalf("process first frame: %v", err)
	}

	if err := estimator.Update([]float32{4}, []float32{5}); err != nil {
		t.Fatalf("update clean power: %v", err)
	}

	_, xi, err := estimator.Process([]float32{2, 4}, []float32{2, 2})
	if err != nil {
		t.Fatalf("process resized frame: %v", err)
	}

	if estimator.initialized {
		t.Fatal("expected resize to reset initialization")
	}

	// bin 0:
	// γ = 2 / 2 = 1
	// ξ = max(1 - 1, 0) = 0
	assertSNRClose(t, xi[0], 0)
	// bin 1:
	//
	// γ = 4 / 2 = 2
	// ξ = max(2 - 1, 0) = 1
	assertSNRClose(t, xi[1], 1)
}

// TestDecisionDirectedReset verify all estimator state is cleared
func TestDecisionDirectedReset(t *testing.T) {
	estimator := NewDecisionDirected(DefaultDecisionDirectedConfig())

	_, _, err := estimator.Process([]float32{10}, []float32{5})
	if err != nil {
		t.Fatalf("process SNR: %v", err)
	}

	if err := estimator.Update([]float32{5}, []float32{5}); err != nil {
		t.Fatalf("update clean power: %v", err)
	}

	estimator.Reset()

	if estimator.gamma != nil {
		t.Fatalf("expected nil gamma, got %v", estimator.gamma)
	}

	if estimator.xi != nil {
		t.Fatalf("expected nil xi, got %v", estimator.xi)
	}

	if estimator.previousCleanPower != nil {
		t.Fatalf(
			"expected nil previous clean power, got %v", estimator.previousCleanPower,
		)
	}

	if estimator.initialized {
		t.Fatal("expected estimator to be uninitialized")
	}
}

// assertSNRClose verify two float32 values are approximately equal
func assertSNRClose(t *testing.T, got float32, want float32) {
	t.Helper()

	diff := math.Abs(
		float64(got - want),
	)

	if diff > snrTestTolerance {
		t.Fatalf("expected %v, got %v", want, got)
	}
}
