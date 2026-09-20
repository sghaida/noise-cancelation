package suppressor

import (
	"errors"
	"math"
	"testing"

	"github.com/sghaida/noise-cancelation/audio"
	"github.com/sghaida/noise-cancelation/dsp/snr"
)

const logMMSETestTolerance = 1e-5

// TestDefaultLogMMSEConfig verify the default suppressor configuration
func TestDefaultLogMMSEConfig(t *testing.T) {
	cfg := DefaultLogMMSEConfig()

	assertLogMMSEClose(t, cfg.MinGain, 0.05)
	assertLogMMSEClose(t, cfg.MaxGain, 1)
	assertLogMMSEClose(t, cfg.Floor, 1e-12)
}

// TestNewLogMMSEInvalidConfig verify invalid configuration values are
// replaced by safe defaults
func TestNewLogMMSEInvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  LogMMSEConfig
		want LogMMSEConfig
	}{
		{
			name: "minimum gain below zero",
			cfg: LogMMSEConfig{
				MinGain: -0.1,
				MaxGain: 1,
				Floor:   1e-12,
			},
			want: LogMMSEConfig{
				MinGain: 0.05,
				MaxGain: 1,
				Floor:   1e-12,
			},
		},
		{
			name: "minimum gain above one",
			cfg: LogMMSEConfig{
				MinGain: 2,
				MaxGain: 1,
				Floor:   1e-12,
			},
			want: LogMMSEConfig{
				MinGain: 0.05,
				MaxGain: 1,
				Floor:   1e-12,
			},
		},
		{
			name: "maximum gain zero",
			cfg: LogMMSEConfig{
				MinGain: 0.05,
				MaxGain: 0,
				Floor:   1e-12,
			},
			want: LogMMSEConfig{
				MinGain: 0.05,
				MaxGain: 1,
				Floor:   1e-12,
			},
		},
		{
			name: "maximum gain above one",
			cfg: LogMMSEConfig{
				MinGain: 0.05,
				MaxGain: 2,
				Floor:   1e-12,
			},
			want: LogMMSEConfig{
				MinGain: 0.05,
				MaxGain: 1,
				Floor:   1e-12,
			},
		},
		{
			name: "minimum gain above maximum gain",
			cfg: LogMMSEConfig{
				MinGain: 0.8,
				MaxGain: 0.5,
				Floor:   1e-12,
			},
			want: LogMMSEConfig{
				MinGain: 0.05,
				MaxGain: 0.5,
				Floor:   1e-12,
			},
		},
		{
			name: "invalid floor",
			cfg: LogMMSEConfig{
				MinGain: 0.05,
				MaxGain: 1,
				Floor:   0,
			},
			want: LogMMSEConfig{
				MinGain: 0.05,
				MaxGain: 1,
				Floor:   1e-12,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			suppressor := NewLogMMSE(tt.cfg, snr.DefaultDecisionDirectedConfig())
			assertLogMMSEClose(t, suppressor.cfg.MinGain, tt.want.MinGain)
			assertLogMMSEClose(t, suppressor.cfg.MaxGain, tt.want.MaxGain)
			assertLogMMSEClose(t, suppressor.cfg.Floor, tt.want.Floor)
		})
	}
}

// TestLogMMSEExponentialIntegral verify representative E1 values
func TestLogMMSEExponentialIntegral(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		want float64
	}{
		{
			name: "point one",
			x:    0.1,
			want: 1.82292395841939,
		},
		{
			name: "one",
			x:    1,
			want: 0.21938393439552,
		},
		{
			name: "two",
			x:    2,
			want: 0.04890051070806,
		},
		{
			name: "ten",
			x:    10,
			want: 0.00000415696892968532,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := expIntegralE1(tt.x)

			if math.Abs(got-tt.want) > 1e-10 {
				t.Fatalf("expected %.14f, got %.14f", tt.want, got)
			}
		})
	}
}

// TestLogMMSEExponentialIntegralInvalid verify invalid input
func TestLogMMSEExponentialIntegralInvalid(t *testing.T) {
	got := expIntegralE1(0)

	if !math.IsInf(got, 1) {
		t.Fatalf("expected positive infinity, got %v", got)
	}
}

// TestLogMMSEGainIncreasesWithSNR verify speech dominated bins receive
// greater gain than noise dominated bins
func TestLogMMSEGainIncreasesWithSNR(t *testing.T) {
	low := logMMSEGain(1, 0.1, 1e-12)

	high := logMMSEGain(20, 10, 1e-12)

	if high <= low {
		t.Fatalf("expected high-SNR gain %v to exceed low-SNR gain %v", high, low)
	}
}

// TestClampLogMMSEGain verify gain limits
func TestClampLogMMSEGain(t *testing.T) {
	assertLogMMSEClose(t, clampLogMMSEGain(0.01, 0.05, 1), 0.05)
	assertLogMMSEClose(t, clampLogMMSEGain(0.5, 0.05, 1), 0.5)
	assertLogMMSEClose(t, clampLogMMSEGain(2, 0.05, 1), 1)
}

// TestLogMMSEProcess verify spectral suppression and clean power output
func TestLogMMSEProcess(t *testing.T) {
	suppressor := NewLogMMSE(DefaultLogMMSEConfig(), snr.DefaultDecisionDirectedConfig())

	spectrum := audio.Spectrum{
		Bins: []complex64{complex(1, 0), complex(2, 0), complex(10, 0)},
	}

	noisePSD := []float32{1, 1, 1}

	output, err := suppressor.Process(spectrum, noisePSD)
	if err != nil {
		t.Fatalf("process Log-MMSE: %v", err)
	}

	if len(output.Bins) != len(spectrum.Bins) {
		t.Fatalf("expected %d bins, got %d", len(spectrum.Bins), len(output.Bins))
	}

	if len(output.Power) != len(spectrum.Bins) {
		t.Fatalf("expected %d power bins, got %d", len(spectrum.Bins), len(output.Power))
	}

	for k, bin := range output.Bins {
		re := real(bin)
		im := imag(bin)

		expectedPower := re*re + im*im

		assertLogMMSEClose(t, output.Power[k], expectedPower)
	}

	// Every bin should be attenuated or preserved but never amplified
	for k := range output.Bins {
		inputMagnitude :=
			float32(
				math.Hypot(float64(real(spectrum.Bins[k])), float64(imag(spectrum.Bins[k]))),
			)

		outputMagnitude :=
			float32(
				math.Hypot(float64(real(output.Bins[k])), float64(imag(output.Bins[k]))),
			)

		if outputMagnitude > inputMagnitude+logMMSETestTolerance {
			t.Fatalf("bin %d amplified from %v to %v", k, inputMagnitude, outputMagnitude)
		}
	}
}

// TestLogMMSEProcessSizeMismatch verify invalid noise PSD size
func TestLogMMSEProcessSizeMismatch(t *testing.T) {
	suppressor := NewLogMMSE(
		DefaultLogMMSEConfig(),
		snr.DefaultDecisionDirectedConfig(),
	)

	_, err := suppressor.Process(
		audio.Spectrum{Bins: make([]complex64, 3)},
		[]float32{1, 1},
	)

	if err == nil {
		t.Fatal(
			"expected spectrum size mismatch error",
		)
	}
}

// TestLogMMSEEmptySpectrum verify empty input
func TestLogMMSEEmptySpectrum(t *testing.T) {
	suppressor := NewLogMMSE(
		DefaultLogMMSEConfig(),
		snr.DefaultDecisionDirectedConfig(),
	)

	output, err := suppressor.Process(audio.Spectrum{}, nil)
	if err != nil {
		t.Fatalf("process empty spectrum: %v", err)
	}

	if output.Bins != nil {
		t.Fatalf("expected nil bins, got %v", output.Bins)
	}
}

// TestLogMMSEReset verifies suppressor state can be cleared
func TestLogMMSEReset(t *testing.T) {
	suppressor := NewLogMMSE(
		DefaultLogMMSEConfig(),
		snr.DefaultDecisionDirectedConfig(),
	)

	_, err := suppressor.Process(
		audio.Spectrum{
			Bins: []complex64{complex(2, 0)},
		},
		[]float32{1},
	)
	if err != nil {
		t.Fatalf("process Log-MMSE: %v", err)
	}

	suppressor.Reset()

	if suppressor.cleanPower != nil {
		t.Fatalf("expected nil clean power, got %v", suppressor.cleanPower)
	}
}

// TestLogMMSEGainNegativeGamma verify negative posteriori SNR values
// are clamped to zero before calculating the gain
func TestLogMMSEGainNegativeGamma(t *testing.T) {
	got := logMMSEGain(-1, 1, 1e-12)
	want := logMMSEGain(0, 1, 1e-12)
	assertLogMMSEClose(t, got, want)
}

// TestLogMMSEGainNegativeXi verify negative priori SNR values are
// clamped to zero before calculating the gain
func TestLogMMSEGainNegativeXi(t *testing.T) {
	got := logMMSEGain(1, -1, 1e-12)
	assertLogMMSEClose(t, got, 0)
}

// TestLogMMSEGainInvalidResult verify invalid floating-point gain results
// are converted to zero instead of propagating NaN or infinity
func TestLogMMSEGainInvalidResult(t *testing.T) {
	tests := []struct {
		name  string
		gamma float32
		xi    float32
	}{
		{
			name:  "NaN gamma",
			gamma: float32(math.NaN()),
			xi:    1,
		},
		{
			name:  "positive infinity gamma",
			gamma: float32(math.Inf(1)),
			xi:    1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := logMMSEGain(tt.gamma, tt.xi, 1e-12)
			assertLogMMSEClose(t, got, 0)
		})
	}
}

// logMMSETestSNREstimator provide controllable SNR behavior for testing
// Log MMSE error handling independently from the production SNR estimator
type logMMSETestSNREstimator struct {
	gamma []float32
	xi    []float32

	processErr error
	updateErr  error
}

// Process return configured SNR values or the configured processing error
func (e *logMMSETestSNREstimator) Process(power []float32, noisePSD []float32) ([]float32, []float32, error) {
	if e.processErr != nil {
		return nil, nil, e.processErr
	}

	return e.gamma, e.xi, nil
}

// Update returns the configured state-update error
func (e *logMMSETestSNREstimator) Update(cleanPower, noisePSD []float32) error {
	return e.updateErr
}

// Reset satisfy the SNR processor contract
func (e *logMMSETestSNREstimator) Reset() {
}

// TestLogMMSEProcessSNRError verifies SNR estimation errors are returned
// with Log MMSE processing context
func TestLogMMSEProcessSNRError(t *testing.T) {
	expectedErr := errors.New("SNR estimation failure")

	suppressor := NewLogMMSE(
		DefaultLogMMSEConfig(),
		snr.DefaultDecisionDirectedConfig(),
	)

	suppressor.snrEstimator = &logMMSETestSNREstimator{processErr: expectedErr}

	_, err := suppressor.Process(
		audio.Spectrum{Bins: []complex64{complex(1, 0)}},
		[]float32{1},
	)

	if err == nil {
		t.Fatal("expected SNR estimation error")
	}

	if !errors.Is(err, expectedErr) {
		t.Fatalf("expected wrapped error %v, got %v", expectedErr, err)
	}
}

// TestLogMMSEProcessSNRUpdateError verify SNR state update errors are
// returned with Log MMSE processing context
func TestLogMMSEProcessSNRUpdateError(t *testing.T) {
	expectedErr := errors.New(
		"SNR update failure",
	)

	suppressor := NewLogMMSE(
		DefaultLogMMSEConfig(),
		snr.DefaultDecisionDirectedConfig(),
	)

	suppressor.snrEstimator = &logMMSETestSNREstimator{
		gamma: []float32{2},
		xi:    []float32{1},

		updateErr: expectedErr,
	}

	_, err := suppressor.Process(
		audio.Spectrum{Bins: []complex64{complex(2, 0)}},
		[]float32{1},
	)

	if err == nil {
		t.Fatal("expected SNR update error")
	}

	if !errors.Is(err, expectedErr) {
		t.Fatalf("expected wrapped error %v, got %v", expectedErr, err)
	}
}

// TestLogMMSEExponentialIntegralContinuedFractionGuards verify very small
// continued fraction denominators are protected by the configured minimum
func TestLogMMSEExponentialIntegralContinuedFractionGuards(t *testing.T) {
	// use an intentionally large fpMin so both denominator guards are
	// exercised without relying on floating point underflow
	got := expIntegralE1ContinuedFraction(2, 1e-14, 10, 100)

	if math.IsNaN(got) {
		t.Fatal("expected finite result, got NaN")
	}

	if math.IsInf(got, 0) {
		t.Fatalf("expected finite result, got %v", got)
	}

	if got <= 0 {
		t.Fatalf("expected positive result, got %v", got)
	}
}

// assertLogMMSEClose verify two float32 values are approximately equal
func assertLogMMSEClose(t *testing.T, got float32, want float32) {
	t.Helper()

	diff := math.Abs(float64(got - want))

	if diff > logMMSETestTolerance {
		t.Fatalf("expected %v, got %v", want, got)
	}
}
