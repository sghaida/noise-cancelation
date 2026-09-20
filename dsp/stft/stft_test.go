package stft

import (
	"math"
	"testing"
)

func TestNewAnalyzer(t *testing.T) {
	tests := []struct {
		name    string
		fftSize int
		hopSize int
		wantErr bool
	}{
		{"valid", 256, 128, false},
		{"zero FFT size", 0, 128, true},
		{"negative FFT size", -1, 128, true},
		{"non power of two", 250, 128, true},
		{"zero hop size", 256, 0, true},
		{"negative hop size", 256, -1, true},
		{"hop larger than FFT", 256, 257, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := New(tt.fftSize, tt.hopSize)

			if tt.wantErr {
				if err == nil {
					t.Fatal("New() returned nil error")
				}
				return
			}

			if err != nil {
				t.Fatalf("New() returned error: %v", err)
			}

			if got.FFTSize() != tt.fftSize {
				t.Errorf("FFTSize() = %d, want %d",
					got.FFTSize(), tt.fftSize)
			}

			if got.HopSize() != tt.hopSize {
				t.Errorf("HopSize() = %d, want %d",
					got.HopSize(), tt.hopSize)
			}
		})
	}
}

func TestAnalyzerEmptyInput(t *testing.T) {
	analyzer := newTestAnalyzer(t)

	got, err := analyzer.Process(nil)
	if err != nil {
		t.Fatalf("Process() returned error: %v", err)
	}

	if got != nil {
		t.Fatalf("Process() = %#v, want nil", got)
	}
}

func TestAnalyzerBuffersIncompleteFrame(t *testing.T) {
	analyzer := newTestAnalyzer(t)

	spectra, err := analyzer.Process(make([]float32, 160))
	if err != nil {
		t.Fatalf("Process() returned error: %v", err)
	}

	if len(spectra) != 0 {
		t.Fatalf("spectra count = %d, want 0", len(spectra))
	}

	if len(analyzer.buffer) != 160 {
		t.Fatalf("buffer length = %d, want 160", len(analyzer.buffer))
	}
}

func TestAnalyzerBuffersAcrossCalls(t *testing.T) {
	analyzer := newTestAnalyzer(t)

	_, err := analyzer.Process(make([]float32, 160))
	if err != nil {
		t.Fatalf("first Process() returned error: %v", err)
	}

	spectra, err := analyzer.Process(make([]float32, 160))
	if err != nil {
		t.Fatalf("second Process() returned error: %v", err)
	}

	if len(spectra) != 1 {
		t.Fatalf("spectra count = %d, want 1", len(spectra))
	}

	if len(analyzer.buffer) != 192 {
		t.Fatalf("buffer length = %d, want 192", len(analyzer.buffer))
	}
}

func TestAnalyzerProducesOverlappingFrames(t *testing.T) {
	analyzer := newTestAnalyzer(t)

	spectra, err := analyzer.Process(make([]float32, 384))
	if err != nil {
		t.Fatalf("Process() returned error: %v", err)
	}

	if len(spectra) != 2 {
		t.Fatalf("spectra count = %d, want 2", len(spectra))
	}

	if len(analyzer.buffer) != 128 {
		t.Fatalf("buffer length = %d, want 128", len(analyzer.buffer))
	}
}

func TestAnalyzerSpectrumSize(t *testing.T) {
	analyzer := newTestAnalyzer(t)

	spectra, err := analyzer.Process(make([]float32, 256))
	if err != nil {
		t.Fatalf("Process() returned error: %v", err)
	}

	if len(spectra) != 1 {
		t.Fatalf("spectra count = %d, want 1", len(spectra))
	}

	if len(spectra[0].Bins) != 129 {
		t.Errorf("bin count = %d, want 129", len(spectra[0].Bins))
	}

	if len(spectra[0].Power) != 129 {
		t.Errorf("power count = %d, want 129", len(spectra[0].Power))
	}
}

func TestAnalyzerDetectsKnownFrequency(t *testing.T) {
	const sampleRate = 8000
	const frequency = 1000
	const fftSize = 256

	analyzer := newTestAnalyzer(t)
	samples := make([]float32, fftSize)

	for i := range samples {
		angle := 2 * math.Pi * frequency * float64(i) / sampleRate
		samples[i] = float32(math.Sin(angle))
	}

	spectra, err := analyzer.Process(samples)
	if err != nil {
		t.Fatalf("Process() returned error: %v", err)
	}

	power := spectra[0].Power
	peakBin := strongestBin(power)
	wantBin := frequency * fftSize / sampleRate

	if peakBin != wantBin {
		t.Fatalf("peak bin = %d, want %d", peakBin, wantBin)
	}
}

func TestAnalyzerFlush(t *testing.T) {
	analyzer := newTestAnalyzer(t)

	_, err := analyzer.Process(make([]float32, 160))
	if err != nil {
		t.Fatalf("Process() returned error: %v", err)
	}

	spectra, err := analyzer.Flush()
	if err != nil {
		t.Fatalf("Flush() returned error: %v", err)
	}

	if len(spectra) != 1 {
		t.Fatalf("Flush() returned %d spectra, want 1", len(spectra))
	}

	if len(spectra[0].Bins) != 129 {
		t.Fatalf("bin count = %d, want 129", len(spectra[0].Bins))
	}

	if len(analyzer.buffer) != 0 {
		t.Fatalf("buffer length = %d, want 0", len(analyzer.buffer))
	}
}

func TestAnalyzerFlushEmpty(t *testing.T) {
	analyzer := newTestAnalyzer(t)

	spectra, err := analyzer.Flush()
	if err != nil {
		t.Fatalf("Flush() returned error: %v", err)
	}

	if spectra != nil {
		t.Fatalf("Flush() = %#v, want nil", spectra)
	}
}

func TestAnalyzerFlushOnlyOnce(t *testing.T) {
	analyzer := newTestAnalyzer(t)

	_, err := analyzer.Process(make([]float32, 100))
	if err != nil {
		t.Fatalf("Process() returned error: %v", err)
	}

	first, err := analyzer.Flush()
	if err != nil {
		t.Fatalf("first Flush() returned error: %v", err)
	}

	second, err := analyzer.Flush()
	if err != nil {
		t.Fatalf("second Flush() returned error: %v", err)
	}

	if len(first) != 1 {
		t.Fatalf("first Flush() count = %d, want 1", len(first))
	}

	if second != nil {
		t.Fatalf("second Flush() = %#v, want nil", second)
	}
}

func TestAnalyzerReset(t *testing.T) {
	analyzer := newTestAnalyzer(t)

	_, err := analyzer.Process(make([]float32, 160))
	if err != nil {
		t.Fatalf("Process() returned error: %v", err)
	}

	analyzer.Reset()

	if len(analyzer.buffer) != 0 {
		t.Fatalf("buffer length = %d, want 0", len(analyzer.buffer))
	}
}

func newTestAnalyzer(t *testing.T) *Analyzer {
	t.Helper()

	analyzer, err := New(256, 128)
	if err != nil {
		t.Fatalf("New() returned error: %v", err)
	}

	return analyzer
}

func strongestBin(power []float32) int {
	peakBin := 0

	for i := 1; i < len(power); i++ {
		if power[i] > power[peakBin] {
			peakBin = i
		}
	}

	return peakBin
}
