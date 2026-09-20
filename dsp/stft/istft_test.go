package stft

import (
	"math"
	"testing"

	"github.com/sghaida/noise-cancelation/audio"
)

func TestNewSynthesizer(t *testing.T) {
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
			got, err := NewSynthesizer(tt.fftSize, tt.hopSize)

			if tt.wantErr {
				if err == nil {
					t.Fatal("NewSynthesizer() returned nil error")
				}
				return
			}

			if err != nil {
				t.Fatalf("NewSynthesizer() returned error: %v", err)
			}

			if got.fftSize != tt.fftSize {
				t.Errorf("fftSize = %d, want %d",
					got.fftSize, tt.fftSize)
			}

			if got.hopSize != tt.hopSize {
				t.Errorf("hopSize = %d, want %d",
					got.hopSize, tt.hopSize)
			}
		})
	}
}

func TestSynthesizerRejectsWrongBinCount(t *testing.T) {
	synth := newTestSynthesizer(t)

	spectrum := audio.Spectrum{
		Bins: make([]complex64, 128),
	}

	_, err := synth.Process(spectrum)
	if err == nil {
		t.Fatal("Process() returned nil error")
	}
}

func TestSynthesizerOutputSize(t *testing.T) {
	analyzer := newTestAnalyzer(t)
	synth := newTestSynthesizer(t)

	spectra, err := analyzer.Process(make([]float32, 256))
	if err != nil {
		t.Fatalf("Analyzer.Process() returned error: %v", err)
	}

	output, err := synth.Process(spectra[0])
	if err != nil {
		t.Fatalf("Process() returned error: %v", err)
	}

	if len(output) != 128 {
		t.Fatalf("output length = %d, want 128", len(output))
	}
}

func TestSynthesizerSilence(t *testing.T) {
	analyzer := newTestAnalyzer(t)
	synth := newTestSynthesizer(t)

	spectra, err := analyzer.Process(make([]float32, 256))
	if err != nil {
		t.Fatalf("Analyzer.Process() returned error: %v", err)
	}

	output, err := synth.Process(spectra[0])
	if err != nil {
		t.Fatalf("Process() returned error: %v", err)
	}

	for i, value := range output {
		if value != 0 {
			t.Fatalf("output[%d] = %f, want 0", i, value)
		}
	}
}

func TestSynthesizerFlushWithoutProcess(t *testing.T) {
	synth := newTestSynthesizer(t)

	output := synth.Flush()

	if output != nil {
		t.Fatalf("Flush() = %#v, want nil", output)
	}
}

func TestSynthesizerFlushReturnsTail(t *testing.T) {
	analyzer := newTestAnalyzer(t)
	synth := newTestSynthesizer(t)

	samples := sineWave(256, 1000, 8000)

	spectra, err := analyzer.Process(samples)
	if err != nil {
		t.Fatalf("Analyzer.Process() returned error: %v", err)
	}

	_, err = synth.Process(spectra[0])
	if err != nil {
		t.Fatalf("Synthesizer.Process() returned error: %v", err)
	}

	tail := synth.Flush()

	if len(tail) != 128 {
		t.Fatalf("Flush() length = %d, want 128", len(tail))
	}
}

func TestSynthesizerFlushOnlyOnce(t *testing.T) {
	analyzer := newTestAnalyzer(t)
	synth := newTestSynthesizer(t)

	spectra, err := analyzer.Process(make([]float32, 256))
	if err != nil {
		t.Fatalf("Analyzer.Process() returned error: %v", err)
	}

	_, err = synth.Process(spectra[0])
	if err != nil {
		t.Fatalf("Synthesizer.Process() returned error: %v", err)
	}

	first := synth.Flush()
	second := synth.Flush()

	if len(first) != 128 {
		t.Fatalf("first Flush() length = %d, want 128", len(first))
	}

	if second != nil {
		t.Fatalf("second Flush() = %#v, want nil", second)
	}
}

func TestSynthesizerReset(t *testing.T) {
	synth := newTestSynthesizer(t)

	for i := range synth.overlap {
		synth.overlap[i] = 1
		synth.windowWeight[i] = 1
	}

	synth.pending = true
	synth.Reset()

	if synth.pending {
		t.Fatal("pending = true after Reset()")
	}

	for i := range synth.overlap {
		if synth.overlap[i] != 0 {
			t.Fatalf("overlap[%d] = %f, want 0",
				i, synth.overlap[i])
		}

		if synth.windowWeight[i] != 0 {
			t.Fatalf("windowWeight[%d] = %f, want 0",
				i, synth.windowWeight[i])
		}
	}
}

func TestSTFTISTFTRoundTrip(t *testing.T) {
	const (
		sampleRate  = 8000
		sampleCount = 1024
		fftSize     = 256
		hopSize     = 128
		frequency   = 1000
	)

	analyzer, err := New(fftSize, hopSize)
	if err != nil {
		t.Fatalf("New() returned error: %v", err)
	}

	synth, err := NewSynthesizer(fftSize, hopSize)
	if err != nil {
		t.Fatalf("NewSynthesizer() returned error: %v", err)
	}

	input := sineWave(sampleCount, frequency, sampleRate)

	spectra, err := analyzer.Process(input)
	if err != nil {
		t.Fatalf("Analyzer.Process() returned error: %v", err)
	}

	finalSpectra, err := analyzer.Flush()
	if err != nil {
		t.Fatalf("Analyzer.Flush() returned error: %v", err)
	}

	spectra = append(spectra, finalSpectra...)

	var output []float32

	for _, spectrum := range spectra {
		frame, err := synth.Process(spectrum)
		if err != nil {
			t.Fatalf("Synthesizer.Process() returned error: %v", err)
		}

		output = append(output, frame...)
	}

	output = append(output, synth.Flush()...)

	if len(output) < len(input) {
		t.Fatalf(
			"output length = %d, need at least %d",
			len(output), len(input),
		)
	}

	output = output[:len(input)]

	// Ignore the beginning and end where the Hann window has very small
	// weights and floating-point errors are amplified during normalization
	start := hopSize
	end := len(input) - hopSize

	const tolerance = 1e-4

	for i := start; i < end; i++ {
		diff := math.Abs(float64(output[i] - input[i]))

		if diff > tolerance {
			t.Fatalf(
				"sample[%d] = %.6f, want %.6f, diff %.6f",
				i, output[i], input[i], diff,
			)
		}
	}
}

func newTestSynthesizer(t *testing.T) *Synthesizer {
	t.Helper()

	synth, err := NewSynthesizer(256, 128)
	if err != nil {
		t.Fatalf("NewSynthesizer() returned error: %v", err)
	}

	return synth
}

func sineWave(size int, frequency int, sampleRate int) []float32 {
	samples := make([]float32, size)

	for i := range samples {
		angle := 2 * math.Pi * float64(frequency*i)
		angle /= float64(sampleRate)

		samples[i] = float32(math.Sin(angle))
	}

	return samples
}
