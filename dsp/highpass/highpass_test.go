package highpass

import (
	"math"
	"testing"
)

func TestCalculateCoefficient(t *testing.T) {
	tests := []struct {
		name       string
		sampleRate int
		cutoffHz   float32
		want       float32
	}{
		{
			name:       "80 Hz at 8 kHz",
			sampleRate: 8000,
			cutoffHz:   80,
			want:       0.9391014,
		},
		{
			name:       "100 Hz at 8 kHz",
			sampleRate: 8000,
			cutoffHz:   100,
			want:       0.92446524,
		},
		{
			name:       "zero cutoff",
			sampleRate: 8000,
			cutoffHz:   0,
			want:       1,
		},
		{
			name:       "negative cutoff",
			sampleRate: 8000,
			cutoffHz:   -10,
			want:       1,
		},
		{
			name:       "zero sample rate",
			sampleRate: 0,
			cutoffHz:   80,
			want:       0,
		},
		{
			name:       "negative sample rate",
			sampleRate: -8000,
			cutoffHz:   80,
			want:       0,
		},
	}

	const tolerance = 1e-6

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateCoefficient(tt.sampleRate, tt.cutoffHz)

			if math.Abs(float64(got-tt.want)) > tolerance {
				t.Fatalf(
					"calculateCoefficient(%d, %f) = %f, want %f",
					tt.sampleRate, tt.cutoffHz, got, tt.want,
				)
			}
		})
	}
}

func TestFilter_Process(t *testing.T) {
	filter := New(8000, 80)

	input := []float32{0.1, 0.2, 0.3, 0.2, 0.1}

	// Calculate the expected output manually using
	// y[n] = x[n] - x[n-1] + r*y[n-1]
	r := calculateCoefficient(8000, 80)

	want := make([]float32, len(input))

	var x1 float32
	var y1 float32

	for i, x := range input {
		y := x - x1 + r*y1

		want[i] = y

		x1 = x
		y1 = y
	}

	got, err := filter.Process(input)
	if err != nil {
		t.Fatalf("Process() returned unexpected error: %v", err)
	}

	if len(got) != len(want) {
		t.Fatalf(
			"Process() returned %d samples, want %d",
			len(got), len(want),
		)
	}

	const tolerance = 1e-6

	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Errorf(
				"sample[%d] = %f, want %f",
				i, got[i], want[i],
			)
		}
	}
}

func TestFilter_ProcessInPlace(t *testing.T) {
	filter := New(8000, 80)

	samples := []float32{0.1, 0.2, 0.3}

	originalPointer := &samples[0]

	got, err := filter.Process(samples)
	if err != nil {
		t.Fatalf("Process() returned unexpected error: %v", err)
	}

	if len(got) == 0 {
		t.Fatal("Process() returned an empty slice")
	}

	// The high-pass implementation is expected to process
	// the provided slice in place
	if &got[0] != originalPointer {
		t.Fatal("Process() allocated a new output slice")
	}
}

func TestFilter_ProcessEmptySamples(t *testing.T) {
	filter := New(8000, 80)

	got, err := filter.Process(nil)
	if err != nil {
		t.Fatalf("Process() returned unexpected error: %v", err)
	}

	if got != nil {
		t.Fatalf("Process(nil) = %#v, want nil", got)
	}
}

func TestFilter_StateIsPreservedBetweenFrames(t *testing.T) {
	const sampleRate = 8000
	const cutoffHz = float32(80)

	filter := New(sampleRate, cutoffHz)

	frame1 := []float32{0.1, 0.2}

	frame2 := []float32{0.3, 0.4}

	_, err := filter.Process(frame1)
	if err != nil {
		t.Fatalf("Process(frame1) returned unexpected error: %v", err)
	}

	got, err := filter.Process(frame2)
	if err != nil {
		t.Fatalf("Process(frame2) returned unexpected error: %v", err)
	}

	// Calculate what the first sample of frame2 should be while
	// preserving the filter state from frame1
	r := calculateCoefficient(sampleRate, cutoffHz)

	y0 := float32(0.1)

	y1 := float32(0.2) - float32(0.1) + r*y0

	want := float32(0.3) - float32(0.2) + r*y1

	const tolerance = 1e-6 // 0.000001

	if math.Abs(float64(got[0]-want)) > tolerance {
		t.Fatalf(
			"first sample of second frame = %f, want %f",
			got[0], want,
		)
	}
}

func TestFilter_Reset(t *testing.T) {
	filter := New(8000, 80)

	samples := []float32{0.5, 0.5, 0.5}

	_, err := filter.Process(samples)
	if err != nil {
		t.Fatalf("Process() returned unexpected error: %v", err)
	}

	if filter.previousInput == 0 && filter.previousOutput == 0 {
		t.Fatal("expected filter to contain state after processing")
	}

	filter.Reset()

	if filter.previousInput != 0 {
		t.Fatalf("x1 = %f after Reset(), want 0", filter.previousInput)
	}

	if filter.previousOutput != 0 {
		t.Fatalf("y1 = %f after Reset(), want 0", filter.previousOutput)
	}
}

func TestFilter_RemovesDC(t *testing.T) {
	filter := New(8000, 80)

	// A constant signal represents a DC component
	//
	// The first sample produces a transient, after which the
	// output should decay towards zero
	samples := make([]float32, 1000)

	for i := range samples {
		samples[i] = 0.5
	}

	got, err := filter.Process(samples)
	if err != nil {
		t.Fatalf("Process() returned unexpected error: %v", err)
	}

	if got[0] == 0 {
		t.Fatal("expected first sample to contain the initial transient")
	}

	// After enough samples the constant DC component should be
	// almost completely removed
	last := got[len(got)-1]

	if math.Abs(float64(last)) > 1e-4 {
		t.Fatalf(
			"DC component did not decay sufficiently, last sample = %f",
			last,
		)
	}
}
