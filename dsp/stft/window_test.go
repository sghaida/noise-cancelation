package stft

import (
	"math"
	"testing"
)

func TestHannWindow(t *testing.T) {
	tests := []struct {
		name string
		size int
	}{
		{
			name: "single sample",
			size: 1,
		},
		{
			name: "small window",
			size: 8,
		},
		{
			name: "STFT window",
			size: 256,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hannWindow(tt.size)

			if len(got) != tt.size {
				t.Fatalf("window length = %d, want %d", len(got), tt.size)
			}

			if tt.size == 1 {
				if got[0] != 1 {
					t.Fatalf("single-sample window value = %f, want 1", got[0])
				}

				return
			}

			assertHannWindowProperties(t, got)
		})
	}
}

func assertHannWindowProperties(t *testing.T, window []float32) {
	t.Helper()

	const tolerance = 1e-6

	if math.Abs(float64(window[0])) > tolerance {
		t.Fatalf("first window value = %f, want 0", window[0])
	}

	if math.Abs(float64(window[len(window)-1])) > tolerance {
		t.Fatalf("last window value = %f, want 0", window[len(window)-1])
	}

	for i := range window {
		j := len(window) - 1 - i

		if math.Abs(float64(window[i]-window[j])) > tolerance {
			t.Fatalf(
				"window is not symmetric at indexes %d and %d: %f != %f",
				i, j, window[i], window[j],
			)
		}
	}

	for i, value := range window {
		if value < 0 || value > 1 {
			t.Fatalf(
				"window[%d] = %f, want value between 0 and 1",
				i, value,
			)
		}
	}
}

func TestHannWindowKnownValues(t *testing.T) {
	const size = 8

	got := hannWindow(size)

	want := []float32{
		0.0000000, 0.1882551, 0.6112605, 0.9504844, 0.9504844, 0.6112605, 0.1882551, 0.0000000,
	}

	const tolerance = 1e-6

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Fatalf("window[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestHannWindowInvalidSize(t *testing.T) {
	tests := []struct {
		name string
		size int
	}{
		{
			name: "zero",
			size: 0,
		},
		{
			name: "negative",
			size: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hannWindow(tt.size)

			if got != nil {
				t.Fatalf("hannWindow(%d) = %#v, want nil", tt.size, got)
			}
		})
	}
}
