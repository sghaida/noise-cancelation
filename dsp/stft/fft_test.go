package stft

import (
	"math"
	"strings"
	"testing"
)

func TestIsPowerOfTwo(t *testing.T) {
	tests := []struct {
		name  string
		value int
		want  bool
	}{
		{
			name:  "zero",
			value: 0,
			want:  false,
		},
		{
			name:  "negative",
			value: -2,
			want:  false,
		},
		{
			name:  "one",
			value: 1,
			want:  true,
		},
		{
			name:  "two",
			value: 2,
			want:  true,
		},
		{
			name:  "three",
			value: 3,
			want:  false,
		},
		{
			name:  "eight",
			value: 8,
			want:  true,
		},
		{
			name:  "two hundred fifty six",
			value: 256,
			want:  true,
		},
		{
			name:  "two hundred fifty",
			value: 250,
			want:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isPowerOfTwo(tt.value)

			if got != tt.want {
				t.Fatalf(
					"isPowerOfTwo(%d) = %v, want %v",
					tt.value, got, tt.want,
				)
			}
		})
	}
}

func TestBitReverse(t *testing.T) {
	samples := []complex64{
		0, 1, 2, 3, 4, 5, 6, 7,
	}

	want := []complex64{
		0, 4, 2, 6, 1, 5, 3, 7,
	}

	bitReverse(samples)

	for i := range want {
		if samples[i] != want[i] {
			t.Fatalf(
				"samples[%d] = %v, want %v",
				i, samples[i], want[i],
			)
		}
	}
}

func TestFFTEmpty(t *testing.T) {
	var samples []complex64

	err := fft(samples)
	if err != nil {
		t.Fatalf(
			"fft() returned unexpected error: %v", err,
		)
	}
}

func TestFFTRejectsNonPowerOfTwo(t *testing.T) {
	samples := []complex64{1, 2, 3}

	err := fft(samples)
	if err == nil {
		t.Fatal(
			"fft() returned nil error for non-power-of-two input",
		)
	}

	if !strings.Contains(err.Error(), "must be a power of two") {
		t.Fatalf(
			"fft() error = %q, want power-of-two error", err.Error(),
		)
	}
}

func TestFFTImpulse(t *testing.T) {
	samples := []complex64{
		1, 0, 0, 0, 0, 0, 0, 0,
	}

	err := fft(samples)
	if err != nil {
		t.Fatalf("fft() returned unexpected error: %v", err)
	}

	// An impulse in the time domain contains equal energy
	// at every frequency bin
	want := []complex64{
		1, 1, 1, 1, 1, 1, 1, 1,
	}

	assertComplexSlicesClose(t, samples, want, 1e-6)
}

func TestFFTConstantSignal(t *testing.T) {
	samples := []complex64{
		1, 1, 1, 1, 1, 1, 1, 1,
	}

	err := fft(samples)
	if err != nil {
		t.Fatalf("fft() returned unexpected error: %v", err)
	}

	// A constant signal contains only DC
	//
	// The DC bin equals the sum of all input samples
	// while all other bins should be approximately zero
	want := []complex64{
		8, 0, 0, 0, 0, 0, 0, 0,
	}

	assertComplexSlicesClose(
		t, samples, want, 1e-6,
	)
}

func TestFFTSingleBinSineWave(t *testing.T) {
	const size = 8

	samples := make([]complex64, size)

	// Generate exactly one FFT-bin frequency
	//
	// For an 8-point FFT this sine wave completes one cycle
	// across the full input frame and therefore falls into bin 1
	for n := range samples {
		angle := 2 * math.Pi * float64(n) / float64(size)

		samples[n] = complex(float32(math.Sin(angle)), 0)
	}

	err := fft(samples)
	if err != nil {
		t.Fatalf("fft() returned unexpected error: %v", err)
	}

	// A real-valued sine wave creates two mirrored peaks
	// at bins 1 and N-1
	//
	// Their expected imaginary values are:
	//
	// X[1]   = -jN/2
	// X[N-1] = +jN/2
	const tolerance = 1e-5

	for i, value := range samples {
		switch i {
		case 1:
			assertComplexClose(
				t, value, complex(0, -4), tolerance,
			)

		case size - 1:
			assertComplexClose(
				t, value, complex(0, 4), tolerance,
			)

		default:
			assertComplexClose(
				t, value, 0, tolerance,
			)
		}
	}
}

func TestFFTAndIFFTRoundTrip(t *testing.T) {
	original := []complex64{
		complex(0.2, 0),
		complex(-0.4, 0),
		complex(0.8, 0),
		complex(0.1, 0),
		complex(-0.7, 0),
		complex(0.3, 0),
		complex(0.5, 0),
		complex(-0.2, 0),
	}

	samples := append(
		[]complex64(nil),
		original...,
	)

	err := fft(samples)
	if err != nil {
		t.Fatalf("fft() returned unexpected error: %v", err)
	}

	err = ifft(samples)
	if err != nil {
		t.Fatalf("ifft() returned unexpected error: %v", err)
	}

	assertComplexSlicesClose(
		t, samples, original, 1e-5,
	)
}

func TestIFFTEmpty(t *testing.T) {
	var samples []complex64

	err := ifft(samples)
	if err != nil {
		t.Fatalf("ifft() returned unexpected error: %v", err)
	}
}

func TestIFFTRejectsNonPowerOfTwo(t *testing.T) {
	samples := []complex64{1, 2, 3}

	err := ifft(samples)
	if err == nil {
		t.Fatal("ifft() returned nil error for non-power-of-two input")
	}

	if !strings.Contains(err.Error(), "must be a power of two") {
		t.Fatalf(
			"ifft() error = %q, want power-of-two error", err.Error(),
		)
	}
}

func TestFFTKnownValues(t *testing.T) {
	samples := []complex64{
		1, 2, 3, 4,
	}

	err := fft(samples)
	if err != nil {
		t.Fatalf("fft() returned unexpected error: %v", err)
	}

	// DFT of [1, 2, 3, 4]
	//
	// X[0] = 10
	// X[1] = -2 + 2i
	// X[2] = -2
	// X[3] = -2 - 2i
	want := []complex64{
		complex(10, 0),
		complex(-2, 2),
		complex(-2, 0),
		complex(-2, -2),
	}

	assertComplexSlicesClose(
		t, samples, want, 1e-5,
	)
}

// assertComplexSlicesClose verifies that two complex sample slices are
// equal within the provided floating-point tolerance
func assertComplexSlicesClose(
	t *testing.T,
	got []complex64,
	want []complex64,
	tolerance float64,
) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}

	for i := range want {
		assertComplexClose(t, got[i], want[i], tolerance)
	}
}

// assertComplexClose verifies that the real and imaginary parts of two
// complex values are equal within the provided tolerance
func assertComplexClose(
	t *testing.T,
	got complex64,
	want complex64,
	tolerance float64,
) {
	t.Helper()

	realDifference := math.Abs(
		float64(
			real(got) - real(want),
		),
	)

	imaginaryDifference := math.Abs(
		float64(imag(got) - imag(want)),
	)

	if realDifference > tolerance ||
		imaginaryDifference > tolerance {
		t.Fatalf(
			"got %v, want %v, real difference %.8f, imaginary difference %.8f",
			got, want, realDifference, imaginaryDifference,
		)
	}
}
