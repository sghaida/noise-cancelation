package audio

import (
	"math"
	"testing"
)

func newValidSpectrum() *Spectrum {
	return &Spectrum{
		Bins: make([]complex64, 129),
		Format: Format{
			SampleRate: SampleRate8K,
			Channels:   1,
		},
		FFTSize:      256,
		WindowSize:   256,
		HopSize:      128,
		Sequence:     10,
		SampleOffset: 160,
	}
}

func TestNewSpectrum(t *testing.T) {
	tests := []struct {
		name       string
		bins       []complex64
		format     Format
		fftSize    int
		windowSize int
		hopSize    int
		wantErr    bool
	}{
		{
			name: "valid spectrum",
			bins: make([]complex64, 129),
			format: Format{
				SampleRate: SampleRate8K,
				Channels:   1,
			},
			fftSize:    256,
			windowSize: 256,
			hopSize:    128,
			wantErr:    false,
		},
		{
			name: "invalid bin count",
			bins: make([]complex64, 128),
			format: Format{
				SampleRate: SampleRate8K,
				Channels:   1,
			},
			fftSize:    256,
			windowSize: 256,
			hopSize:    128,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewSpectrum(
				tt.bins,
				tt.format,
				tt.fftSize,
				tt.windowSize,
				tt.hopSize,
			)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}

				if got != nil {
					t.Fatal("expected nil spectrum")
				}

				return
			}

			if err != nil {
				t.Fatalf(
					"unexpected error: %v",
					err,
				)
			}

			if got == nil {
				t.Fatal("expected spectrum")
			}

			if got.FFTSize != tt.fftSize {
				t.Fatalf(
					"expected FFT size %d, got %d",
					tt.fftSize,
					got.FFTSize,
				)
			}

			if got.WindowSize != tt.windowSize {
				t.Fatalf(
					"expected window size %d, got %d",
					tt.windowSize,
					got.WindowSize,
				)
			}

			if got.HopSize != tt.hopSize {
				t.Fatalf(
					"expected hop size %d, got %d",
					tt.hopSize,
					got.HopSize,
				)
			}
		})
	}
}

func TestSpectrumValidate(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() *Spectrum
		wantErr bool
	}{
		{
			name: "nil spectrum",
			setup: func() *Spectrum {
				return nil
			},
			wantErr: true,
		},
		{
			name:    "valid",
			setup:   newValidSpectrum,
			wantErr: false,
		},
		{
			name: "valid with power",
			setup: func() *Spectrum {
				s := newValidSpectrum()
				s.Power = make([]float32, 129)

				return s
			},
			wantErr: false,
		},
		{
			name: "invalid format",
			setup: func() *Spectrum {
				s := newValidSpectrum()

				s.Format = Format{
					SampleRate: 12345,
					Channels:   1,
				}

				return s
			},
			wantErr: true,
		},
		{
			name: "zero FFT size",
			setup: func() *Spectrum {
				s := newValidSpectrum()
				s.FFTSize = 0

				return s
			},
			wantErr: true,
		},
		{
			name: "negative hop size",
			setup: func() *Spectrum {
				s := newValidSpectrum()
				s.HopSize = -1

				return s
			},
			wantErr: true,
		},
		{
			name: "hop larger than window",
			setup: func() *Spectrum {
				s := newValidSpectrum()
				s.HopSize = 257

				return s
			},
			wantErr: true,
		},
		{
			name: "incorrect bin count",
			setup: func() *Spectrum {
				s := newValidSpectrum()
				s.Bins = make([]complex64, 128)

				return s
			},
			wantErr: true,
		},
		{
			name: "incorrect power size",
			setup: func() *Spectrum {
				s := newValidSpectrum()
				s.Power = make([]float32, 128)

				return s
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spectrum := tt.setup()

			err := spectrum.Validate()

			if tt.wantErr && err == nil {
				t.Fatal("expected error")
			}

			if !tt.wantErr && err != nil {
				t.Fatalf(
					"unexpected error: %v",
					err,
				)
			}
		})
	}
}

func TestSpectrumBinCount(t *testing.T) {
	tests := []struct {
		name     string
		spectrum *Spectrum
		expected int
	}{
		{
			name:     "nil",
			spectrum: nil,
			expected: 0,
		},
		{
			name:     "valid",
			spectrum: newValidSpectrum(),
			expected: 129,
		},
		{
			name: "empty",
			spectrum: &Spectrum{
				Bins: nil,
			},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.spectrum.BinCount()

			if got != tt.expected {
				t.Fatalf(
					"expected %d, got %d",
					tt.expected,
					got,
				)
			}
		})
	}
}

func TestSpectrumBinWidthHz(t *testing.T) {
	tests := []struct {
		name     string
		spectrum *Spectrum
		expected float64
	}{
		{
			name:     "nil",
			spectrum: nil,
			expected: 0,
		},
		{
			name:     "valid",
			spectrum: newValidSpectrum(),
			expected: 31.25,
		},
		{
			name: "zero FFT size",
			spectrum: &Spectrum{
				Format: Format{
					SampleRate: SampleRate8K,
					Channels:   1,
				},
				FFTSize: 0,
			},
			expected: 0,
		},
		{
			name: "negative FFT size",
			spectrum: &Spectrum{
				Format: Format{
					SampleRate: SampleRate8K,
					Channels:   1,
				},
				FFTSize: -1,
			},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.spectrum.BinWidthHz()

			if got != tt.expected {
				t.Fatalf(
					"expected %f, got %f",
					tt.expected,
					got,
				)
			}
		})
	}
}

func TestSpectrumBinFrequency(t *testing.T) {
	spectrum := newValidSpectrum()

	tests := []struct {
		name     string
		spectrum *Spectrum
		bin      int
		expected float64
	}{
		{
			name:     "nil spectrum",
			spectrum: nil,
			bin:      1,
			expected: 0,
		},
		{
			name:     "negative bin",
			spectrum: spectrum,
			bin:      -1,
			expected: 0,
		},
		{
			name:     "out of range bin",
			spectrum: spectrum,
			bin:      129,
			expected: 0,
		},
		{
			name:     "DC",
			spectrum: spectrum,
			bin:      0,
			expected: 0,
		},
		{
			name:     "bin 1",
			spectrum: spectrum,
			bin:      1,
			expected: 31.25,
		},
		{
			name:     "500 Hz",
			spectrum: spectrum,
			bin:      16,
			expected: 500,
		},
		{
			name:     "1000 Hz",
			spectrum: spectrum,
			bin:      32,
			expected: 1000,
		},
		{
			name:     "2000 Hz",
			spectrum: spectrum,
			bin:      64,
			expected: 2000,
		},
		{
			name:     "Nyquist",
			spectrum: spectrum,
			bin:      128,
			expected: 4000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got :=
				tt.spectrum.BinFrequency(
					tt.bin,
				)

			if got != tt.expected {
				t.Fatalf(
					"expected %f Hz, got %f Hz",
					tt.expected,
					got,
				)
			}
		})
	}
}

func TestSpectrumFrequencyBin(t *testing.T) {
	valid := newValidSpectrum()

	clamped := &Spectrum{
		Bins: make([]complex64, 1),
		Format: Format{
			SampleRate: SampleRate8K,
			Channels:   1,
		},
		FFTSize: 256,
	}

	invalidFFT := newValidSpectrum()
	invalidFFT.FFTSize = 0

	tests := []struct {
		name     string
		spectrum *Spectrum
		hz       float64
		expected int
	}{
		{
			name:     "nil",
			spectrum: nil,
			hz:       1000,
			expected: 0,
		},
		{
			name:     "zero FFT size",
			spectrum: invalidFFT,
			hz:       1000,
			expected: 0,
		},
		{
			name:     "negative frequency",
			spectrum: valid,
			hz:       -100,
			expected: 0,
		},
		{
			name:     "zero frequency",
			spectrum: valid,
			hz:       0,
			expected: 0,
		},
		{
			name:     "31.25 Hz",
			spectrum: valid,
			hz:       31.25,
			expected: 1,
		},
		{
			name:     "round closest bin",
			spectrum: valid,
			hz:       50,
			expected: 2,
		},
		{
			name:     "500 Hz",
			spectrum: valid,
			hz:       500,
			expected: 16,
		},
		{
			name:     "1000 Hz",
			spectrum: valid,
			hz:       1000,
			expected: 32,
		},
		{
			name:     "2000 Hz",
			spectrum: valid,
			hz:       2000,
			expected: 64,
		},
		{
			name:     "at Nyquist",
			spectrum: valid,
			hz:       4000,
			expected: 128,
		},
		{
			name:     "above Nyquist",
			spectrum: valid,
			hz:       5000,
			expected: 128,
		},
		{
			name:     "calculated bin exceeds available bins",
			spectrum: clamped,
			hz:       1000,
			expected: 0,
		},
		{
			name:     "NaN frequency",
			spectrum: valid,
			hz:       math.NaN(),
			expected: 0,
		},
		{
			name:     "positive infinity",
			spectrum: valid,
			hz:       math.Inf(1),
			expected: 128,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got :=
				tt.spectrum.FrequencyBin(
					tt.hz,
				)

			if got != tt.expected {
				t.Fatalf(
					"expected bin %d, got %d",
					tt.expected,
					got,
				)
			}
		})
	}
}

func TestSpectrumNyquistBin(t *testing.T) {
	tests := []struct {
		name     string
		spectrum *Spectrum
		expected int
	}{
		{
			name:     "nil",
			spectrum: nil,
			expected: 0,
		},
		{
			name:     "empty",
			spectrum: &Spectrum{},
			expected: 0,
		},
		{
			name:     "valid",
			spectrum: newValidSpectrum(),
			expected: 128,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.spectrum.NyquistBin()

			if got != tt.expected {
				t.Fatalf(
					"expected %d, got %d",
					tt.expected,
					got,
				)
			}
		})
	}
}

func TestSpectrumNyquistHz(t *testing.T) {
	tests := []struct {
		name     string
		spectrum *Spectrum
		expected float64
	}{
		{
			name:     "nil",
			spectrum: nil,
			expected: 0,
		},
		{
			name:     "8 kHz",
			spectrum: newValidSpectrum(),
			expected: 4000,
		},
		{
			name: "48 kHz",
			spectrum: &Spectrum{
				Format: Format{
					SampleRate: SampleRate48K,
					Channels:   1,
				},
			},
			expected: 24000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.spectrum.NyquistHz()

			if got != tt.expected {
				t.Fatalf(
					"expected %f, got %f",
					tt.expected,
					got,
				)
			}
		})
	}
}

func TestSpectrumEnsurePower(t *testing.T) {
	tests := []struct {
		name     string
		spectrum *Spectrum
		expected []float32
		wantNil  bool
	}{
		{
			name:     "nil",
			spectrum: nil,
			wantNil:  true,
		},
		{
			name: "allocate power",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
					complex(5, 12),
				},
			},
			expected: []float32{
				25,
				169,
			},
		},
		{
			name: "reuse existing power",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
					complex(5, 12),
				},
				Power: make([]float32, 2),
			},
			expected: []float32{
				25,
				169,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got :=
				tt.spectrum.EnsurePower()

			if tt.wantNil {
				if got != nil {
					t.Fatalf(
						"expected nil, got %#v",
						got,
					)
				}

				return
			}

			if len(got) != len(tt.expected) {
				t.Fatalf(
					"expected length %d, got %d",
					len(tt.expected),
					len(got),
				)
			}

			for i := range tt.expected {
				if got[i] != tt.expected[i] {
					t.Fatalf(
						"index %d: expected %f, got %f",
						i,
						tt.expected[i],
						got[i],
					)
				}
			}
		})
	}
}

func TestSpectrumEnsurePowerRecalculates(t *testing.T) {
	spectrum := &Spectrum{
		Bins: []complex64{
			complex(3, 4),
		},
	}

	spectrum.EnsurePower()

	spectrum.Bins[0] =
		complex(6, 8)

	got :=
		spectrum.EnsurePower()

	if got[0] != 100 {
		t.Fatalf(
			"expected 100, got %f",
			got[0],
		)
	}
}

func TestSpectrumRecomputePower(t *testing.T) {
	tests := []struct {
		name             string
		spectrum         *Spectrum
		expected         []float32
		expectedCapacity int
		wantNil          bool
	}{
		{
			name:     "nil",
			spectrum: nil,
			wantNil:  true,
		},
		{
			name: "allocate new power slice",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
					complex(5, 12),
				},
			},
			expected: []float32{
				25,
				169,
			},
			expectedCapacity: 2,
		},
		{
			name: "reuse existing capacity",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
					complex(5, 12),
				},
				Power: make(
					[]float32,
					0,
					4,
				),
			},
			expected: []float32{
				25,
				169,
			},
			expectedCapacity: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got :=
				tt.spectrum.RecomputePower()

			if tt.wantNil {
				if got != nil {
					t.Fatalf(
						"expected nil, got %#v",
						got,
					)
				}

				return
			}

			if len(got) != len(tt.expected) {
				t.Fatalf(
					"expected length %d, got %d",
					len(tt.expected),
					len(got),
				)
			}

			if cap(got) != tt.expectedCapacity {
				t.Fatalf(
					"expected capacity %d, got %d",
					tt.expectedCapacity,
					cap(got),
				)
			}

			for i := range tt.expected {
				if got[i] != tt.expected[i] {
					t.Fatalf(
						"index %d: expected %f, got %f",
						i,
						tt.expected[i],
						got[i],
					)
				}
			}
		})
	}
}

func TestSpectrumMagnitude(t *testing.T) {
	tests := []struct {
		name     string
		spectrum *Spectrum
		bin      int
		expected float32
	}{
		{
			name:     "nil",
			spectrum: nil,
			bin:      0,
			expected: 0,
		},
		{
			name: "negative bin",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
				},
			},
			bin:      -1,
			expected: 0,
		},
		{
			name: "out of range",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
				},
			},
			bin:      1,
			expected: 0,
		},
		{
			name: "3 4 5 triangle",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
				},
			},
			bin:      0,
			expected: 5,
		},
		{
			name: "sqrt two",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(1, 1),
				},
			},
			bin: 0,
			expected: float32(
				math.Sqrt(2),
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got :=
				tt.spectrum.Magnitude(
					tt.bin,
				)

			if math.Abs(
				float64(got-tt.expected),
			) > 1e-6 {
				t.Fatalf(
					"expected %f, got %f",
					tt.expected,
					got,
				)
			}
		})
	}
}

func TestSpectrumPowerAt(t *testing.T) {
	tests := []struct {
		name     string
		spectrum *Spectrum
		bin      int
		expected float32
	}{
		{
			name:     "nil",
			spectrum: nil,
			bin:      0,
			expected: 0,
		},
		{
			name: "negative bin",
			spectrum: &Spectrum{
				Bins: []complex64{1},
			},
			bin:      -1,
			expected: 0,
		},
		{
			name: "out of range",
			spectrum: &Spectrum{
				Bins: []complex64{1},
			},
			bin:      1,
			expected: 0,
		},
		{
			name: "calculate without cache",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
				},
			},
			bin:      0,
			expected: 25,
		},
		{
			name: "use cached power",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
				},
				Power: []float32{
					123,
				},
			},
			bin:      0,
			expected: 123,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got :=
				tt.spectrum.PowerAt(
					tt.bin,
				)

			if got != tt.expected {
				t.Fatalf(
					"expected %f, got %f",
					tt.expected,
					got,
				)
			}
		})
	}
}

func TestSpectrumClone(t *testing.T) {
	tests := []struct {
		name        string
		spectrum    *Spectrum
		expectNil   bool
		expectPower bool
	}{
		{
			name:      "nil",
			spectrum:  nil,
			expectNil: true,
		},
		{
			name: "without power",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
				},
				Format: Format{
					SampleRate: SampleRate8K,
					Channels:   1,
				},
				FFTSize:      256,
				WindowSize:   256,
				HopSize:      128,
				Sequence:     10,
				SampleOffset: 160,
			},
			expectPower: false,
		},
		{
			name: "with power",
			spectrum: &Spectrum{
				Bins: []complex64{
					complex(3, 4),
				},
				Power: []float32{
					25,
				},
				Format: Format{
					SampleRate: SampleRate8K,
					Channels:   1,
				},
				FFTSize:      256,
				WindowSize:   256,
				HopSize:      128,
				Sequence:     10,
				SampleOffset: 160,
			},
			expectPower: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clone :=
				tt.spectrum.Clone()

			if tt.expectNil {
				if clone != nil {
					t.Fatal(
						"expected nil clone",
					)
				}

				return
			}

			assertSpectrumClone(t, clone, tt.spectrum, tt.expectPower)
		})
	}
}

func assertSpectrumClone(t *testing.T, clone, original *Spectrum, expectPower bool) {
	t.Helper()

	if clone == nil {
		t.Fatal("expected clone")
	}

	if clone == original {
		t.Fatal("expected different Spectrum instance")
	}

	assertSpectrumCloneFields(t, clone, original)
	assertSpectrumCloneBuffers(t, clone, original, expectPower)
}

func assertSpectrumCloneFields(t *testing.T, clone, original *Spectrum) {
	t.Helper()

	if clone.Format != original.Format {
		t.Fatal("format not copied")
	}

	if clone.FFTSize != original.FFTSize {
		t.Fatal("FFT size not copied")
	}

	if clone.WindowSize != original.WindowSize {
		t.Fatal("window size not copied")
	}

	if clone.HopSize != original.HopSize {
		t.Fatal("hop size not copied")
	}

	if clone.Sequence != original.Sequence {
		t.Fatal("sequence not copied")
	}

	if clone.SampleOffset != original.SampleOffset {
		t.Fatal("sample offset not copied")
	}

	if len(clone.Bins) != len(original.Bins) {
		t.Fatal("bins not copied")
	}
}

func assertSpectrumCloneBuffers(
	t *testing.T,
	clone, original *Spectrum,
	expectPower bool,
) {
	t.Helper()

	if expectPower && clone.Power == nil {
		t.Fatal("expected power copy")
	}

	if !expectPower && clone.Power != nil {
		t.Fatal("expected nil power")
	}
}

func TestSpectrumCloneDeepCopy(t *testing.T) {
	tests := []struct {
		name      string
		withPower bool
	}{
		{
			name:      "bins",
			withPower: false,
		},
		{
			name:      "bins and power",
			withPower: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spectrum := &Spectrum{
				Bins: []complex64{
					complex(3, 4),
				},
			}

			if tt.withPower {
				spectrum.Power =
					[]float32{
						25,
					}
			}

			clone :=
				spectrum.Clone()

			clone.Bins[0] =
				complex(100, 200)

			if spectrum.Bins[0] ==
				clone.Bins[0] {

				t.Fatal(
					"bins were not deep copied",
				)
			}

			if tt.withPower {
				clone.Power[0] = 999

				if spectrum.Power[0] ==
					clone.Power[0] {

					t.Fatal(
						"power was not deep copied",
					)
				}
			}
		})
	}
}
