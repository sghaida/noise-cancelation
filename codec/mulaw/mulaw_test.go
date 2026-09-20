package mulaw

import (
	"math"
	"testing"
)

func TestMuLaw_Metadata(t *testing.T) {
	m := NewMuLaw()

	tests := []struct {
		name string
		got  any
		want any
	}{
		{
			name: "name",
			got:  m.Name(),
			want: "mulaw",
		},
		{
			name: "sample rate",
			got:  m.SampleRate(),
			want: 8000,
		},
		{
			name: "channels",
			got:  m.Channels(),
			want: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.want {
				t.Fatalf("got %v, want %v", tt.got, tt.want)
			}
		})
	}
}

func TestNewMuLaw(t *testing.T) {
	tests := []struct {
		name string
	}{
		{
			name: "returns codec instance",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewMuLaw()

			if got == nil {
				t.Fatal("NewMuLaw() returned nil")
			}
		})
	}
}

func TestMuLaw_Decode(t *testing.T) {
	m := NewMuLaw()

	tests := []struct {
		name  string
		input []byte
	}{
		{
			name:  "empty",
			input: nil,
		},
		{
			name: "silence",
			input: []byte{
				0xFF,
				0x7F,
			},
		},
		{
			name:  "positive and negative samples",
			input: []byte{0xCE, 0x4E},
		},
		{
			name:  "multiple samples",
			input: []byte{0xFF, 0xFE, 0xFD, 0xFC},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := m.Decode(tt.input)

			if len(got) != len(tt.input) {
				t.Fatalf(
					"Decode() len = %d, want %d",
					len(got), len(tt.input),
				)
			}

			for i, encoded := range tt.input {
				want := float32(ToPCM16(encoded)) / 32768.0

				if got[i] != want {
					t.Errorf(
						"Decode()[%d] = %f, want %f",
						i, got[i], want,
					)
				}
			}
		})
	}
}

func TestMuLaw_Encode(t *testing.T) {
	m := NewMuLaw()

	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "empty",
			input: nil,
		},
		{
			name:  "normal range",
			input: []float32{-1.0, -0.5, 0, 0.5, 1.0},
		},
		{
			name:  "clamps values outside normalized range",
			input: []float32{-2.0, 2.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := m.Encode(tt.input)

			if len(got) != len(tt.input) {
				t.Fatalf(
					"Encode() len = %d, want %d",
					len(got), len(tt.input),
				)
			}

			for i, sample := range tt.input {
				clamped := clampSample(sample)

				want := ToMuLaw(int16(clamped * 32767.0))

				if got[i] != want {
					t.Errorf(
						"Encode()[%d] = %#x, want %#x",
						i, got[i], want,
					)
				}
			}
		})
	}
}

func TestToPCM16(t *testing.T) {
	tests := []struct {
		name    string
		encoded byte
		want    int16
	}{
		{
			name:    "positive silence",
			encoded: 0xFF,
			want:    0,
		},
		{
			name:    "negative silence",
			encoded: 0x7F,
			want:    0,
		},
		{
			name:    "positive sample",
			encoded: 0xCE,
			want:    988,
		},
		{
			name:    "negative sample",
			encoded: 0x4E,
			want:    -988,
		},
		{
			name:    "maximum positive magnitude",
			encoded: 0x80,
			want:    32124,
		},
		{
			name:    "maximum negative magnitude",
			encoded: 0x00,
			want:    -32124,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ToPCM16(tt.encoded)

			if got != tt.want {
				t.Fatalf(
					"ToPCM16(%#x) = %d, want %d",
					tt.encoded, got, tt.want,
				)
			}
		})
	}
}

func TestToMuLaw(t *testing.T) {
	tests := []struct {
		name  string
		input int16
	}{
		{
			name:  "zero",
			input: 0,
		},
		{
			name:  "small positive",
			input: 100,
		},
		{
			name:  "small negative",
			input: -100,
		},
		{
			name:  "medium positive",
			input: 1000,
		},
		{
			name:  "medium negative",
			input: -1000,
		},
		{
			name:  "large positive",
			input: 30000,
		},
		{
			name:  "large negative",
			input: -30000,
		},
		{
			name:  "maximum int16",
			input: 32767,
		},
		{
			name:  "minimum int16",
			input: -32768,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded := ToMuLaw(tt.input)
			decoded := ToPCM16(encoded)

			diff := math.Abs(
				float64(tt.input) - float64(decoded),
			)

			// µ-law is lossy, especially for high magnitudes.
			if diff > 1500 {
				t.Errorf(
					"round trip too inaccurate: input=%d encoded=%#x decoded=%d diff=%.0f",
					tt.input, encoded, decoded, diff,
				)
			}
		})
	}
}

func TestDecode(t *testing.T) {
	src := []byte{0xFF, 0xFE, 0xFD, 0xFC}

	tests := []struct {
		name           string
		dst            []float32
		expectReuse    bool
		expectedLen    int
		expectedMinCap int
	}{
		{
			name:           "allocates when destination capacity is insufficient",
			dst:            make([]float32, 0, 1),
			expectReuse:    false,
			expectedLen:    len(src),
			expectedMinCap: len(src),
		},
		{
			name:           "reuses destination when capacity is sufficient",
			dst:            make([]float32, 0, len(src)),
			expectReuse:    true,
			expectedLen:    len(src),
			expectedMinCap: len(src),
		},
		{
			name:           "reuses larger destination",
			dst:            make([]float32, 0, 10),
			expectReuse:    true,
			expectedLen:    len(src),
			expectedMinCap: len(src),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var original *float32

			if cap(tt.dst) > 0 {
				full := tt.dst[:cap(tt.dst)]
				original = &full[0]
			}

			got := DecodeMuLaw(tt.dst, src)
			assertDecodedBuffer(t, got, original, src, tt.expectedLen, tt.expectedMinCap, tt.expectReuse)
		})
	}
}

func assertDecodedBuffer(
	t *testing.T,
	got []float32,
	original *float32,
	src []byte,
	expectedLen, expectedMinCap int,
	expectReuse bool,
) {
	t.Helper()

	if len(got) != expectedLen {
		t.Fatalf("Decode() len = %d, want %d", len(got), expectedLen)
	}

	if cap(got) < expectedMinCap {
		t.Fatalf("Decode() cap = %d, want at least %d", cap(got), expectedMinCap)
	}

	if expectReuse && &got[0] != original {
		t.Fatal("Decode() did not reuse destination buffer")
	}

	if !expectReuse && &got[0] == original {
		t.Fatal("Decode() unexpectedly reused insufficient buffer")
	}

	for i, encoded := range src {
		want := float32(ToPCM16(encoded)) / 32768.0

		if got[i] != want {
			t.Errorf("Decode()[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestEncode(t *testing.T) {
	src := []float32{-2.0, -0.5, 0, 0.5, 2.0}

	tests := []struct {
		name           string
		dst            []byte
		expectReuse    bool
		expectedLen    int
		expectedMinCap int
	}{
		{
			name:           "allocates when destination capacity is insufficient",
			dst:            make([]byte, 0, 1),
			expectReuse:    false,
			expectedLen:    len(src),
			expectedMinCap: len(src),
		},
		{
			name:           "reuses destination when capacity is sufficient",
			dst:            make([]byte, 0, len(src)),
			expectReuse:    true,
			expectedLen:    len(src),
			expectedMinCap: len(src),
		},
		{
			name:           "reuses larger destination",
			dst:            make([]byte, 0, 10),
			expectReuse:    true,
			expectedLen:    len(src),
			expectedMinCap: len(src),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var original *byte

			if cap(tt.dst) > 0 {
				full := tt.dst[:cap(tt.dst)]
				original = &full[0]
			}

			got := EncodeMuLaw(tt.dst, src)
			assertEncodedBuffer(t, got, original, src, tt.expectedLen, tt.expectedMinCap, tt.expectReuse)
		})
	}
}

func assertEncodedBuffer(
	t *testing.T,
	got []byte,
	original *byte,
	src []float32,
	expectedLen, expectedMinCap int,
	expectReuse bool,
) {
	t.Helper()

	if len(got) != expectedLen {
		t.Fatalf("Encode() len = %d, want %d", len(got), expectedLen)
	}

	if cap(got) < expectedMinCap {
		t.Fatalf("Encode() cap = %d, want at least %d", cap(got), expectedMinCap)
	}

	if expectReuse && &got[0] != original {
		t.Fatal("Encode() did not reuse destination buffer")
	}

	if !expectReuse && &got[0] == original {
		t.Fatal("Encode() unexpectedly reused insufficient buffer")
	}

	for i, sample := range src {
		clamped := clampSample(sample)
		want := ToMuLaw(int16(clamped * 32767.0))

		if got[i] != want {
			t.Errorf("Encode()[%d] = %#x, want %#x", i, got[i], want)
		}
	}
}

func TestClampSample(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{
			name:  "below minimum",
			input: -2.0,
			want:  -1.0,
		},
		{
			name:  "minimum",
			input: -1.0,
			want:  -1.0,
		},
		{
			name:  "negative within range",
			input: -0.5,
			want:  -0.5,
		},
		{
			name:  "zero",
			input: 0,
			want:  0,
		},
		{
			name:  "positive within range",
			input: 0.5,
			want:  0.5,
		},
		{
			name:  "maximum",
			input: 1.0,
			want:  1.0,
		},
		{
			name:  "above maximum",
			input: 2.0,
			want:  1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := clampSample(tt.input)

			if got != tt.want {
				t.Fatalf(
					"clampSample(%f) = %f, want %f",
					tt.input, got, tt.want,
				)
			}
		})
	}
}
