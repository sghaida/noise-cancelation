package audio_test

import (
	"math"
	"testing"

	"github.com/sghaida/noise-cancelation/audio"
)

func closeFloat32(
	a float32,
	b float32,
	epsilon float32,
) bool {
	diff := a - b

	if diff < 0 {
		diff = -diff
	}

	return diff <= epsilon
}

func TestClampSample(t *testing.T) {
	tests := []struct {
		input    float32
		expected float32
	}{
		{-2, -1},
		{-1, -1},
		{-0.5, -0.5},
		{0, 0},
		{0.5, 0.5},
		{1, 1},
		{2, 1},
	}

	for _, tt := range tests {
		got :=
			audio.ClampSample(
				tt.input,
			)

		if got != tt.expected {
			t.Fatalf(
				"input=%f expected=%f got=%f",
				tt.input,
				tt.expected,
				got,
			)
		}
	}
}

func TestClampSamples(t *testing.T) {
	samples := []float32{
		-2,
		-0.5,
		0,
		0.5,
		2,
	}

	audio.ClampSamples(samples)

	expected := []float32{
		-1,
		-0.5,
		0,
		0.5,
		1,
	}

	for i := range expected {
		if samples[i] != expected[i] {
			t.Fatalf(
				"index=%d expected=%f got=%f",
				i,
				expected[i],
				samples[i],
			)
		}
	}
}

func TestPeak(t *testing.T) {
	samples := []float32{
		-0.2,
		0.5,
		-0.9,
		0.4,
	}

	got :=
		audio.Peak(samples)

	if got != 0.9 {
		t.Fatalf(
			"expected 0.9, got %f",
			got,
		)
	}
}

func TestPeakEmpty(t *testing.T) {
	if got := audio.Peak(nil); got != 0 {
		t.Fatalf(
			"expected 0, got %f",
			got,
		)
	}
}

func TestRMS(t *testing.T) {
	samples := []float32{
		1,
		-1,
		1,
		-1,
	}

	got :=
		audio.RMS(samples)

	if !closeFloat32(
		got,
		1,
		1e-6,
	) {
		t.Fatalf(
			"expected 1, got %f",
			got,
		)
	}
}

func TestRMSSineEquivalent(t *testing.T) {
	// Simple approximation of samples at ±1 and 0
	// isn't sufficient for exact sine RMS, so generate one.
	const count = 10000

	samples :=
		make([]float32, count)

	for i := range samples {
		phase :=
			2 *
				math.Pi *
				float64(i) /
				float64(count)

		samples[i] =
			float32(
				math.Sin(phase),
			)
	}

	got := audio.RMS(samples)

	expected :=
		float32(
			1 / math.Sqrt(2),
		)

	if !closeFloat32(
		got,
		expected,
		0.001,
	) {
		t.Fatalf(
			"expected %.6f, got %.6f",
			expected,
			got,
		)
	}
}

func TestRMSEmpty(t *testing.T) {
	if got := audio.RMS(nil); got != 0 {
		t.Fatalf(
			"expected 0, got %f",
			got,
		)
	}
}

func TestDBFS(t *testing.T) {
	tests := []struct {
		input    float32
		expected float32
	}{
		{1, 0},
		{0.5, -6.0206},
		{0.1, -20},
	}

	for _, tt := range tests {
		got :=
			audio.DBFS(
				tt.input,
			)

		if !closeFloat32(
			got,
			tt.expected,
			0.01,
		) {
			t.Fatalf(
				"input=%f expected %.4f dBFS, got %.4f",
				tt.input,
				tt.expected,
				got,
			)
		}
	}
}

func TestDBFSNegativeAmplitude(t *testing.T) {
	got :=
		audio.DBFS(-0.5)

	expected :=
		float32(-6.0206)

	if !closeFloat32(
		got,
		expected,
		0.01,
	) {
		t.Fatalf(
			"expected %.4f, got %.4f",
			expected,
			got,
		)
	}
}

func TestDBFSZeroIsFinite(t *testing.T) {
	got :=
		audio.DBFS(0)

	if math.IsInf(
		float64(got),
		0,
	) {
		t.Fatal(
			"expected finite dBFS value",
		)
	}

	if math.IsNaN(
		float64(got),
	) {
		t.Fatal(
			"expected non-NaN value",
		)
	}
}

func TestRMSDBFS(t *testing.T) {
	samples := []float32{
		0.5,
		-0.5,
		0.5,
		-0.5,
	}

	got :=
		audio.RMSDBFS(samples)

	if !closeFloat32(
		got,
		-6.0206,
		0.01,
	) {
		t.Fatalf(
			"expected -6.0206 dBFS, got %.4f",
			got,
		)
	}
}

func TestPeakDBFS(t *testing.T) {
	samples := []float32{
		0.1,
		-0.5,
		0.2,
	}

	got :=
		audio.PeakDBFS(samples)

	if !closeFloat32(
		got,
		-6.0206,
		0.01,
	) {
		t.Fatalf(
			"expected -6.0206 dBFS, got %.4f",
			got,
		)
	}
}

func TestDBToLinear(t *testing.T) {
	tests := []struct {
		db       float32
		expected float32
	}{
		{0, 1},
		{-6.0206, 0.5},
		{-20, 0.1},
		{6.0206, 2},
	}

	for _, tt := range tests {
		got :=
			audio.DBToLinear(
				tt.db,
			)

		if !closeFloat32(
			got,
			tt.expected,
			0.001,
		) {
			t.Fatalf(
				"%f dB expected %f, got %f",
				tt.db,
				tt.expected,
				got,
			)
		}
	}
}

func TestLinearToDB(t *testing.T) {
	got :=
		audio.LinearToDB(
			0.5,
		)

	if !closeFloat32(
		got,
		-6.0206,
		0.01,
	) {
		t.Fatalf(
			"expected -6.0206, got %.4f",
			got,
		)
	}
}

func TestApplyGain(t *testing.T) {
	samples := []float32{
		1,
		-1,
		0.5,
	}

	audio.ApplyGain(
		samples,
		0.5,
	)

	expected := []float32{
		0.5,
		-0.5,
		0.25,
	}

	for i := range samples {
		if samples[i] != expected[i] {
			t.Fatalf(
				"index=%d expected=%f got=%f",
				i,
				expected[i],
				samples[i],
			)
		}
	}
}

func TestApplyGainDB(t *testing.T) {
	samples := []float32{
		1,
		-1,
	}

	audio.ApplyGainDB(
		samples,
		-6.0206,
	)

	if !closeFloat32(
		samples[0],
		0.5,
		0.001,
	) {
		t.Fatalf(
			"expected approximately 0.5, got %f",
			samples[0],
		)
	}

	if !closeFloat32(
		samples[1],
		-0.5,
		0.001,
	) {
		t.Fatalf(
			"expected approximately -0.5, got %f",
			samples[1],
		)
	}
}

func TestCopySamples(t *testing.T) {
	src := []float32{
		0.1,
		0.2,
		0.3,
	}

	got :=
		audio.CopySamples(
			nil,
			src,
		)

	if len(got) != len(src) {
		t.Fatalf(
			"expected len %d, got %d",
			len(src),
			len(got),
		)
	}

	for i := range src {
		if got[i] != src[i] {
			t.Fatalf(
				"index=%d expected=%f got=%f",
				i,
				src[i],
				got[i],
			)
		}
	}

	got[0] = 1

	if src[0] == 1 {
		t.Fatal(
			"destination should not alias source",
		)
	}
}

func TestCopySamplesReusesCapacity(t *testing.T) {
	dst :=
		make([]float32, 0, 10)

	src :=
		[]float32{
			1,
			2,
			3,
		}

	got :=
		audio.CopySamples(
			dst,
			src,
		)

	if len(got) != 3 {
		t.Fatalf(
			"expected len 3, got %d",
			len(got),
		)
	}

	if cap(got) < 10 {
		t.Fatalf(
			"expected existing capacity to be reused",
		)
	}
}
