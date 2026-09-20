package audio_test

import (
	"math"
	"testing"
	"time"

	"github.com/sghaida/noise-cancelation/audio"
)

func TestSampleRateIsSupported(t *testing.T) {
	tests := []struct {
		rate     audio.SampleRate
		expected bool
	}{
		{audio.SampleRate8K, true},
		{audio.SampleRate16K, true},
		{audio.SampleRate22050, true},
		{audio.SampleRate24K, true},
		{audio.SampleRate48K, true},
		{44100, false},
		{96000, false},
	}

	for _, tt := range tests {
		if got := tt.rate.IsSupported(); got != tt.expected {
			t.Fatalf(
				"rate %d: expected %v, got %v",
				tt.rate,
				tt.expected,
				got,
			)
		}
	}
}

func TestSampleRateNyquistHz(t *testing.T) {
	tests := []struct {
		rate     audio.SampleRate
		expected float64
	}{
		{audio.SampleRate8K, 4000},
		{audio.SampleRate16K, 8000},
		{audio.SampleRate22050, 11025},
		{audio.SampleRate24K, 12000},
		{audio.SampleRate48K, 24000},
	}

	for _, tt := range tests {
		got := tt.rate.NyquistHz()

		if got != tt.expected {
			t.Fatalf(
				"rate %d: expected %.2f Hz, got %.2f Hz",
				tt.rate,
				tt.expected,
				got,
			)
		}
	}
}

func TestSamplesForDuration(t *testing.T) {
	tests := []struct {
		rate     audio.SampleRate
		duration time.Duration
		expected int
	}{
		{
			audio.SampleRate8K,
			20 * time.Millisecond,
			160,
		},
		{
			audio.SampleRate16K,
			20 * time.Millisecond,
			320,
		},
		{
			audio.SampleRate22050,
			20 * time.Millisecond,
			441,
		},
		{
			audio.SampleRate24K,
			20 * time.Millisecond,
			480,
		},
		{
			audio.SampleRate48K,
			20 * time.Millisecond,
			960,
		},
		{
			audio.SampleRate8K,
			10 * time.Second,
			80000,
		},
		{
			audio.SampleRate48K,
			10 * time.Second,
			480000,
		},
	}

	for _, tt := range tests {
		got :=
			tt.rate.SamplesForDuration(
				tt.duration,
			)

		if got != tt.expected {
			t.Fatalf(
				"rate=%d duration=%s: expected %d, got %d",
				tt.rate,
				tt.duration,
				tt.expected,
				got,
			)
		}
	}
}

func TestSamplesForDurationInvalid(t *testing.T) {
	if got := audio.SampleRate8K.SamplesForDuration(0); got != 0 {
		t.Fatalf("expected 0, got %d", got)
	}

	if got := audio.SampleRate8K.SamplesForDuration(-time.Second); got != 0 {
		t.Fatalf("expected 0, got %d", got)
	}

	var rate audio.SampleRate

	if got := rate.SamplesForDuration(time.Second); got != 0 {
		t.Fatalf("expected 0, got %d", got)
	}
}

func TestDurationForSamples(t *testing.T) {
	tests := []struct {
		rate     audio.SampleRate
		samples  int
		expected time.Duration
	}{
		{
			audio.SampleRate8K,
			160,
			20 * time.Millisecond,
		},
		{
			audio.SampleRate16K,
			320,
			20 * time.Millisecond,
		},
		{
			audio.SampleRate22050,
			441,
			20 * time.Millisecond,
		},
		{
			audio.SampleRate24K,
			480,
			20 * time.Millisecond,
		},
		{
			audio.SampleRate48K,
			960,
			20 * time.Millisecond,
		},
	}

	for _, tt := range tests {
		got :=
			tt.rate.DurationForSamples(
				tt.samples,
			)

		diff :=
			math.Abs(
				float64(got - tt.expected),
			)

		// Allow nanosecond-level rounding differences.
		if diff > float64(time.Nanosecond) {
			t.Fatalf(
				"rate=%d samples=%d: expected %s, got %s",
				tt.rate,
				tt.samples,
				tt.expected,
				got,
			)
		}
	}
}

func TestDurationForSamplesInvalid(t *testing.T) {
	if got := audio.SampleRate8K.DurationForSamples(0); got != 0 {
		t.Fatalf(
			"expected 0 duration for zero samples, got %s",
			got,
		)
	}

	if got := audio.SampleRate8K.DurationForSamples(-1); got != 0 {
		t.Fatalf(
			"expected 0 duration for negative samples, got %s",
			got,
		)
	}

	var rate audio.SampleRate

	if got := rate.DurationForSamples(160); got != 0 {
		t.Fatalf(
			"expected 0 duration for zero sample rate, got %s",
			got,
		)
	}
}
func TestSampleRateString(t *testing.T) {
	got := audio.SampleRate16K.String()

	if got != "16000 Hz" {
		t.Fatalf(
			"expected %q, got %q",
			"16000 Hz",
			got,
		)
	}
}
