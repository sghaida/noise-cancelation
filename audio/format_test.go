package audio_test

import (
	"strings"
	"testing"
	"time"

	"github.com/sghaida/noise-cancelation/audio"
)

func TestNewFormat(t *testing.T) {
	tests := []struct {
		name       string
		sampleRate audio.SampleRate
		channels   int
		wantErr    bool
	}{
		{
			name:       "8k mono",
			sampleRate: audio.SampleRate8K,
			channels:   1,
		},
		{
			name:       "16k mono",
			sampleRate: audio.SampleRate16K,
			channels:   1,
		},
		{
			name:       "22050 mono",
			sampleRate: audio.SampleRate22050,
			channels:   1,
		},
		{
			name:       "24k mono",
			sampleRate: audio.SampleRate24K,
			channels:   1,
		},
		{
			name:       "48k stereo",
			sampleRate: audio.SampleRate48K,
			channels:   2,
		},
		{
			name:       "unsupported sample rate",
			sampleRate: 44100,
			channels:   1,
			wantErr:    true,
		},
		{
			name:       "zero sample rate",
			sampleRate: 0,
			channels:   1,
			wantErr:    true,
		},
		{
			name:       "zero channels",
			sampleRate: audio.SampleRate8K,
			channels:   0,
			wantErr:    true,
		},
		{
			name:       "negative channels",
			sampleRate: audio.SampleRate8K,
			channels:   -1,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := audio.NewFormat(
				tt.sampleRate,
				tt.channels,
			)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}

				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if got.SampleRate != tt.sampleRate {
				t.Fatalf(
					"expected sample rate %d, got %d",
					tt.sampleRate,
					got.SampleRate,
				)
			}

			if got.Channels != tt.channels {
				t.Fatalf(
					"expected %d channels, got %d",
					tt.channels,
					got.Channels,
				)
			}
		})
	}
}

func TestFormatValidate(t *testing.T) {
	valid := audio.Format{
		SampleRate: audio.SampleRate16K,
		Channels:   1,
	}

	if err := valid.Validate(); err != nil {
		t.Fatalf("expected valid format: %v", err)
	}
}

func TestFormatNyquistHz(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate48K,
		Channels:   1,
	}

	if got := format.NyquistHz(); got != 24000 {
		t.Fatalf(
			"expected 24000 Hz, got %.2f",
			got,
		)
	}
}

func TestFormatInterleavedSamplesForDuration(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate48K,
		Channels:   2,
	}

	got :=
		format.InterleavedSamplesForDuration(
			20 * time.Millisecond,
		)

	if got != 1920 {
		t.Fatalf(
			"expected 1920, got %d",
			got,
		)
	}
}

func TestFormatMonoStereo(t *testing.T) {
	mono := audio.Format{
		SampleRate: audio.SampleRate8K,
		Channels:   1,
	}

	if !mono.IsMono() {
		t.Fatal("expected mono")
	}

	if mono.IsStereo() {
		t.Fatal("did not expect stereo")
	}

	stereo := audio.Format{
		SampleRate: audio.SampleRate48K,
		Channels:   2,
	}

	if !stereo.IsStereo() {
		t.Fatal("expected stereo")
	}

	if stereo.IsMono() {
		t.Fatal("did not expect mono")
	}
}

func TestFormatString(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate16K,
		Channels:   1,
	}

	got := format.String()

	if !strings.Contains(got, "16000") {
		t.Fatalf(
			"expected sample rate in string, got %q",
			got,
		)
	}

	if !strings.Contains(got, "1") {
		t.Fatalf(
			"expected channel count in string, got %q",
			got,
		)
	}
}
