package audio_test

import (
	"testing"
	"time"

	"github.com/sghaida/noise-cancelation/audio"
)

func TestNewFrame(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate8K,
		Channels:   1,
	}

	samples := make([]float32, 160)

	frame, err := audio.NewFrame(
		samples,
		format,
		10,
		1600,
	)

	if err != nil {
		t.Fatalf(
			"unexpected error: %v",
			err,
		)
	}

	if frame.Sequence != 10 {
		t.Fatalf(
			"expected sequence 10, got %d",
			frame.Sequence,
		)
	}

	if frame.SampleOffset != 1600 {
		t.Fatalf(
			"expected offset 1600, got %d",
			frame.SampleOffset,
		)
	}

	if len(frame.Samples) != 160 {
		t.Fatalf(
			"expected 160 samples, got %d",
			len(frame.Samples),
		)
	}
}

func TestNewFrameEmptySamples(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate8K,
		Channels:   1,
	}

	_, err := audio.NewFrame(
		nil,
		format,
		0,
		0,
	)

	if err == nil {
		t.Fatal("expected error")
	}
}

func TestFrameValidateNil(t *testing.T) {
	var frame *audio.Frame

	if err := frame.Validate(); err == nil {
		t.Fatal("expected error")
	}
}

func TestFrameValidateInvalidFormat(t *testing.T) {
	frame := &audio.Frame{
		Samples: []float32{0},
		Format: audio.Format{
			SampleRate: 12345,
			Channels:   1,
		},
	}

	if err := frame.Validate(); err == nil {
		t.Fatal("expected format validation error")
	}
}

func TestFrameValidateStereoSampleCount(t *testing.T) {
	frame := &audio.Frame{
		Samples: []float32{
			0,
			0,
			0,
		},
		Format: audio.Format{
			SampleRate: audio.SampleRate48K,
			Channels:   2,
		},
	}

	if err := frame.Validate(); err == nil {
		t.Fatal(
			"expected error for non-divisible stereo samples",
		)
	}
}

func TestFrameSamplesPerChannelMono(t *testing.T) {
	frame := &audio.Frame{
		Samples: make([]float32, 320),
		Format: audio.Format{
			SampleRate: audio.SampleRate16K,
			Channels:   1,
		},
	}

	if got := frame.SamplesPerChannel(); got != 320 {
		t.Fatalf(
			"expected 320, got %d",
			got,
		)
	}
}

func TestFrameSamplesPerChannelStereo(t *testing.T) {
	frame := &audio.Frame{
		Samples: make([]float32, 1920),
		Format: audio.Format{
			SampleRate: audio.SampleRate48K,
			Channels:   2,
		},
	}

	if got := frame.SamplesPerChannel(); got != 960 {
		t.Fatalf(
			"expected 960, got %d",
			got,
		)
	}
}

func TestFrameSamplesPerChannelNil(t *testing.T) {
	var frame *audio.Frame

	if got := frame.SamplesPerChannel(); got != 0 {
		t.Fatalf(
			"expected 0 for nil frame, got %d",
			got,
		)
	}
}

func TestFrameSamplesPerChannelZeroChannels(t *testing.T) {
	frame := &audio.Frame{
		Samples: []float32{
			0.1,
			0.2,
		},
		Format: audio.Format{
			SampleRate: audio.SampleRate8K,
			Channels:   0,
		},
	}

	if got := frame.SamplesPerChannel(); got != 0 {
		t.Fatalf(
			"expected 0 for zero channels, got %d",
			got,
		)
	}
}

func TestFrameSamplesPerChannelNegativeChannels(t *testing.T) {
	frame := &audio.Frame{
		Samples: []float32{
			0.1,
			0.2,
		},
		Format: audio.Format{
			SampleRate: audio.SampleRate8K,
			Channels:   -1,
		},
	}

	if got := frame.SamplesPerChannel(); got != 0 {
		t.Fatalf(
			"expected 0 for negative channels, got %d",
			got,
		)
	}
}

func TestFrameDuration(t *testing.T) {
	tests := []struct {
		name     string
		rate     audio.SampleRate
		samples  int
		channels int
		expected time.Duration
	}{
		{
			name:     "8k mono",
			rate:     audio.SampleRate8K,
			samples:  160,
			channels: 1,
			expected: 20 * time.Millisecond,
		},
		{
			name:     "16k mono",
			rate:     audio.SampleRate16K,
			samples:  320,
			channels: 1,
			expected: 20 * time.Millisecond,
		},
		{
			name:     "48k stereo",
			rate:     audio.SampleRate48K,
			samples:  1920,
			channels: 2,
			expected: 20 * time.Millisecond,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			frame := &audio.Frame{
				Samples: make(
					[]float32,
					tt.samples,
				),
				Format: audio.Format{
					SampleRate: tt.rate,
					Channels:   tt.channels,
				},
			}

			if got := frame.Duration(); got != tt.expected {
				t.Fatalf(
					"expected %s, got %s",
					tt.expected,
					got,
				)
			}
		})
	}
}

func TestFrameDurationNil(t *testing.T) {
	var frame *audio.Frame

	if got := frame.Duration(); got != 0 {
		t.Fatalf(
			"expected zero duration for nil frame, got %s",
			got,
		)
	}
}

func TestFrameStartTime(t *testing.T) {
	frame := &audio.Frame{
		Format: audio.Format{
			SampleRate: audio.SampleRate8K,
			Channels:   1,
		},
		SampleOffset: 8000,
	}

	if got := frame.StartTime(); got != time.Second {
		t.Fatalf(
			"expected 1 second, got %s",
			got,
		)
	}
}

func TestFrameStartTimeNil(t *testing.T) {
	var frame *audio.Frame

	if got := frame.StartTime(); got != 0 {
		t.Fatalf(
			"expected zero start time for nil frame, got %s",
			got,
		)
	}
}

func TestFrameEndTime(t *testing.T) {
	frame := &audio.Frame{
		Samples: make([]float32, 160),
		Format: audio.Format{
			SampleRate: audio.SampleRate8K,
			Channels:   1,
		},
		SampleOffset: 8000,
	}

	expected :=
		time.Second +
			20*time.Millisecond

	if got := frame.EndTime(); got != expected {
		t.Fatalf(
			"expected %s, got %s",
			expected,
			got,
		)
	}
}

func TestFrameEndTimeNil(t *testing.T) {
	var frame *audio.Frame

	if got := frame.EndTime(); got != 0 {
		t.Fatalf(
			"expected zero end time for nil frame, got %s",
			got,
		)
	}
}

func TestFrameCloneDeepCopy(t *testing.T) {
	frame := &audio.Frame{
		Samples: []float32{
			0.1,
			0.2,
			0.3,
		},
		Format: audio.Format{
			SampleRate: audio.SampleRate8K,
			Channels:   1,
		},
		Sequence:     12,
		SampleOffset: 160,
	}

	clone := frame.Clone()

	if clone == frame {
		t.Fatal("clone should be a new object")
	}

	if clone.Sequence != frame.Sequence {
		t.Fatal("sequence was not copied")
	}

	if clone.SampleOffset != frame.SampleOffset {
		t.Fatal("sample offset was not copied")
	}

	clone.Samples[0] = 1

	if frame.Samples[0] == 1 {
		t.Fatal(
			"modifying clone changed original sample buffer",
		)
	}
}

func TestFrameCloneNil(t *testing.T) {
	var frame *audio.Frame

	if frame.Clone() != nil {
		t.Fatal("expected nil")
	}
}

func TestFrameResetSamples(t *testing.T) {
	frame := &audio.Frame{
		Samples: []float32{
			1,
			-0.5,
			0.25,
		},
	}

	frame.ResetSamples()

	for i, sample := range frame.Samples {
		if sample != 0 {
			t.Fatalf(
				"sample %d expected 0, got %f",
				i,
				sample,
			)
		}
	}
}

func TestFrameResetSamplesNil(t *testing.T) {
	var frame *audio.Frame

	// Should not panic.
	frame.ResetSamples()
}
