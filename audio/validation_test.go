package audio_test

import (
	"testing"
	"time"

	"github.com/sghaida/noise-cancelation/audio"
)

func TestValidateFrequency(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate8K,
		Channels:   1,
	}

	tests := []struct {
		name    string
		hz      float64
		wantErr bool
	}{
		{
			name: "zero",
			hz:   0,
		},
		{
			name: "80 Hz",
			hz:   80,
		},
		{
			name: "3999 Hz",
			hz:   3999,
		},
		{
			name:    "Nyquist",
			hz:      4000,
			wantErr: true,
		},
		{
			name:    "above Nyquist",
			hz:      5000,
			wantErr: true,
		},
		{
			name:    "negative",
			hz:      -1,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err :=
				audio.ValidateFrequency(
					tt.hz,
					format,
				)

			if tt.wantErr &&
				err == nil {

				t.Fatal(
					"expected error",
				)
			}

			if !tt.wantErr &&
				err != nil {

				t.Fatalf(
					"unexpected error: %v",
					err,
				)
			}
		})
	}
}

func TestValidateFrequencyDifferentSampleRates(t *testing.T) {
	tests := []struct {
		rate      audio.SampleRate
		frequency float64
		wantErr   bool
	}{
		{
			rate:      audio.SampleRate8K,
			frequency: 5000,
			wantErr:   true,
		},
		{
			rate:      audio.SampleRate16K,
			frequency: 5000,
			wantErr:   false,
		},
		{
			rate:      audio.SampleRate48K,
			frequency: 20000,
			wantErr:   false,
		},
		{
			rate:      audio.SampleRate48K,
			frequency: 24000,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		format := audio.Format{
			SampleRate: tt.rate,
			Channels:   1,
		}

		err :=
			audio.ValidateFrequency(
				tt.frequency,
				format,
			)

		if tt.wantErr && err == nil {
			t.Fatalf(
				"rate=%d frequency=%f: expected error",
				tt.rate,
				tt.frequency,
			)
		}

		if !tt.wantErr && err != nil {
			t.Fatalf(
				"rate=%d frequency=%f: unexpected error: %v",
				tt.rate,
				tt.frequency,
				err,
			)
		}
	}
}

func TestValidateDuration(t *testing.T) {
	tests := []struct {
		name     string
		duration time.Duration
		wantErr  bool
	}{
		{
			name:     "20 ms",
			duration: 20 * time.Millisecond,
		},
		{
			name:     "10 seconds",
			duration: 10 * time.Second,
		},
		{
			name:     "zero",
			duration: 0,
			wantErr:  true,
		},
		{
			name:     "negative",
			duration: -time.Millisecond,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err :=
				audio.ValidateDuration(
					"test",
					tt.duration,
				)

			if tt.wantErr &&
				err == nil {

				t.Fatal(
					"expected error",
				)
			}

			if !tt.wantErr &&
				err != nil {

				t.Fatalf(
					"unexpected error: %v",
					err,
				)
			}
		})
	}
}

func TestValidateFrameSizeMono(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate8K,
		Channels:   1,
	}

	err :=
		audio.ValidateFrameSize(
			make([]float32, 160),
			format,
		)

	if err != nil {
		t.Fatalf(
			"unexpected error: %v",
			err,
		)
	}
}

func TestValidateFrameSizeStereo(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate48K,
		Channels:   2,
	}

	err :=
		audio.ValidateFrameSize(
			make([]float32, 1920),
			format,
		)

	if err != nil {
		t.Fatalf(
			"unexpected error: %v",
			err,
		)
	}
}

func TestValidateFrameSizeStereoOddSamples(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate48K,
		Channels:   2,
	}

	err :=
		audio.ValidateFrameSize(
			make([]float32, 1919),
			format,
		)

	if err == nil {
		t.Fatal(
			"expected error",
		)
	}
}

func TestValidateFrameSizeEmpty(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate8K,
		Channels:   1,
	}

	err :=
		audio.ValidateFrameSize(
			nil,
			format,
		)

	if err == nil {
		t.Fatal(
			"expected error",
		)
	}
}

func TestValidateFrameSizeInvalidChannels(t *testing.T) {
	format := audio.Format{
		SampleRate: audio.SampleRate8K,
		Channels:   0,
	}

	err :=
		audio.ValidateFrameSize(
			[]float32{0},
			format,
		)

	if err == nil {
		t.Fatal(
			"expected error",
		)
	}
}
