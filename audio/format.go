package audio

import (
	"fmt"
	"time"
)

// Format describes linear PCM audio used internally by the DSP pipeline
// dsp layer should always operate on normalized float32 samples
// usually in the range: -1.0 <= sample <= 1.0
// encoding such as µlaw or PCM16 intentionally does not belong here
// as encoding belongs to the codec package
type Format struct {
	SampleRate SampleRate
	Channels   int
}

// NewFormat creates and validates an audio format
func NewFormat(sampleRate SampleRate, channels int) (Format, error) {
	format := Format{
		SampleRate: sampleRate,
		Channels:   channels,
	}

	if err := format.Validate(); err != nil {
		return Format{}, err
	}

	return format, nil
}

// Validate verifies the format
func (f Format) Validate() error {
	if f.SampleRate <= 0 {
		return fmt.Errorf("audio: invalid sample rate: %d", f.SampleRate)
	}

	if !f.SampleRate.IsSupported() {
		return fmt.Errorf("audio: unsupported sample rate: %d Hz", f.SampleRate)
	}

	if f.Channels <= 0 {
		return fmt.Errorf("audio: channels must be greater than zero: %d", f.Channels)
	}

	return nil
}

// NyquistHz returns the Nyquist frequency
// Nyquist = sampleRate / 2
// for 8 kHz: 8000 / 2 = 4000 Hz
func (f Format) NyquistHz() float64 {
	return f.SampleRate.NyquistHz()
}

// SamplesForDuration returns the number of samples per channel
// corresponding to a duration
// example: 16 kHz * 20 ms = 320 samples
func (f Format) SamplesForDuration(duration time.Duration) int {
	return f.SampleRate.SamplesForDuration(duration)
}

// DurationForSamples returns the duration represented by the specified
// number of samples per channel
func (f Format) DurationForSamples(samples int) time.Duration {
	return f.SampleRate.DurationForSamples(samples)
}

// InterleavedSamplesForDuration returns the number of samples for all channels
// example: 48 kHz, 20 ms (window), 2 channels
// gives: 960 samples/channel, 1920 interleaved float32 values
func (f Format) InterleavedSamplesForDuration(duration time.Duration) int {
	return f.SamplesForDuration(duration) * f.Channels
}

// IsMono reports whether the format has one channel
func (f Format) IsMono() bool {
	return f.Channels == 1
}

// IsStereo reports whether the format has two channels
func (f Format) IsStereo() bool {
	return f.Channels == 2
}

// String returns a human readable description
func (f Format) String() string {
	return fmt.Sprintf("%d Hz / %d channel(s)", f.SampleRate, f.Channels)
}

func roundPositive(v float64) int {
	return int(v + 0.5)
}
