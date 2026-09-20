package audio

import (
	"fmt"
	"time"
)

// SampleRate represents samples per second
// example: 8000 = 8 kHz, 16000 = 16 kHz, 48000 = 48 kHz
type SampleRate int

// supported sample rates in samples per second
const (
	SampleRate8K    SampleRate = 8_000
	SampleRate16K   SampleRate = 16_000
	SampleRate22050 SampleRate = 22_050
	SampleRate24K   SampleRate = 24_000
	SampleRate48K   SampleRate = 48_000
)

// SupportedSampleRates contains all supported sampling rates
var SupportedSampleRates = []SampleRate{
	SampleRate8K,
	SampleRate16K,
	SampleRate22050,
	SampleRate24K,
	SampleRate48K,
}

// IsSupported checks whether the sample rate is supported
func (s SampleRate) IsSupported() bool {
	switch s {
	case SampleRate8K,
		SampleRate16K,
		SampleRate22050,
		SampleRate24K,
		SampleRate48K:
		return true

	default:
		return false
	}
}

// NyquistHz returns sampleRate / 2
func (s SampleRate) NyquistHz() float64 {
	return float64(s) / 2.0
}

// SamplesForDuration converts time to a sample count
// it rounds to the closest whole sample
// example: 22050 Hz * 20 ms = 441 samples
func (s SampleRate) SamplesForDuration(duration time.Duration) int {
	if duration <= 0 || s <= 0 {
		return 0
	}

	return roundPositive(float64(s) * duration.Seconds())
}

// DurationForSamples converts samples to duration
func (s SampleRate) DurationForSamples(samples int) time.Duration {
	if samples <= 0 || s <= 0 {
		return 0
	}

	seconds := float64(samples) / float64(s)

	return time.Duration(seconds * float64(time.Second))
}

// String returns the sample rate as Hz
func (s SampleRate) String() string {
	return fmt.Sprintf("%d Hz", s)
}
