package audio

import (
	"fmt"
	"time"
)

// ValidateFrequency verifies that a frequency lies inside the valid range for the format
//
//	0 <= frequency < Nyquist
func ValidateFrequency(hz float64, format Format) error {
	if hz < 0 {
		return fmt.Errorf("audio: frequency cannot be negative: %.2f Hz", hz)
	}

	if hz >= format.NyquistHz() {
		return fmt.Errorf(
			"audio: frequency %.2f Hz must be below Nyquist %.2f Hz",
			hz, format.NyquistHz(),
		)
	}

	return nil
}

// ValidateDuration verifies a positive duration
func ValidateDuration(name string, duration time.Duration) error {
	if duration <= 0 {
		return fmt.Errorf("audio: %s must be greater than zero: %s", name, duration)
	}

	return nil
}

// ValidateFrameSize verifies that a sample slice contains complete
// multichannel samples
func ValidateFrameSize(samples []float32, format Format,
) error {
	if len(samples) == 0 {
		return fmt.Errorf("audio: no samples provided")
	}

	if format.Channels <= 0 {
		return fmt.Errorf("audio: invalid channel count: %d", format.Channels)
	}

	if len(samples)%format.Channels != 0 {
		return fmt.Errorf(
			"audio: %d samples cannot be evenly divided across %d channels",
			len(samples), format.Channels,
		)
	}

	return nil
}
