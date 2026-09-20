package audio

import (
	"fmt"
	"time"
)

// Frame represents a block of time domain PCM samples
//
// samples should normally contain normalized float32 PCM: -1.0 ... +1.0
// for mono len(Samples) == number of samples
// for multichannel interleaved pcm [L0, R0, L1, R1, ...]
type Frame struct {
	// Samples contains normalized linear PCM
	Samples []float32

	// Format describes the audio represented by samples
	Format Format

	// Sequence identifies the logical frame within the stream
	// usually 0, 1, 2, 3, ...
	Sequence uint64

	// SampleOffset is the position of this frame first sample, measured per channel
	// example for 20 ms frames at 8 kHz
	//	frame 0 -> offset 0
	//	frame 1 -> offset 160
	//	frame 2 -> offset 320
	SampleOffset uint64
}

// NewFrame creates a frame
func NewFrame(samples []float32, format Format, sequence uint64, sampleOffset uint64) (*Frame, error) {
	frame := &Frame{
		Samples:      samples,
		Format:       format,
		Sequence:     sequence,
		SampleOffset: sampleOffset,
	}

	if err := frame.Validate(); err != nil {
		return nil, err
	}

	return frame, nil
}

// Validate checks whether the frame is structurally valid
func (f *Frame) Validate() error {
	if f == nil {
		return fmt.Errorf("audio: frame is nil")
	}

	if err := f.Format.Validate(); err != nil {
		return err
	}

	if len(f.Samples) == 0 {
		return fmt.Errorf("audio: frame contains no samples")
	}

	if len(f.Samples)%f.Format.Channels != 0 {
		return fmt.Errorf(
			"audio: sample count %d is not divisible by channels %d",
			len(f.Samples), f.Format.Channels,
		)
	}

	return nil
}

// SamplesPerChannel returns the number of samples in one channel
func (f *Frame) SamplesPerChannel() int {
	if f == nil || f.Format.Channels <= 0 {
		return 0
	}

	return len(f.Samples) / f.Format.Channels
}

// Duration returns the duration represented by the frame
func (f *Frame) Duration() time.Duration {
	if f == nil {
		return 0
	}

	return f.Format.DurationForSamples(f.SamplesPerChannel())
}

// StartTime returns the stream relative timestamp of this frame
func (f *Frame) StartTime() time.Duration {
	if f == nil {
		return 0
	}

	if f.Format.SampleRate <= 0 {
		return 0
	}

	maxInt := int(^uint(0) >> 1)

	if f.SampleOffset > uint64(maxInt) {
		return time.Duration(1<<63 - 1)
	}

	return f.Format.DurationForSamples(int(f.SampleOffset)) //nolint:gosec // sample offset is checked against the platform int limit
}

// EndTime returns the stream relative time at which this frame ends
func (f *Frame) EndTime() time.Duration {
	return f.StartTime() + f.Duration()
}

// Clone creates a deep copy to duplicate the sample buffer
func (f *Frame) Clone() *Frame {
	if f == nil {
		return nil
	}

	samples := make([]float32, len(f.Samples))

	copy(samples, f.Samples)

	return &Frame{
		Samples:      samples,
		Format:       f.Format,
		Sequence:     f.Sequence,
		SampleOffset: f.SampleOffset,
	}
}

// ResetSamples sets every sample to zero to enable buffer reuse
func (f *Frame) ResetSamples() {
	if f == nil {
		return
	}

	clear(f.Samples)
}
