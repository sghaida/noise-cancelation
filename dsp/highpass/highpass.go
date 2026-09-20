package highpass

import "math"

// Filter removes dc offset and very low frequency components
//
// the filter keeps state between Process calls because audio arrives
// as consecutive frames from the same stream
type Filter struct {
	previousInput       float32
	previousOutput      float32
	feedbackCoefficient float32
}

// New creates a high pass filter for the given sample rate and cutoff frequency
func New(sampleRate int, cutoffHz float32) *Filter {
	return &Filter{
		feedbackCoefficient: calculateCoefficient(sampleRate, cutoffHz),
	}
}

// Process applies the high pass filter to the provided PCM samples
func (f *Filter) Process(samples []float32) ([]float32, error) {
	for i, inputSample := range samples {
		// Apply the first order DC blocking filter
		// previousInput contains the previous input sample
		// previousOutput contains the previous output sample
		// y[n]=x[n]−x[n−1]+r⋅y[n−1]
		outputSample := inputSample - f.previousInput +
			(f.feedbackCoefficient * f.previousOutput)

		f.previousInput = inputSample
		f.previousOutput = outputSample

		samples[i] = outputSample
	}

	return samples, nil
}

// Reset clears the filter state
func (f *Filter) Reset() {
	f.previousInput = 0
	f.previousOutput = 0
}

// calculateCoefficient calculate
// how much of the privious output sample should be remembered by the filter
// value close to 1 keeps more low-frequency content
// smaller value removes more low-frequency content
func calculateCoefficient(sampleRate int, cutoffHz float32) float32 {
	if sampleRate <= 0 {
		return 0
	}

	if cutoffHz <= 0 {
		return 1
	}

	// convert the cutoff frequency into a decay coefficient
	// r = e^(-2πfc/fs)
	// fc = cutoff frequency
	// fs = sample rate
	r := math.Exp(
		-2 * math.Pi * float64(cutoffHz) / float64(sampleRate),
	)

	return float32(r)
}
