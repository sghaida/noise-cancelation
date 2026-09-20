package audio

import "math"

// ClampSample constrains normalized PCM to
//
//	[-1, +1]
func ClampSample(sample float32) float32 {
	if sample > 1 {
		return 1
	}

	if sample < -1 {
		return -1
	}

	return sample
}

// ClampSamples constrains all samples in place
func ClampSamples(samples []float32) {
	for i := range samples {
		samples[i] = ClampSample(samples[i])
	}
}

// Peak calculates absolute peak amplitude
func Peak(samples []float32) float32 {
	var peak float32

	for _, sample := range samples {
		v := abs32(sample)

		if v > peak {
			peak = v
		}
	}

	return peak
}

// RMS returns root mean square amplitude RMS = sqrt(sum(x²) / N)
func RMS(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}

	var sum float64

	for _, sample := range samples {
		v := float64(sample)
		sum += v * v
	}

	return float32(
		math.Sqrt(sum / float64(len(samples))),
	)
}

// DBFS converts a normalized amplitude into decibels relative to full scale
// example
//
//	1.0 -> 0 dBFS
//	0.5 -> approximately -6.02 dBFS
func DBFS(amplitude float32) float32 {
	const epsilon = 1e-12

	v := float64(abs32(amplitude))

	if v < epsilon {
		v = epsilon
	}

	return float32(20 * math.Log10(v))
}

// RMSDBFS calculates the RMS signal level in dBFS
func RMSDBFS(samples []float32) float32 {
	return DBFS(RMS(samples))
}

// PeakDBFS calculates peak level in dBFS
func PeakDBFS(samples []float32) float32 {
	return DBFS(Peak(samples))
}

// DBToLinear converts dB gain to a linear multiplier
//
//	0 dB   -> 1
//	-6 dB  -> ~0.501
//	+6 dB  -> ~1.995
func DBToLinear(db float32) float32 {
	return float32(
		math.Pow(10, float64(db)/20.0),
	)
}

// LinearToDB converts a linear gain to dB
func LinearToDB(value float32) float32 {
	return DBFS(value)
}

// ApplyGain applies a linear gain to samples in place
func ApplyGain(samples []float32, gain float32) {
	for i := range samples {
		samples[i] *= gain
	}
}

// ApplyGainDB applies a gain in dB
func ApplyGainDB(samples []float32, db float32) {
	ApplyGain(samples, DBToLinear(db))
}

// CopySamples copies samples into the destination and grows it if necessary
func CopySamples(dst []float32, src []float32) []float32 {
	if cap(dst) < len(src) {
		dst = make([]float32, len(src))
	} else {
		dst = dst[:len(src)]
	}

	copy(dst, src)

	return dst
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}

	return v
}
