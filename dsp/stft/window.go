package stft

import "math"

// hannWindow create a periodic Hann window for the requested frame size
//
// The Hann window reduce spectral leakage (abrupt cut offs between different frames)
// before the FFT by gradually reducing the signal towards zero at both edges of the analysis frame
func hannWindow(size int) []float32 {
	if size <= 0 {
		return nil
	}

	window := make([]float32, size)

	if size == 1 {
		window[0] = 1

		return window
	}

	// 0.5 - 0.5* math.Cos(2 * math.Pi *float64(i) / float64(size-1))
	// i/size -1 => gives us a value between 0 and 1, 0 in the first sample and 1 in the last sample
	// 2 pi => we are working on a full cosine cycle => this will set the boundaries of the wave
	// 			start 1, quarter 0, middle -1, 3/4  0 and finally at the end is 1
	// -0.5 cos => flip and scale
	// 0.5 sgift the curve upward between 0 and 1
	for i := range window {
		window[i] = float32(
			0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(size-1)),
		)
	}

	return window
}
