package stft

import (
	"fmt"
	"math"
)

// fft calculates the discrete Fourier transform using an iterative
// radix-2 Cooley-Tukey FFT
// The input length must be a power of two as the algorithm is divide and conquer
// original formula X[k]= n=0∑N−1​x[n]e^(−j2πkn/N)
func fft(samples []complex64) error {
	size := len(samples)

	if size == 0 {
		return nil
	}

	if !isPowerOfTwo(size) {
		return fmt.Errorf("FFT size %d must be a power of two", size)
	}

	bitReverse(samples)

	for length := 2; length <= size; length <<= 1 {
		halfLength := length / 2

		// calculate the rotation required
		angle := -2 * math.Pi / float64(length)

		// euler equation ejθ=cos(θ)+jsin(θ)
		// which gives us e^*−j2π/length)
		// to visualize if angle = -π/2 then twiddleStep ≈ 0 - 1i
		// which means the rotation of -90°
		twiddleStep := complex(float32(math.Cos(angle)), float32(math.Sin(angle)))

		for start := 0; start < size; start += length {
			twiddle := complex64(1)

			for i := range halfLength {
				even := samples[start+i]
				odd := twiddle * samples[start+i+halfLength]
				samples[start+i] = even + odd
				samples[start+i+halfLength] = even - odd
				twiddle *= twiddleStep
			}
		}
	}

	return nil
}

// ifft calculates the inverse Fourier transform
//
//	IFFT(X) = conjugate(FFT(conjugate(X))) / N
func ifft(samples []complex64) error {
	size := len(samples)

	if size == 0 {
		return nil
	}

	for i := range samples {
		samples[i] = complex(real(samples[i]), -imag(samples[i]))
	}

	if err := fft(samples); err != nil {
		return err
	}

	scale := float32(size)

	for i := range samples {
		samples[i] = complex(real(samples[i])/scale, -imag(samples[i])/scale)
	}

	return nil
}

// bitReverse rearranges FFT input elements into bit-reversed order
//
// This ordering is required by the iterative radix-2 FFT implementation
// as this enables combining the neighboring values efficiently
// example [0, 1, 2, 3, 4, 5, 6, 7] => [0, 4, 2, 6, 1, 5, 3, 7]
// which enables efficient computation for odd and even values
func bitReverse(samples []complex64) {
	size := len(samples)

	j := 0

	for i := 1; i < size; i++ {
		bit := size >> 1

		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}

		j ^= bit

		if i < j {
			samples[i], samples[j] = samples[j], samples[i]
		}
	}
}

// isPowerOfTwo reports whether a number can be represented as 2^n
func isPowerOfTwo(value int) bool {
	return value > 0 && value&(value-1) == 0
}
