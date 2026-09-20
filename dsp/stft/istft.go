package stft

import (
	"fmt"

	"github.com/sghaida/noise-cancelation/audio"
)

// Synthesizer reconstructs streaming PCM samples from frequency domain
// spectra using inverse FFT and overlap add
type Synthesizer struct {
	fftSize int
	hopSize int

	window []float32

	overlap      []float32
	windowWeight []float32
	// fftScratch stores reusable inverse FFT input and output data
	fftScratch []complex64
	pending    bool
}

// NewSynthesizer creates an inverse stft processor using the same FFT
// and hop sizes as the corresponding Analyzer
func NewSynthesizer(fftSize int, hopSize int) (*Synthesizer, error) {
	if fftSize <= 0 {
		return nil, fmt.Errorf("FFT size must be greater than zero")
	}

	if !isPowerOfTwo(fftSize) {
		return nil, fmt.Errorf("FFT size %d must be a power of two", fftSize)
	}

	if hopSize <= 0 {
		return nil, fmt.Errorf("hop size must be greater than zero")
	}

	if hopSize > fftSize {
		return nil, fmt.Errorf("hop size %d cannot exceed FFT size %d", hopSize, fftSize)
	}

	return &Synthesizer{
		fftSize:      fftSize,
		hopSize:      hopSize,
		window:       hannWindow(fftSize),
		overlap:      make([]float32, fftSize),
		windowWeight: make([]float32, fftSize),
		fftScratch:   make([]complex64, fftSize),
	}, nil
}

// Process converts one spectrum back into time domain PCM samples
//
// the returned slice contains hopSize samples because the remaining
// samples participate in overlap add with subsequent frames
// x[n]= 1/N∑​X[k]e^(j2πkn/N)
func (s *Synthesizer) Process(spectrum audio.Spectrum) ([]float32, error) {
	expectedBins := s.fftSize/2 + 1

	if len(spectrum.Bins) != expectedBins {
		return nil, fmt.Errorf(
			"spectrum contains %d bins, want %d", len(spectrum.Bins), expectedBins,
		)
	}

	fullSpectrum := s.fftScratch
	// Clear previous inverse FFT data before rebuilding the full spectrum
	clear(fullSpectrum)
	copy(fullSpectrum, spectrum.Bins)

	// Reconstruct the negative-frequency half using conjugate symmetry
	//
	// bin 0 is DC and bin fftSize/2 is Nyquist so neither is mirrored
	for i := 1; i < s.fftSize/2; i++ {
		value := spectrum.Bins[i]
		fullSpectrum[s.fftSize-i] = complex(real(value), -imag(value))
	}

	if err := ifft(fullSpectrum); err != nil {
		return nil, fmt.Errorf("calculate inverse FFT: %w", err)
	}

	for i := 0; i < s.fftSize; i++ {
		// Apply the synthesis window and accumulate the frame into
		// the overlap buffer
		windowed := real(fullSpectrum[i]) * s.window[i]

		s.overlap[i] += windowed

		// Track the squared window contribution so overlapping frames
		// can be normalized during reconstruction
		s.windowWeight[i] += s.window[i] * s.window[i]
	}
	s.pending = true

	output := make([]float32, s.hopSize)

	for i := range output {
		weight := s.windowWeight[i]

		if weight > 1e-8 {
			output[i] = s.overlap[i] / weight
		}
	}

	// Shift the overlap buffers left by hopSize so the tail of the
	// current frame can overlap with the next reconstructed frame
	copy(s.overlap, s.overlap[s.hopSize:])

	copy(s.windowWeight, s.windowWeight[s.hopSize:])

	clear(s.overlap[s.fftSize-s.hopSize:])

	clear(s.windowWeight[s.fftSize-s.hopSize:])

	return output, nil
}

// Flush returns the remaining overlap add samples
// the synthesizer state is cleared after flushing
func (s *Synthesizer) Flush() []float32 {
	if !s.pending {
		return nil
	}

	remaining := s.fftSize - s.hopSize
	if remaining <= 0 {
		s.Reset()
		return nil
	}

	output := make([]float32, remaining)

	for i := range output {
		weight := s.windowWeight[i]

		if weight > 1e-8 {
			output[i] = s.overlap[i] / weight
		}
	}

	s.Reset()

	return output
}

// Reset clears all overlap add reconstruction state
func (s *Synthesizer) Reset() {
	clear(s.overlap)
	clear(s.windowWeight)
	s.pending = false
}
