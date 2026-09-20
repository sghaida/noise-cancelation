package stft

import (
	"fmt"

	"github.com/sghaida/noise-cancelation/audio"
)

// Analyzer converts streaming PCM samples into overlapping frequency domain
// spectra using a Hann window followed by an FFT
//
// samples may arrive in arbitrary frame sizes such as the 160 sample
// 20 ms frames produced by an 8 kHz Twilio Media Stream
//
// analyzer maintains state between Process calls and should not be used
// concurrently by multiple audio streams
type Analyzer struct {
	fftSize int
	hopSize int

	window []float32
	buffer []float32

	// fftScratch stores temporary FFT input and output data
	// and is reused for every transformed frame
	fftScratch []complex64
}

// New creates an stft analyzer
//
// fftSize defines the number of samples used for each Fourier transform
// hopSize defines how many new samples are consumed between consecutive
// analysis frames
//
// a common configuration for 8 kHz speech is
//
//	fftSize = 256
//	hopSize = 128
//
// this produces a 32 ms analysis window with 50% overlap
func New(fftSize int, hopSize int) (*Analyzer, error) {
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
		return nil, fmt.Errorf(
			"hop size %d cannot exceed FFT size %d", hopSize, fftSize,
		)
	}

	return &Analyzer{
		fftSize: fftSize,
		hopSize: hopSize,
		window:  hannWindow(fftSize),
		buffer:  make([]float32, 0, fftSize*2),
		// Temporary FFT storage is allocated once and reused for every
		// transformed frame
		fftScratch: make([]complex64, fftSize),
	}, nil
}

// Process appends streaming PCM samples and returns every complete spectrum
// that can be produced from the available buffered samples
//
// samples that do not yet complete an FFT frame remain buffered until
// the next Process call
func (a *Analyzer) Process(samples []float32) ([]audio.Spectrum, error) {
	if len(samples) == 0 {
		return nil, nil
	}

	a.buffer = append(a.buffer, samples...)

	frameCount := 0

	if len(a.buffer) >= a.fftSize {
		frameCount = 1 + (len(a.buffer)-a.fftSize)/a.hopSize
	}

	spectra := make([]audio.Spectrum, 0, frameCount)

	consumed := 0

	for len(a.buffer)-consumed >= a.fftSize {
		frame := a.buffer[consumed : consumed+a.fftSize]

		spectrum, err := a.transform(frame)
		if err != nil {
			a.compactBuffer(consumed)
			return nil, err
		}

		spectra = append(spectra, spectrum)
		consumed += a.hopSize
	}

	a.compactBuffer(consumed)

	return spectra, nil
}

// transform apply the Hann window and performs the FFT for one
// complete analysis frame
func (a *Analyzer) transform(samples []float32) (audio.Spectrum, error) {
	// Reuse the FFT working buffer allocated when the analyzer was created
	bins := a.fftScratch

	for i, sample := range samples {
		// Window the sample before FFT to reduce spectral leakage
		windowedSample := sample * a.window[i]

		bins[i] = complex(windowedSample, 0)
	}

	if err := fft(bins); err != nil {
		return audio.Spectrum{}, fmt.Errorf("calculate FFT: %w", err)
	}

	// For real-valued audio input, the upper half of the FFT mirrors
	// the lower half so only bins 0 through Nyquist need to be retained
	//
	// For fftSize 256 this gives:
	//
	// 256 / 2 + 1 = 129 bins
	usefulBins := a.fftSize/2 + 1

	spectrum := audio.Spectrum{
		Bins: append([]complex64(nil), bins[:usefulBins]...),
	}

	spectrum.RecomputePower()

	return spectrum, nil
}

// Flush emits the final partially buffered analysis frame
// remaining samples are zero padded to fftSize
func (a *Analyzer) Flush() ([]audio.Spectrum, error) {
	if len(a.buffer) == 0 {
		return nil, nil
	}

	frame := make([]float32, a.fftSize)
	copy(frame, a.buffer)

	spectrum, err := a.transform(frame)
	if err != nil {
		return nil, fmt.Errorf("flush STFT frame: %w", err)
	}

	a.buffer = a.buffer[:0]

	return []audio.Spectrum{spectrum}, nil
}

// Reset clears any partially buffered analysis frame
func (a *Analyzer) Reset() {
	a.buffer = a.buffer[:0]
}

// FFTSize returns the configured FFT size
func (a *Analyzer) FFTSize() int {
	return a.fftSize
}

// HopSize returns the configured number of samples advanced between
// consecutive STFT frames
func (a *Analyzer) HopSize() int {
	return a.hopSize
}

// compactBuffer removes consumed samples while retaining the backing buffer
func (a *Analyzer) compactBuffer(consumed int) {
	if consumed <= 0 {
		return
	}

	remaining := len(a.buffer) - consumed
	copy(a.buffer[:remaining], a.buffer[consumed:])
	a.buffer = a.buffer[:remaining]
}
