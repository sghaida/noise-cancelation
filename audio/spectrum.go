package audio

import (
	"fmt"
	"math"
)

// Spectrum represents one frequency domain analysis frame
// bins normally contains only the non-negative frequency portion
// of a real FFT: 0 ... Nyquist
// therefore
//
//	expected bin count = FFTSize/2 + 1
type Spectrum struct {
	// Bins contains complex FFT coefficients X[k]
	Bins []complex64

	// Power optionally caches |X[k]|²
	// when present len(Power) == len(Bins)
	Power []float32

	// Format contains the original audio format
	Format Format

	// FFTSize contains the transform size n
	FFTSize int

	// WindowSize contains the number of actual input samples used before optional zero padding
	WindowSize int

	// HopSize contains the analysis advance in samples
	HopSize int

	// Sequence identifies the frame analysis
	Sequence uint64

	// SampleOffset contains the stream position corresponding to the beginning of the STFT analysis window
	SampleOffset uint64
}

// NewSpectrum creates a spectrum
func NewSpectrum(
	bins []complex64, format Format,
	fftSize int, windowSize int, hopSize int,
) (*Spectrum, error) {
	spectrum := &Spectrum{
		Bins:       bins,
		Format:     format,
		FFTSize:    fftSize,
		WindowSize: windowSize,
		HopSize:    hopSize,
	}

	if err := spectrum.Validate(); err != nil {
		return nil, err
	}

	return spectrum, nil
}

// Validate checks the structural validity of the spectrum
func (s *Spectrum) Validate() error {
	if s == nil {
		return fmt.Errorf("audio: spectrum is nil")
	}

	if err := s.Format.Validate(); err != nil {
		return err
	}

	if err := s.validateDimensions(); err != nil {
		return err
	}

	return s.validateBins()
}

func (s *Spectrum) validateDimensions() error {
	if s.FFTSize <= 0 {
		return fmt.Errorf("audio: FFT size must be greater than zero")
	}

	if s.FFTSize%2 != 0 {
		return fmt.Errorf("audio: FFT size must be even: %d", s.FFTSize)
	}

	if s.WindowSize <= 0 {
		return fmt.Errorf("audio: window size must be greater than zero")
	}

	if s.WindowSize > s.FFTSize {
		return fmt.Errorf(
			"audio: window size %d cannot exceed FFT size %d",
			s.WindowSize, s.FFTSize,
		)
	}

	if s.HopSize <= 0 {
		return fmt.Errorf("audio: hop size must be greater than zero")
	}

	if s.HopSize > s.WindowSize {
		return fmt.Errorf(
			"audio: hop size %d cannot exceed window size %d",
			s.HopSize, s.WindowSize,
		)
	}

	return nil
}

func (s *Spectrum) validateBins() error {
	expectedBins := s.FFTSize/2 + 1

	if len(s.Bins) != expectedBins {
		return fmt.Errorf(
			"audio: expected %d FFT bins, received %d",
			expectedBins, len(s.Bins),
		)
	}

	if s.Power != nil &&
		len(s.Power) != len(s.Bins) {

		return fmt.Errorf(
			"audio: power spectrum contains %d values, expected %d",
			len(s.Power), len(s.Bins),
		)
	}

	return nil
}

// BinCount returns the number of positive frequency FFT bins
func (s *Spectrum) BinCount() int {
	if s == nil {
		return 0
	}

	return len(s.Bins)
}

// BinWidthHz returns the frequency resolution: Δf = fs / N
// Example: fs = 8000, N  = 256
// gives us Δf = 31.25 Hz
func (s *Spectrum) BinWidthHz() float64 {
	if s == nil || s.FFTSize <= 0 {
		return 0
	}

	return float64(s.Format.SampleRate) / float64(s.FFTSize)
}

// BinFrequency returns the center frequency represented by a bin
//
//	f[k] = k * fs / N
func (s *Spectrum) BinFrequency(bin int) float64 {
	if s == nil {
		return 0
	}

	if bin < 0 || bin >= len(s.Bins) {
		return 0
	}

	return float64(bin) * s.BinWidthHz()
}

// FrequencyBin returns the closest FFT bin corresponding to a frequency
func (s *Spectrum) FrequencyBin(hz float64) int {
	if s == nil || s.FFTSize <= 0 || hz <= 0 {
		return 0
	}

	nyquist := s.Format.NyquistHz()

	if hz >= nyquist {
		return len(s.Bins) - 1
	}

	bin := int(math.Round(hz / s.BinWidthHz()))

	if bin < 0 {
		return 0
	}

	if bin >= len(s.Bins) {
		return len(s.Bins) - 1
	}

	return bin
}

// NyquistBin returns the final real FFT bin
func (s *Spectrum) NyquistBin() int {
	if s == nil || len(s.Bins) == 0 {
		return 0
	}

	return len(s.Bins) - 1
}

// NyquistHz returns the highest representable frequency
func (s *Spectrum) NyquistHz() float64 {
	if s == nil {
		return 0
	}

	return s.Format.NyquistHz()
}

// EnsurePower calculates |X[k]|² if necessary
// the result is cached inside Spectrum Power
func (s *Spectrum) EnsurePower() []float32 {
	if s == nil {
		return nil
	}

	if len(s.Power) != len(s.Bins) {
		s.Power = make([]float32, len(s.Bins))
	}

	for i, v := range s.Bins {
		re := real(v)
		im := imag(v)

		s.Power[i] = re*re + im*im
	}

	return s.Power
}

// RecomputePower forces recalculation of |X[k]|²
func (s *Spectrum) RecomputePower() []float32 {
	if s == nil {
		return nil
	}

	if cap(s.Power) < len(s.Bins) {
		s.Power = make([]float32, len(s.Bins))
	} else {
		s.Power = s.Power[:len(s.Bins)]
	}

	for i, v := range s.Bins {
		re := real(v)
		im := imag(v)

		s.Power[i] = re*re + im*im
	}

	return s.Power
}

// Magnitude returns |X[k]| for the specified FFT bin
func (s *Spectrum) Magnitude(bin int) float32 {
	if s == nil || bin < 0 || bin >= len(s.Bins) {
		return 0
	}

	v := s.Bins[bin]

	re := real(v)
	im := imag(v)

	return float32(math.Sqrt(float64(re*re + im*im)))
}

// PowerAt returns |X[k]|²
func (s *Spectrum) PowerAt(bin int) float32 {
	if s == nil || bin < 0 || bin >= len(s.Bins) {
		return 0
	}

	if len(s.Power) == len(s.Bins) {
		return s.Power[bin]
	}

	v := s.Bins[bin]

	re := real(v)
	im := imag(v)

	return re*re + im*im
}

// Clone creates a deep copy
func (s *Spectrum) Clone() *Spectrum {
	if s == nil {
		return nil
	}

	bins := make([]complex64, len(s.Bins))
	copy(bins, s.Bins)

	var power []float32

	if s.Power != nil {
		power = make([]float32, len(s.Power))
		copy(power, s.Power)
	}

	return &Spectrum{
		Bins:         bins,
		Power:        power,
		Format:       s.Format,
		FFTSize:      s.FFTSize,
		WindowSize:   s.WindowSize,
		HopSize:      s.HopSize,
		Sequence:     s.Sequence,
		SampleOffset: s.SampleOffset,
	}
}
