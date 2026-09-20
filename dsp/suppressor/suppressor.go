package suppressor

import "github.com/sghaida/noise-cancelation/audio"

// Suppressor removes estimated noise from frequency domain audio
type Suppressor interface {
	Process(spectrum audio.Spectrum, noisePSD []float32) (audio.Spectrum, error)

	Reset()
}
