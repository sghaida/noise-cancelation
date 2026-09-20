package snr

// Processor defines snr operations required by log mmse
// to use the production decision directed estimator while making error paths
// independently testable
type Processor interface {
	Process(power []float32, noisePSD []float32) ([]float32, []float32, error)
	Update(cleanPower, noisePSD []float32) error
	Reset()
}
