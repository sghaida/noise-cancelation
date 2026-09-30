package enhancer

// Enhancer processes streaming mono PCM audio
//
// Samples use normalized float32 PCM in the range [-1, 1]
type Enhancer interface {
	Process(samples []float32) ([]float32, error)
	Flush() ([]float32, error)
	Reset()
	Close() error
}
