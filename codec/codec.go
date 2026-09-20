package codec

// Codec defines an audio codec capable of converting encoded audio to and from normalized PCM samples
// decoded PCM samples use float32 values in the range [-1.0, +1.0]
type Codec interface {
	// Name returns the codec name
	Name() string
	// SampleRate returns the rate in samples per second Hz
	SampleRate() int
	// Channels returns the number of audio channels used by the codec
	Channels() int

	// Decode decodes encoded audio to normalized PCM in the range [-1.0, +1.0]
	Decode(data []byte) []float32

	// Encode encodes normalized PCM in the range [-1.0, +1.0] back to the codec representation
	Encode(samples []float32) []byte
}
