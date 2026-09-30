package dtln

import "fmt"

const (
	// SampleRate is the fixed sample rate expected by the pretrained DTLN model
	SampleRate = 16000

	// BlockLength is the number of samples used by one DTLN analysis frame
	BlockLength = 512

	// BlockShift is the number of new samples consumed for every inference step
	BlockShift = 128

	// LatencySamples is the fixed model latency for the streaming DTLN framing
	LatencySamples = BlockLength

	// SpectrumBins is the number of bins produced by a real FFT
	//
	// bins = N / 2 + 1
	SpectrumBins = BlockLength/2 + 1

	// StateLayers is the number of recurrent layers in each DTLN separation stage
	StateLayers = 2

	// StateUnits is the number of LSTM units in each recurrent layer
	StateUnits = 128
)

// Config configures a DTLN processor
type Config struct {
	Model1Path string
	Model2Path string
}

// DefaultConfig returns paths for models stored in the repository model directory
func DefaultConfig() Config {
	return Config{
		Model1Path: "models/model_1.onnx",
		Model2Path: "models/model_2.onnx",
	}
}

// validate verifies that all required DTLN configuration values are present
func (c Config) validate() error {
	if c.Model1Path == "" {
		return fmt.Errorf("DTLN model 1 path is required")
	}

	if c.Model2Path == "" {
		return fmt.Errorf("DTLN model 2 path is required")
	}

	return nil
}
