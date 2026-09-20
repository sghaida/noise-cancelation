package dsp

// Processor defines a dsp processing stage
//
// implementations may modify the provided samples in place and return
// the same slice to avoid unnecessary allocations
//
// reset must clear any internal state so the processor can be reused
// for a new audio stream
type Processor interface {
	Process(samples []float32) ([]float32, error)
	Reset()
}

// Pipeline executes dsp processors sequentially
//
// the output of each processor is passed as the input to the next one
type Pipeline struct {
	processors []Processor
}

// NewPipeline creates a dsp pipeline using the provided processors
//
// Processors are executed in the same order they are provided
func NewPipeline(processors ...Processor) *Pipeline {
	return &Pipeline{
		processors: processors,
	}
}

// Process runs the samples through all configured dsp processors
func (p *Pipeline) Process(samples []float32) ([]float32, error) {
	var err error

	for _, processor := range p.processors {
		// A processor may return a different slice, for example when
		// an implementation needs a separate output buffer
		samples, err = processor.Process(samples)
		if err != nil {
			return nil, err
		}
	}

	return samples, nil
}

// Reset resets the internal state of every processor in the pipeline
func (p *Pipeline) Reset() {
	for _, processor := range p.processors {
		processor.Reset()
	}
}
