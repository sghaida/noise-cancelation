package dtln

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/sghaida/noise-cancelation/enhancer"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	_ enhancer.Enhancer = (*Processor)(nil)

	errProcessorClosed = errors.New("DTLN processor is closed")
)

// Processor performs stateful streaming DTLN speech enhancement
//
// One Processor belongs to one audio stream because the neural recurrent state
// contains temporal information from previous frames
type Processor struct {
	model1 modelRunner
	model2 modelRunner

	inputBuffer  []float32
	outputBuffer []float32

	spectrum  []complex64
	magnitude []float32
	timeBlock []float32
	hopOutput []float32

	pending      [BlockShift]float32
	pendingCount int

	mu           sync.Mutex
	closed       bool
	runtimeOwned bool
}

// New creates a streaming DTLN processor
func New(cfg Config) (*Processor, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	if err := acquireRuntime(); err != nil {
		return nil, err
	}

	runtimeOwned := true

	model1, err := modelStageFactory(cfg.Model1Path, ort.NewShape(1, 1, SpectrumBins))
	if err != nil {
		releaseRuntime()

		return nil, fmt.Errorf("create DTLN model 1: %w", err)
	}

	model2, err := modelStageFactory(cfg.Model2Path, ort.NewShape(1, 1, BlockLength))
	if err != nil {
		_ = model1.close()
		releaseRuntime()

		return nil, fmt.Errorf("create DTLN model 2: %w", err)
	}

	return &Processor{
		model1:       model1,
		model2:       model2,
		inputBuffer:  make([]float32, BlockLength),
		outputBuffer: make([]float32, BlockLength),
		spectrum:     make([]complex64, BlockLength),
		magnitude:    make([]float32, SpectrumBins),
		timeBlock:    make([]float32, BlockLength),
		hopOutput:    make([]float32, BlockShift),
		runtimeOwned: runtimeOwned,
	}, nil
}

// Process accepts arbitrary sized 16 kHz PCM chunks and returns all complete DTLN hops currently available
func (p *Processor) Process(samples []float32) ([]float32, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil, errProcessorClosed
	}

	if len(samples) == 0 {
		return nil, nil
	}

	if err := validateSamples(samples); err != nil {
		return nil, err
	}

	hopCount := (p.pendingCount + len(samples)) / BlockShift
	output := make([]float32, 0, hopCount*BlockShift)

	if p.pendingCount > 0 {
		required := BlockShift - p.pendingCount

		if len(samples) < required {
			copy(p.pending[p.pendingCount:], samples)
			p.pendingCount += len(samples)

			return nil, nil
		}

		copy(p.pending[p.pendingCount:], samples[:required])

		hop, err := p.processHop(p.pending[:])
		if err != nil {
			return nil, err
		}

		output = append(output, hop...)
		samples = samples[required:]
		p.pendingCount = 0
		clear(p.pending[:])
	}

	for len(samples) >= BlockShift {
		hop, err := p.processHop(samples[:BlockShift])
		if err != nil {
			return nil, err
		}

		output = append(output, hop...)
		samples = samples[BlockShift:]
	}

	if len(samples) > 0 {
		copy(p.pending[:], samples)
		p.pendingCount = len(samples)
	}

	return output, nil
}

// processHop performs one complete DTLN inference step
//
// Stage one:
//
//	X[k] = FFT{x[n]}
//	M[k] = model1(|X[k]|)
//	Y1[k] = M[k] X[k]
//
// Stage two:
//
//	y1[n] = IFFT{Y1[k]}
//	y2[n] = model2(y1[n])
//
// Streaming output uses overlap add with hop H:
//
//	y[n] = Σm ym[n - mH]
func (p *Processor) processHop(samples []float32) ([]float32, error) {
	if len(samples) != BlockShift {
		return nil, fmt.Errorf("DTLN hop contains %d samples, want %d", len(samples), BlockShift)
	}

	if err := p.prepareInput(samples); err != nil {
		return nil, err
	}

	if err := p.runStageOne(); err != nil {
		return nil, err
	}

	if err := inverseSpectrum(p.spectrum, p.timeBlock); err != nil {
		return nil, err
	}

	enhanced, err := p.model2.run(p.timeBlock)
	if err != nil {
		return nil, fmt.Errorf("DTLN stage 2: %w", err)
	}

	if err := validateFiniteValues(enhanced, "DTLN stage 2 output"); err != nil {
		return nil, err
	}

	p.accumulateOutput(enhanced)

	return p.hopOutput, nil
}

func (p *Processor) prepareInput(samples []float32) error {
	copy(p.inputBuffer, p.inputBuffer[BlockShift:])
	copy(p.inputBuffer[BlockLength-BlockShift:], samples)

	return forwardSpectrum(p.inputBuffer, p.spectrum, p.magnitude)
}

func (p *Processor) runStageOne() error {
	mask, err := p.model1.run(p.magnitude)
	if err != nil {
		return fmt.Errorf("DTLN stage 1: %w", err)
	}

	return applyMask(p.spectrum, mask)
}

func (p *Processor) accumulateOutput(enhanced []float32) {
	copy(p.outputBuffer, p.outputBuffer[BlockShift:])
	clear(p.outputBuffer[BlockLength-BlockShift:])

	for i, sample := range enhanced {
		p.outputBuffer[i] += sample
	}

	for i, sample := range p.outputBuffer[:BlockShift] {
		if sample > 1 {
			sample = 1
		} else if sample < -1 {
			sample = -1
		}

		p.hopOutput[i] = sample
	}
}

// Flush zero pads the final incomplete hop and returns only samples belonging to the original input
func (p *Processor) Flush() ([]float32, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil, errProcessorClosed
	}

	if p.pendingCount == 0 {
		return nil, nil
	}

	validSamples := p.pendingCount

	var padded [BlockShift]float32
	copy(padded[:], p.pending[:p.pendingCount])

	p.pendingCount = 0
	clear(p.pending[:])

	hop, err := p.processHop(padded[:])
	if err != nil {
		return nil, err
	}

	output := make([]float32, validSamples)
	copy(output, hop[:validSamples])

	return output, nil
}

// Reset clears all audio buffers and recurrent model states
func (p *Processor) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()

	clear(p.inputBuffer)
	clear(p.outputBuffer)
	clear(p.spectrum)
	clear(p.magnitude)
	clear(p.timeBlock)
	clear(p.hopOutput)
	clear(p.pending[:])

	p.pendingCount = 0

	if p.model1 != nil {
		p.model1.reset()
	}

	if p.model2 != nil {
		p.model2.reset()
	}
}

// Close releases the ONNX sessions and tensors owned by the processor
func (p *Processor) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil
	}

	p.closed = true

	var model1Err error
	var model2Err error

	if p.model1 != nil {
		model1Err = p.model1.close()
		p.model1 = nil
	}

	if p.model2 != nil {
		model2Err = p.model2.close()
		p.model2 = nil
	}

	if p.runtimeOwned {
		releaseRuntime()
		p.runtimeOwned = false
	}

	return errors.Join(model1Err, model2Err)
}

func validateSamples(samples []float32) error {
	if err := validateFiniteValues(samples, "sample"); err != nil {
		return err
	}

	for index, sample := range samples {
		if sample < -1 || sample > 1 {
			return fmt.Errorf("sample %d = %v is outside normalized range [-1, 1]", index, sample)
		}
	}

	return nil
}

func validateFiniteValues(values []float32, name string) error {
	for index, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("%s value %d is not finite", name, index)
		}
	}

	return nil
}
