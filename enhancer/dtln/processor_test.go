package dtln

import (
	"errors"
	"math"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

type fakeModelRunner struct {
	output     []float32
	runError   error
	closeError error
	runCount   int
	resetCount int
	closeCount int
}

func (m *fakeModelRunner) run(input []float32) ([]float32, error) {
	m.runCount++

	if m.runError != nil {
		return nil, m.runError
	}

	if m.output != nil {
		return m.output, nil
	}

	return input, nil
}

func (m *fakeModelRunner) reset() {
	m.resetCount++
}

func (m *fakeModelRunner) close() error {
	m.closeCount++

	return m.closeError
}

func newTestProcessor(model1, model2 modelRunner) *Processor {
	return &Processor{
		model1:       model1,
		model2:       model2,
		inputBuffer:  make([]float32, BlockLength),
		outputBuffer: make([]float32, BlockLength),
		spectrum:     make([]complex64, BlockLength),
		magnitude:    make([]float32, SpectrumBins),
		timeBlock:    make([]float32, BlockLength),
		hopOutput:    make([]float32, BlockShift),
	}
}

// TestFFTIdentity verifies that a unit spectral mask reconstructs the original frame
func TestFFTIdentity(t *testing.T) {
	input := make([]float32, BlockLength)

	for i := range input {
		timeSeconds := float64(i) / SampleRate

		input[i] = float32(
			0.5*math.Sin(2*math.Pi*440*timeSeconds) +
				0.2*math.Sin(2*math.Pi*1200*timeSeconds),
		)
	}

	spectrum := make([]complex64, BlockLength)
	magnitude := make([]float32, SpectrumBins)
	output := make([]float32, BlockLength)
	mask := make([]float32, SpectrumBins)

	for i := range mask {
		mask[i] = 1
	}

	if err := forwardSpectrum(input, spectrum, magnitude); err != nil {
		t.Fatalf("forward spectrum: %v", err)
	}

	if err := applyMask(spectrum, mask); err != nil {
		t.Fatalf("apply mask: %v", err)
	}

	if err := inverseSpectrum(spectrum, output); err != nil {
		t.Fatalf("inverse spectrum: %v", err)
	}

	const tolerance = float32(1e-5)

	for i := range input {
		difference := input[i] - output[i]

		if difference < 0 {
			difference = -difference
		}

		if difference > tolerance {
			t.Fatalf("sample %d = %v, want %v", i, output[i], input[i])
		}
	}
}

func TestFFTArgumentValidation(t *testing.T) {
	validSamples := make([]float32, BlockLength)
	validSpectrum := make([]complex64, BlockLength)
	validMagnitude := make([]float32, SpectrumBins)

	if err := forwardSpectrum(nil, validSpectrum, validMagnitude); err == nil {
		t.Fatal("forwardSpectrum() accepted an invalid sample length")
	}
	if err := forwardSpectrum(validSamples, nil, validMagnitude); err == nil {
		t.Fatal("forwardSpectrum() accepted an invalid spectrum length")
	}
	if err := forwardSpectrum(validSamples, validSpectrum, nil); err == nil {
		t.Fatal("forwardSpectrum() accepted an invalid magnitude length")
	}

	if err := applyMask(nil, validMagnitude); err == nil {
		t.Fatal("applyMask() accepted an invalid spectrum length")
	}
	if err := applyMask(validSpectrum, nil); err == nil {
		t.Fatal("applyMask() accepted an invalid mask length")
	}
	invalidMask := make([]float32, SpectrumBins)
	invalidMask[4] = float32(math.NaN())
	if err := applyMask(validSpectrum, invalidMask); err == nil {
		t.Fatal("applyMask() accepted a non-finite mask")
	}

	if err := inverseSpectrum(nil, make([]float32, BlockLength)); err == nil {
		t.Fatal("inverseSpectrum() accepted an invalid spectrum length")
	}
	if err := inverseSpectrum(validSpectrum, nil); err == nil {
		t.Fatal("inverseSpectrum() accepted an invalid output length")
	}
}

func TestProcessorChunkingAndFlush(t *testing.T) {
	mask := make([]float32, SpectrumBins)
	for i := range mask {
		mask[i] = 1
	}

	input := make([]float32, BlockShift*5+37)
	for i := range input {
		input[i] = float32(math.Sin(float64(i)*0.03)) * 0.5
	}

	wholeOutput := processTestInput(t, input, mask, false)
	chunkedOutput := processTestInput(t, input, mask, true)

	if len(wholeOutput) != len(input) {
		t.Fatalf("whole output length = %d, want %d", len(wholeOutput), len(input))
	}

	if len(chunkedOutput) != len(input) {
		t.Fatalf("chunked output length = %d, want %d", len(chunkedOutput), len(input))
	}

	for i := range wholeOutput {
		if math.Abs(float64(wholeOutput[i]-chunkedOutput[i])) > 1e-5 {
			t.Fatalf("output sample %d differs: whole=%v chunked=%v", i, wholeOutput[i], chunkedOutput[i])
		}
	}
}

func processTestInput(t *testing.T, input, mask []float32, chunked bool) []float32 {
	t.Helper()

	processor := newTestProcessor(
		&fakeModelRunner{output: mask},
		&fakeModelRunner{},
	)
	output := make([]float32, 0, len(input))

	if !chunked {
		processed, err := processor.Process(input)
		if err != nil {
			t.Fatalf("process whole input: %v", err)
		}

		output = append(output, processed...)
	} else {
		output = append(output, processTestChunks(t, processor, input)...)
	}

	tail, err := processor.Flush()
	if err != nil {
		t.Fatalf("flush input: %v", err)
	}

	return append(output, tail...)
}

func processTestChunks(t *testing.T, processor *Processor, input []float32) []float32 {
	t.Helper()

	output := make([]float32, 0, len(input))

	for start := 0; start < len(input); {
		end := min(start+1+(start%320), len(input))
		processed, err := processor.Process(input[start:end])
		if err != nil {
			t.Fatalf("process chunk %d:%d: %v", start, end, err)
		}

		output = append(output, processed...)
		start = end
	}

	return output
}

func TestProcessorRejectsInvalidSamples(t *testing.T) {
	tests := []struct {
		name   string
		sample float32
	}{
		{name: "nan", sample: float32(math.NaN())},
		{name: "positive infinity", sample: float32(math.Inf(1))},
		{name: "negative infinity", sample: float32(math.Inf(-1))},
		{name: "above range", sample: 1.01},
		{name: "below range", sample: -1.01},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			processor := newTestProcessor(&fakeModelRunner{}, &fakeModelRunner{})

			if _, err := processor.Process([]float32{test.sample}); err == nil {
				t.Fatal("Process() returned nil error")
			}

			if processor.pendingCount != 0 {
				t.Fatalf("pending count = %d, want 0", processor.pendingCount)
			}
		})
	}
}

func TestProcessorEmptyInput(t *testing.T) {
	processor := newTestProcessor(&fakeModelRunner{}, &fakeModelRunner{})

	output, err := processor.Process(nil)
	if err != nil {
		t.Fatalf("Process() returned error for empty input: %v", err)
	}
	if len(output) != 0 {
		t.Fatalf("empty input output length = %d, want 0", len(output))
	}
}

func TestProcessorFlushIsIdempotent(t *testing.T) {
	processor := newTestProcessor(&fakeModelRunner{}, &fakeModelRunner{})

	if _, err := processor.Process(make([]float32, BlockShift+1)); err != nil {
		t.Fatalf("process: %v", err)
	}

	first, err := processor.Flush()
	if err != nil {
		t.Fatalf("first flush: %v", err)
	}

	second, err := processor.Flush()
	if err != nil {
		t.Fatalf("second flush: %v", err)
	}

	if len(first) != 1 {
		t.Fatalf("first flush length = %d, want 1", len(first))
	}

	if len(second) != 0 {
		t.Fatalf("second flush length = %d, want 0", len(second))
	}
}

func TestProcessorResetClearsState(t *testing.T) {
	model1 := &fakeModelRunner{output: make([]float32, SpectrumBins)}
	model2 := &fakeModelRunner{}
	processor := newTestProcessor(model1, model2)

	if _, err := processor.Process(make([]float32, BlockShift)); err != nil {
		t.Fatalf("process: %v", err)
	}

	processor.Reset()

	if model1.resetCount != 1 || model2.resetCount != 1 {
		t.Fatalf("reset counts = %d and %d, want 1 and 1", model1.resetCount, model2.resetCount)
	}

	if processor.pendingCount != 0 {
		t.Fatalf("pending count = %d, want 0", processor.pendingCount)
	}

	for i, value := range processor.inputBuffer {
		if value != 0 {
			t.Fatalf("input buffer value %d = %v, want 0", i, value)
		}
	}

	for i, value := range processor.outputBuffer {
		if value != 0 {
			t.Fatalf("output buffer value %d = %v, want 0", i, value)
		}
	}
}

func TestProcessorCloseAndInferenceErrors(t *testing.T) {
	closeError := errors.New("close failed")
	model1 := &fakeModelRunner{runError: errors.New("inference failed"), closeError: closeError}
	model2 := &fakeModelRunner{closeError: closeError}
	processor := newTestProcessor(model1, model2)
	processor.runtimeOwned = true

	if _, err := processor.Process(make([]float32, BlockShift)); err == nil {
		t.Fatal("Process() returned nil inference error")
	}

	if err := processor.Close(); !errors.Is(err, closeError) {
		t.Fatalf("Close() error = %v, want %v", err, closeError)
	}

	if err := processor.Close(); err != nil {
		t.Fatalf("second Close() returned error: %v", err)
	}

	if _, err := processor.Process(make([]float32, BlockShift)); !errors.Is(err, errProcessorClosed) {
		t.Fatalf("Process() error = %v, want %v", err, errProcessorClosed)
	}

	if _, err := processor.Flush(); !errors.Is(err, errProcessorClosed) {
		t.Fatalf("Flush() error = %v, want %v", err, errProcessorClosed)
	}

	if model1.closeCount != 1 || model2.closeCount != 1 {
		t.Fatalf("close counts = %d and %d, want 1 and 1", model1.closeCount, model2.closeCount)
	}
}

func TestProcessorHopValidation(t *testing.T) {
	processor := newTestProcessor(&fakeModelRunner{}, &fakeModelRunner{})

	if _, err := processor.processHop(nil); err == nil {
		t.Fatal("processHop() returned nil error for an invalid hop")
	}
}

func TestProcessorStageTwoError(t *testing.T) {
	mask := make([]float32, SpectrumBins)
	for i := range mask {
		mask[i] = 1
	}
	processor := newTestProcessor(
		&fakeModelRunner{output: mask},
		&fakeModelRunner{runError: errors.New("stage two failed")},
	)

	if _, err := processor.Process(make([]float32, BlockShift)); err == nil {
		t.Fatal("Process() returned nil stage-two error")
	}
}

func TestProcessorFlushInferenceError(t *testing.T) {
	processor := newTestProcessor(
		&fakeModelRunner{runError: errors.New("flush inference failed")},
		&fakeModelRunner{},
	)

	if _, err := processor.Process([]float32{0}); err != nil {
		t.Fatalf("Process() returned error before flush: %v", err)
	}
	if _, err := processor.Flush(); err == nil {
		t.Fatal("Flush() returned nil inference error")
	}
}

func TestNewValidationAndRuntimeRequirement(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("New() returned nil error for an empty config")
	}

	if _, err := New(DefaultConfig()); err == nil {
		t.Fatal("New() returned nil error without an initialized runtime")
	}
}

func TestNewWithModelFactory(t *testing.T) {
	originalReady := isRuntimeReady
	originalFactory := modelStageFactory
	isRuntimeReady = func() bool { return true }
	modelStageFactory = func(string, ort.Shape) (modelRunner, error) {
		return &fakeModelRunner{}, nil
	}
	defer func() {
		isRuntimeReady = originalReady
		modelStageFactory = originalFactory
	}()

	processor, err := New(Config{Model1Path: "one", Model2Path: "two"})
	if err != nil {
		t.Fatalf("New() returned error: %v", err)
	}

	if err := processor.Close(); err != nil {
		t.Fatalf("Close() returned error: %v", err)
	}
}

func TestNewModelFactoryErrors(t *testing.T) {
	originalReady := isRuntimeReady
	originalFactory := modelStageFactory
	isRuntimeReady = func() bool { return true }
	defer func() {
		isRuntimeReady = originalReady
		modelStageFactory = originalFactory
	}()

	modelStageFactory = func(string, ort.Shape) (modelRunner, error) {
		return nil, errors.New("model creation failed")
	}
	if _, err := New(Config{Model1Path: "one", Model2Path: "two"}); err == nil {
		t.Fatal("New() returned nil first-stage error")
	}

	first := &fakeModelRunner{}
	callCount := 0
	modelStageFactory = func(string, ort.Shape) (modelRunner, error) {
		callCount++
		if callCount == 1 {
			return first, nil
		}

		return nil, errors.New("second model creation failed")
	}
	if _, err := New(Config{Model1Path: "one", Model2Path: "two"}); err == nil {
		t.Fatal("New() returned nil second-stage error")
	}
	if first.closeCount != 1 {
		t.Fatalf("first model close count = %d, want 1", first.closeCount)
	}
}

func TestProcessorRejectsNonFiniteStageOutput(t *testing.T) {
	mask := make([]float32, SpectrumBins)
	for i := range mask {
		mask[i] = 1
	}

	output := make([]float32, BlockLength)
	output[3] = float32(math.NaN())
	processor := newTestProcessor(
		&fakeModelRunner{output: mask},
		&fakeModelRunner{output: output},
	)

	if _, err := processor.Process(make([]float32, BlockShift)); err == nil {
		t.Fatal("Process() returned nil stage output error")
	}
}

func BenchmarkProcessorProcessHop(b *testing.B) {
	mask := make([]float32, SpectrumBins)
	for i := range mask {
		mask[i] = 1
	}

	processor := newTestProcessor(
		&fakeModelRunner{output: mask},
		&fakeModelRunner{output: make([]float32, BlockLength)},
	)
	samples := make([]float32, BlockShift)

	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		if _, err := processor.processHop(samples); err != nil {
			b.Fatal(err)
		}
	}
}
