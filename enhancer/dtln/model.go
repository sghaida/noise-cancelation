package dtln

import (
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// modelIO identifies the data and recurrent state tensors for one model stage
type modelIO struct {
	dataInput   string
	stateInput  string
	dataOutput  string
	stateOutput string
}

// modelSession runs and releases one ONNX model session
type modelSession interface {
	// Run executes inference with the tensors bound to the session
	Run() error
	// Destroy releases resources owned by the session
	Destroy() error
}

// modelTensor provides typed data access and lifecycle control for one tensor
type modelTensor interface {
	// GetData returns the mutable tensor values
	GetData() []float32
	// ZeroContents clears every tensor value
	ZeroContents()
	// Destroy releases resources owned by the tensor
	Destroy() error
	// native returns the ONNX Runtime value wrapped by the tensor
	native() ort.Value
}

// runtimeModelTensor adapts an ONNX Runtime tensor to modelTensor
type runtimeModelTensor struct {
	*ort.Tensor[float32]
}

// native returns the wrapped ONNX Runtime tensor value
func (t *runtimeModelTensor) native() ort.Value {
	return t.Tensor
}

// modelSessionOptions configures execution for one ONNX model session
type modelSessionOptions interface {
	// SetExecutionMode selects sequential or parallel graph execution
	SetExecutionMode(ort.ExecutionMode) error
	// SetGraphOptimizationLevel selects the graph optimization level
	SetGraphOptimizationLevel(ort.GraphOptimizationLevel) error
	// SetIntraOpNumThreads sets threads used within graph operations
	SetIntraOpNumThreads(int) error
	// SetInterOpNumThreads sets threads used across graph operations
	SetInterOpNumThreads(int) error
	// Destroy releases resources owned by the options
	Destroy() error
	// native returns the wrapped ONNX Runtime session options
	native() *ort.SessionOptions
}

// runtimeSessionOptions adapts ONNX Runtime options to modelSessionOptions
type runtimeSessionOptions struct {
	*ort.SessionOptions
}

// native returns the wrapped ONNX Runtime session options
func (o *runtimeSessionOptions) native() *ort.SessionOptions {
	return o.SessionOptions
}

// modelStage owns the tensors and session for one DTLN inference stage
type modelStage struct {
	session modelSession

	dataInput   modelTensor
	stateInput  modelTensor
	dataOutput  modelTensor
	stateOutput modelTensor
}

// modelRunner advances one stateful DTLN inference stage
type modelRunner interface {
	// run processes one input tensor and returns the model output
	run(input []float32) ([]float32, error)
	// reset clears recurrent and tensor state
	reset()
	// close releases resources owned by the model stage
	close() error
}

// modelStageFactory creates model stages and can be replaced by tests
var modelStageFactory = func(path string, dataShape ort.Shape, cfg Config) (modelRunner, error) {
	return newModelStage(path, dataShape, cfg)
}

// modelInfoReader reads model tensor metadata and can be replaced by tests
var modelInfoReader = ort.GetInputOutputInfo

// modelTensorFactory creates model tensors and can be replaced by tests
var modelTensorFactory = func(shape ort.Shape) (modelTensor, error) {
	tensor, err := ort.NewEmptyTensor[float32](shape)
	if err != nil {
		return nil, err
	}

	return &runtimeModelTensor{Tensor: tensor}, nil
}

// modelSessionFactory creates model sessions and can be replaced by tests
var modelSessionFactory = func(
	path string,
	inputNames []string,
	outputNames []string,
	inputs []modelTensor,
	outputs []modelTensor,
	options modelSessionOptions,
) (modelSession, error) {
	nativeInputs := make([]ort.Value, len(inputs))
	for index, input := range inputs {
		nativeInputs[index] = input.native()
	}
	nativeOutputs := make([]ort.Value, len(outputs))
	for index, output := range outputs {
		nativeOutputs[index] = output.native()
	}

	return ort.NewAdvancedSession(
		path,
		inputNames,
		outputNames,
		nativeInputs,
		nativeOutputs,
		options.native(),
	)
}

// sessionOptionsFactory creates session options and can be replaced by tests
var sessionOptionsFactory = func() (modelSessionOptions, error) {
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}

	return &runtimeSessionOptions{SessionOptions: options}, nil
}

// stateShape returns the recurrent DTLN state shape
//
// State layout is:
//
//	[batch, layer, unit, hidden_or_cell]
//
// The final dimension contains the LSTM hidden and cell states
func stateShape() ort.Shape {
	return ort.NewShape(1, StateLayers, StateUnits, 2)
}

// newModelStage creates one DTLN ONNX inference stage
func newModelStage(path string, dataShape ort.Shape, cfg Config) (*modelStage, error) {
	ioNames, err := resolveModelIO(path, dataShape)
	if err != nil {
		return nil, err
	}

	stage := &modelStage{}

	dataInput, err := modelTensorFactory(dataShape)
	if err != nil {
		return nil, fmt.Errorf("create data input tensor: %w", err)
	}
	stage.dataInput = dataInput

	stateInput, err := modelTensorFactory(stateShape())
	if err != nil {
		_ = stage.close()

		return nil, fmt.Errorf("create state input tensor: %w", err)
	}
	stage.stateInput = stateInput

	dataOutput, err := modelTensorFactory(dataShape)
	if err != nil {
		_ = stage.close()

		return nil, fmt.Errorf("create data output tensor: %w", err)
	}
	stage.dataOutput = dataOutput

	stateOutput, err := modelTensorFactory(stateShape())
	if err != nil {
		_ = stage.close()

		return nil, fmt.Errorf("create state output tensor: %w", err)
	}
	stage.stateOutput = stateOutput

	inputNames := []string{ioNames.dataInput, ioNames.stateInput}
	outputNames := []string{ioNames.dataOutput, ioNames.stateOutput}

	inputs := []modelTensor{dataInput, stateInput}
	outputs := []modelTensor{dataOutput, stateOutput}

	sessionOptions, err := newSessionOptions(cfg)
	if err != nil {
		_ = stage.close()

		return nil, err
	}
	defer func() { _ = sessionOptions.Destroy() }()

	stage.session, err = modelSessionFactory(
		path,
		inputNames,
		outputNames,
		inputs,
		outputs,
		sessionOptions,
	)
	if err != nil {
		_ = stage.close()

		return nil, fmt.Errorf("create ONNX session: %w", err)
	}

	return stage, nil
}

// newSessionOptions creates optimized sequential options using the configured thread counts
func newSessionOptions(cfg Config) (modelSessionOptions, error) {
	options, err := sessionOptionsFactory()
	if err != nil {
		return nil, fmt.Errorf("create ONNX session options: %w", err)
	}

	configure := func(err error, name string) (modelSessionOptions, error) {
		if err == nil {
			return options, nil
		}

		_ = options.Destroy()

		return nil, fmt.Errorf("configure ONNX %s: %w", name, err)
	}

	if _, err := configure(options.SetExecutionMode(ort.ExecutionModeSequential), "execution mode"); err != nil {
		return nil, err
	}
	if _, err := configure(options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll), "graph optimization"); err != nil {
		return nil, err
	}
	if _, err := configure(options.SetIntraOpNumThreads(cfg.IntraOpNumThreads), "intra-op threads"); err != nil {
		return nil, err
	}
	if _, err := configure(options.SetInterOpNumThreads(cfg.InterOpNumThreads), "inter-op threads"); err != nil {
		return nil, err
	}

	return options, nil
}

// resolveModelIO discovers DTLN tensor names using their expected shapes
//
// Model generated tensor names are intentionally not hard coded
func resolveModelIO(path string, dataShape ort.Shape) (modelIO, error) {
	inputs, outputs, err := modelInfoReader(path)
	if err != nil {
		return modelIO{}, fmt.Errorf("inspect ONNX model %s: %w", path, err)
	}

	dataInput, err := findTensorName(inputs, dataShape)
	if err != nil {
		return modelIO{}, fmt.Errorf("find data input: %w", err)
	}

	stateInput, err := findTensorName(inputs, stateShape())
	if err != nil {
		return modelIO{}, fmt.Errorf("find state input: %w", err)
	}

	dataOutput, err := findTensorName(outputs, dataShape)
	if err != nil {
		return modelIO{}, fmt.Errorf("find data output: %w", err)
	}

	stateOutput, err := findTensorName(outputs, stateShape())
	if err != nil {
		return modelIO{}, fmt.Errorf("find state output: %w", err)
	}

	return modelIO{
		dataInput:   dataInput,
		stateInput:  stateInput,
		dataOutput:  dataOutput,
		stateOutput: stateOutput,
	}, nil
}

// findTensorName finds exactly one float32 tensor matching the requested shape
func findTensorName(infos []ort.InputOutputInfo, shape ort.Shape) (string, error) {
	var name string

	for _, info := range infos {
		if info.DataType != ort.TensorElementDataTypeFloat {
			continue
		}

		if !info.Dimensions.Equals(shape) {
			continue
		}

		if name != "" {
			return "", fmt.Errorf("multiple tensors match shape %s", shape)
		}

		name = info.Name
	}

	if name == "" {
		return "", fmt.Errorf("tensor with shape %s was not found", shape)
	}

	return name, nil
}

// run processes one tensor and advances the recurrent LSTM state
func (m *modelStage) run(input []float32) ([]float32, error) {
	if m == nil || m.dataInput == nil {
		return nil, errors.New("model stage is not initialized")
	}

	data := m.dataInput.GetData()

	if len(input) != len(data) {
		return nil, fmt.Errorf("model input contains %d values, want %d", len(input), len(data))
	}

	if err := validateFiniteValues(input, "model input"); err != nil {
		return nil, err
	}

	if m.session == nil {
		return nil, errors.New("model stage session is not initialized")
	}

	copy(data, input)

	if err := m.session.Run(); err != nil {
		return nil, fmt.Errorf("run ONNX session: %w", err)
	}

	if err := validateFiniteValues(m.dataOutput.GetData(), "model output"); err != nil {
		return nil, err
	}

	if err := validateFiniteValues(m.stateOutput.GetData(), "model state"); err != nil {
		return nil, err
	}

	copy(m.stateInput.GetData(), m.stateOutput.GetData())

	return m.dataOutput.GetData(), nil
}

// reset clears all model inputs, outputs and recurrent states
func (m *modelStage) reset() {
	if m == nil {
		return
	}

	if m.dataInput != nil {
		m.dataInput.ZeroContents()
	}
	if m.stateInput != nil {
		m.stateInput.ZeroContents()
	}
	if m.dataOutput != nil {
		m.dataOutput.ZeroContents()
	}
	if m.stateOutput != nil {
		m.stateOutput.ZeroContents()
	}
}

// close releases the ONNX session and tensor resources
func (m *modelStage) close() error {
	if m == nil {
		return nil
	}

	var errs []error

	if m.session != nil {
		errs = append(errs, m.session.Destroy())
		m.session = nil
	}

	if m.dataInput != nil {
		errs = append(errs, m.dataInput.Destroy())
		m.dataInput = nil
	}

	if m.stateInput != nil {
		errs = append(errs, m.stateInput.Destroy())
		m.stateInput = nil
	}

	if m.dataOutput != nil {
		errs = append(errs, m.dataOutput.Destroy())
		m.dataOutput = nil
	}

	if m.stateOutput != nil {
		errs = append(errs, m.stateOutput.Destroy())
		m.stateOutput = nil
	}

	return errors.Join(errs...)
}
