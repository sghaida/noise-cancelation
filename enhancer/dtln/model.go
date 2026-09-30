package dtln

import (
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

type modelIO struct {
	dataInput   string
	stateInput  string
	dataOutput  string
	stateOutput string
}

type modelSession interface {
	Run() error
	Destroy() error
}

type modelTensor interface {
	GetData() []float32
	ZeroContents()
	Destroy() error
}

type modelStage struct {
	session modelSession

	dataInput   modelTensor
	stateInput  modelTensor
	dataOutput  modelTensor
	stateOutput modelTensor
}

type modelRunner interface {
	run(input []float32) ([]float32, error)
	reset()
	close() error
}

var modelStageFactory = func(path string, dataShape ort.Shape) (modelRunner, error) {
	return newModelStage(path, dataShape)
}

var modelInfoReader = ort.GetInputOutputInfo

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
func newModelStage(path string, dataShape ort.Shape) (*modelStage, error) {
	ioNames, err := resolveModelIO(path, dataShape)
	if err != nil {
		return nil, err
	}

	stage := &modelStage{}

	dataInput, err := ort.NewEmptyTensor[float32](dataShape)
	if err != nil {
		return nil, fmt.Errorf("create data input tensor: %w", err)
	}
	stage.dataInput = dataInput

	stateInput, err := ort.NewEmptyTensor[float32](stateShape())
	if err != nil {
		_ = stage.close()

		return nil, fmt.Errorf("create state input tensor: %w", err)
	}
	stage.stateInput = stateInput

	dataOutput, err := ort.NewEmptyTensor[float32](dataShape)
	if err != nil {
		_ = stage.close()

		return nil, fmt.Errorf("create data output tensor: %w", err)
	}
	stage.dataOutput = dataOutput

	stateOutput, err := ort.NewEmptyTensor[float32](stateShape())
	if err != nil {
		_ = stage.close()

		return nil, fmt.Errorf("create state output tensor: %w", err)
	}
	stage.stateOutput = stateOutput

	inputNames := []string{ioNames.dataInput, ioNames.stateInput}
	outputNames := []string{ioNames.dataOutput, ioNames.stateOutput}

	inputs := []ort.Value{dataInput, stateInput}
	outputs := []ort.Value{dataOutput, stateOutput}

	stage.session, err = ort.NewAdvancedSession(path, inputNames, outputNames, inputs, outputs, nil)
	if err != nil {
		_ = stage.close()

		return nil, fmt.Errorf("create ONNX session: %w", err)
	}

	return stage, nil
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
