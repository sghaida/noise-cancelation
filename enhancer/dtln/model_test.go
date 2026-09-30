package dtln

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

type fakeModelTensor struct {
	data         []float32
	destroyErr   error
	zeroCount    int
	destroyCount int
}

func (t *fakeModelTensor) GetData() []float32 {
	return t.data
}

func (t *fakeModelTensor) ZeroContents() {
	clear(t.data)
	t.zeroCount++
}

func (t *fakeModelTensor) Destroy() error {
	t.destroyCount++

	return t.destroyErr
}

func (t *fakeModelTensor) native() ort.Value {
	return nil
}

type fakeModelSession struct {
	runError     error
	destroyError error
	runCount     int
}

type fakeSessionOptions struct {
	executionMode     ort.ExecutionMode
	optimizationLevel ort.GraphOptimizationLevel
	intraOpThreads    int
	interOpThreads    int
	failAt            string
	destroyCount      int
}

func (o *fakeSessionOptions) SetExecutionMode(mode ort.ExecutionMode) error {
	o.executionMode = mode
	return o.settingError("execution")
}

func (o *fakeSessionOptions) SetGraphOptimizationLevel(level ort.GraphOptimizationLevel) error {
	o.optimizationLevel = level
	return o.settingError("optimization")
}

func (o *fakeSessionOptions) SetIntraOpNumThreads(threads int) error {
	o.intraOpThreads = threads
	return o.settingError("intra-op")
}

func (o *fakeSessionOptions) SetInterOpNumThreads(threads int) error {
	o.interOpThreads = threads
	return o.settingError("inter-op")
}

func (o *fakeSessionOptions) Destroy() error {
	o.destroyCount++
	return nil
}

func (o *fakeSessionOptions) native() *ort.SessionOptions {
	return nil
}

func (o *fakeSessionOptions) settingError(name string) error {
	if o.failAt == name {
		return errors.New("setting failed")
	}

	return nil
}

func (s *fakeModelSession) Run() error {
	s.runCount++

	return s.runError
}

func (s *fakeModelSession) Destroy() error {
	return s.destroyError
}

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.Model1Path != "models/model_1.onnx" {
		t.Fatalf("model 1 path = %q, want models/model_1.onnx", config.Model1Path)
	}

	if config.Model2Path != "models/model_2.onnx" {
		t.Fatalf("model 2 path = %q, want models/model_2.onnx", config.Model2Path)
	}
	if config.IntraOpNumThreads != 1 || config.InterOpNumThreads != 1 {
		t.Fatalf(
			"default thread counts = %d and %d, want 1 and 1",
			config.IntraOpNumThreads,
			config.InterOpNumThreads,
		)
	}

	if err := config.validate(); err != nil {
		t.Fatalf("default config validation: %v", err)
	}
}

func TestConfigValidate(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		want   string
	}{
		{name: "missing model 1", config: Config{Model2Path: "model-2"}, want: "model 1"},
		{name: "missing model 2", config: Config{Model1Path: "model-1"}, want: "model 2"},
		{name: "negative intra-op threads", config: Config{Model1Path: "model-1", Model2Path: "model-2", IntraOpNumThreads: -1}, want: "intra-op"},
		{name: "negative inter-op threads", config: Config{Model1Path: "model-1", Model2Path: "model-2", InterOpNumThreads: -1}, want: "inter-op"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.config.validate()
			if err == nil {
				t.Fatal("validate() returned nil error")
			}

			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validate() error = %q, want %q", err, test.want)
			}
		})
	}
}

func TestFindTensorName(t *testing.T) {
	shape := ort.NewShape(1, 2)
	infos := []ort.InputOutputInfo{
		{Name: "wrong-type", Dimensions: shape, DataType: ort.TensorElementDataTypeInt32},
		{Name: "wrong-shape", Dimensions: ort.NewShape(1, 3), DataType: ort.TensorElementDataTypeFloat},
		{Name: "match", Dimensions: shape, DataType: ort.TensorElementDataTypeFloat},
	}

	name, err := findTensorName(infos, shape)
	if err != nil {
		t.Fatalf("findTensorName() error: %v", err)
	}

	if name != "match" {
		t.Fatalf("findTensorName() = %q, want match", name)
	}

	if _, err := findTensorName(nil, shape); err == nil {
		t.Fatal("findTensorName() returned nil error for missing tensor")
	}

	duplicate := make([]ort.InputOutputInfo, len(infos), len(infos)+1)
	copy(duplicate, infos)
	duplicate = append(duplicate, ort.InputOutputInfo{
		Name:       "duplicate",
		Dimensions: shape,
		DataType:   ort.TensorElementDataTypeFloat,
	})
	if _, err := findTensorName(duplicate, shape); err == nil {
		t.Fatal("findTensorName() returned nil error for duplicate tensors")
	}
}

func TestStateShape(t *testing.T) {
	want := ort.NewShape(1, StateLayers, StateUnits, 2)

	if !stateShape().Equals(want) {
		t.Fatalf("stateShape() = %s, want %s", stateShape(), want)
	}
}

func TestNewSessionOptions(t *testing.T) {
	originalFactory := sessionOptionsFactory
	defer func() { sessionOptionsFactory = originalFactory }()

	options := &fakeSessionOptions{}
	sessionOptionsFactory = func() (modelSessionOptions, error) {
		return options, nil
	}

	config := Config{IntraOpNumThreads: 2, InterOpNumThreads: 3}
	created, err := newSessionOptions(config)
	if err != nil {
		t.Fatalf("newSessionOptions() error: %v", err)
	}
	if created != options {
		t.Fatal("newSessionOptions() returned different options")
	}
	if options.executionMode != ort.ExecutionModeSequential {
		t.Fatalf("execution mode = %v, want sequential", options.executionMode)
	}
	if options.optimizationLevel != ort.GraphOptimizationLevelEnableAll {
		t.Fatalf("optimization = %v, want enable all", options.optimizationLevel)
	}
	if options.intraOpThreads != 2 || options.interOpThreads != 3 {
		t.Fatalf("thread counts = %d and %d, want 2 and 3", options.intraOpThreads, options.interOpThreads)
	}
}

func TestNewSessionOptionsErrors(t *testing.T) {
	originalFactory := sessionOptionsFactory
	defer func() { sessionOptionsFactory = originalFactory }()

	sessionOptionsFactory = func() (modelSessionOptions, error) {
		return nil, errors.New("create failed")
	}
	if _, err := newSessionOptions(Config{}); err == nil {
		t.Fatal("newSessionOptions() returned nil creation error")
	}

	for _, setting := range []string{"execution", "optimization", "intra-op", "inter-op"} {
		t.Run(setting, func(t *testing.T) {
			options := &fakeSessionOptions{failAt: setting}
			sessionOptionsFactory = func() (modelSessionOptions, error) {
				return options, nil
			}

			if _, err := newSessionOptions(Config{}); err == nil {
				t.Fatal("newSessionOptions() returned nil setting error")
			}
			if options.destroyCount != 1 {
				t.Fatalf("destroy count = %d, want 1", options.destroyCount)
			}
		})
	}
}

func TestValidateFiniteValues(t *testing.T) {
	if err := validateFiniteValues([]float32{0, -1, 1}, "test"); err != nil {
		t.Fatalf("validateFiniteValues() unexpected error: %v", err)
	}

	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		if err := validateFiniteValues([]float32{value}, "test"); err == nil {
			t.Fatal("validateFiniteValues() returned nil error for invalid value")
		}
	}
}

func TestModelStageValidationAndCleanup(t *testing.T) {
	var nilStage *modelStage
	nilStage.reset()
	if err := nilStage.close(); err != nil {
		t.Fatalf("nil model stage close: %v", err)
	}
	if _, err := nilStage.run(nil); err == nil {
		t.Fatal("nil model stage run returned nil error")
	}

	stage := &modelStage{}
	stage.reset()
	if _, err := stage.run(nil); err == nil {
		t.Fatal("uninitialized model stage run returned nil error")
	}

	stage.dataInput = &fakeModelTensor{data: make([]float32, 1)}
	if _, err := stage.run([]float32{0}); err == nil {
		t.Fatal("model stage without session returned nil error")
	}

	if err := stage.close(); err != nil {
		t.Fatalf("empty model stage close: %v", err)
	}
}

func TestModelInspectionErrors(t *testing.T) {
	const missingPath = "does-not-exist.onnx"

	if _, err := resolveModelIO(missingPath, ort.NewShape(1)); err == nil {
		t.Fatal("resolveModelIO() returned nil error for a missing model")
	}

	if _, err := newModelStage(missingPath, ort.NewShape(1), Config{}); err == nil {
		t.Fatal("newModelStage() returned nil error for a missing model")
	}
}

func TestNewModelStage(t *testing.T) {
	originalReader := modelInfoReader
	originalTensorFactory := modelTensorFactory
	originalSessionFactory := modelSessionFactory
	originalOptionsFactory := sessionOptionsFactory
	defer func() {
		modelInfoReader = originalReader
		modelTensorFactory = originalTensorFactory
		modelSessionFactory = originalSessionFactory
		sessionOptionsFactory = originalOptionsFactory
	}()

	dataShape := ort.NewShape(1, 1, SpectrumBins)
	state := stateShape()
	modelInfoReader = func(string) ([]ort.InputOutputInfo, []ort.InputOutputInfo, error) {
		return []ort.InputOutputInfo{
				{Name: "data-input", Dimensions: dataShape, DataType: ort.TensorElementDataTypeFloat},
				{Name: "state-input", Dimensions: state, DataType: ort.TensorElementDataTypeFloat},
			}, []ort.InputOutputInfo{
				{Name: "data-output", Dimensions: dataShape, DataType: ort.TensorElementDataTypeFloat},
				{Name: "state-output", Dimensions: state, DataType: ort.TensorElementDataTypeFloat},
			}, nil
	}

	createdTensors := make([]*fakeModelTensor, 0, 4)
	modelTensorFactory = func(ort.Shape) (modelTensor, error) {
		tensor := &fakeModelTensor{data: make([]float32, 1)}
		createdTensors = append(createdTensors, tensor)

		return tensor, nil
	}
	options := &fakeSessionOptions{}
	sessionOptionsFactory = func() (modelSessionOptions, error) {
		return options, nil
	}
	session := &fakeModelSession{}
	modelSessionFactory = func(
		string,
		[]string,
		[]string,
		[]modelTensor,
		[]modelTensor,
		modelSessionOptions,
	) (modelSession, error) {
		return session, nil
	}

	stage, err := newModelStage("model.onnx", dataShape, DefaultConfig())
	if err != nil {
		t.Fatalf("newModelStage() error: %v", err)
	}
	if stage.session != session {
		t.Fatal("newModelStage() did not retain session")
	}
	if len(createdTensors) != 4 {
		t.Fatalf("created tensors = %d, want 4", len(createdTensors))
	}
	if options.destroyCount != 1 {
		t.Fatalf("session options destroy count = %d, want 1", options.destroyCount)
	}
}

func TestNewModelStageCreationErrors(t *testing.T) {
	originalReader := modelInfoReader
	originalTensorFactory := modelTensorFactory
	originalSessionFactory := modelSessionFactory
	originalOptionsFactory := sessionOptionsFactory
	defer func() {
		modelInfoReader = originalReader
		modelTensorFactory = originalTensorFactory
		modelSessionFactory = originalSessionFactory
		sessionOptionsFactory = originalOptionsFactory
	}()

	dataShape := ort.NewShape(1, 1, SpectrumBins)
	state := stateShape()
	modelInfoReader = func(string) ([]ort.InputOutputInfo, []ort.InputOutputInfo, error) {
		return []ort.InputOutputInfo{
				{Name: "data-input", Dimensions: dataShape, DataType: ort.TensorElementDataTypeFloat},
				{Name: "state-input", Dimensions: state, DataType: ort.TensorElementDataTypeFloat},
			}, []ort.InputOutputInfo{
				{Name: "data-output", Dimensions: dataShape, DataType: ort.TensorElementDataTypeFloat},
				{Name: "state-output", Dimensions: state, DataType: ort.TensorElementDataTypeFloat},
			}, nil
	}
	sessionOptionsFactory = func() (modelSessionOptions, error) {
		return &fakeSessionOptions{}, nil
	}
	modelSessionFactory = func(
		string,
		[]string,
		[]string,
		[]modelTensor,
		[]modelTensor,
		modelSessionOptions,
	) (modelSession, error) {
		return nil, errors.New("session failed")
	}

	for failureIndex := range 5 {
		t.Run(fmt.Sprintf("failure-%d", failureIndex), func(t *testing.T) {
			creationIndex := 0
			modelTensorFactory = func(ort.Shape) (modelTensor, error) {
				if creationIndex == failureIndex {
					return nil, errors.New("tensor failed")
				}
				creationIndex++

				return &fakeModelTensor{data: make([]float32, 1)}, nil
			}

			if _, err := newModelStage("model.onnx", dataShape, DefaultConfig()); err == nil {
				t.Fatal("newModelStage() returned nil creation error")
			}
		})
	}
}

func TestResolveModelIO(t *testing.T) {
	originalReader := modelInfoReader
	modelInfoReader = func(string) ([]ort.InputOutputInfo, []ort.InputOutputInfo, error) {
		dataShape := ort.NewShape(1, 1, SpectrumBins)
		state := stateShape()

		return []ort.InputOutputInfo{
				{Name: "data-input", Dimensions: dataShape, DataType: ort.TensorElementDataTypeFloat},
				{Name: "state-input", Dimensions: state, DataType: ort.TensorElementDataTypeFloat},
			}, []ort.InputOutputInfo{
				{Name: "data-output", Dimensions: dataShape, DataType: ort.TensorElementDataTypeFloat},
				{Name: "state-output", Dimensions: state, DataType: ort.TensorElementDataTypeFloat},
			}, nil
	}
	defer func() { modelInfoReader = originalReader }()

	resolved, err := resolveModelIO("model.onnx", ort.NewShape(1, 1, SpectrumBins))
	if err != nil {
		t.Fatalf("resolveModelIO() returned error: %v", err)
	}

	if resolved.dataInput != "data-input" || resolved.stateInput != "state-input" {
		t.Fatalf("resolved inputs = %#v", resolved)
	}
	if resolved.dataOutput != "data-output" || resolved.stateOutput != "state-output" {
		t.Fatalf("resolved outputs = %#v", resolved)
	}
}

func TestResolveModelIOErrors(t *testing.T) {
	originalReader := modelInfoReader
	defer func() { modelInfoReader = originalReader }()

	dataShape := ort.NewShape(1, 1, SpectrumBins)
	state := stateShape()
	dataInput := ort.InputOutputInfo{Name: "data-input", Dimensions: dataShape, DataType: ort.TensorElementDataTypeFloat}
	stateInput := ort.InputOutputInfo{Name: "state-input", Dimensions: state, DataType: ort.TensorElementDataTypeFloat}
	dataOutput := ort.InputOutputInfo{Name: "data-output", Dimensions: dataShape, DataType: ort.TensorElementDataTypeFloat}
	stateOutput := ort.InputOutputInfo{Name: "state-output", Dimensions: state, DataType: ort.TensorElementDataTypeFloat}

	tests := []struct {
		name    string
		inputs  []ort.InputOutputInfo
		outputs []ort.InputOutputInfo
	}{
		{name: "missing data input", inputs: []ort.InputOutputInfo{stateInput}, outputs: []ort.InputOutputInfo{dataOutput, stateOutput}},
		{name: "missing state input", inputs: []ort.InputOutputInfo{dataInput}, outputs: []ort.InputOutputInfo{dataOutput, stateOutput}},
		{name: "missing data output", inputs: []ort.InputOutputInfo{dataInput, stateInput}, outputs: []ort.InputOutputInfo{stateOutput}},
		{name: "missing state output", inputs: []ort.InputOutputInfo{dataInput, stateInput}, outputs: []ort.InputOutputInfo{dataOutput}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			modelInfoReader = func(string) ([]ort.InputOutputInfo, []ort.InputOutputInfo, error) {
				return test.inputs, test.outputs, nil
			}

			if _, err := resolveModelIO("model.onnx", dataShape); err == nil {
				t.Fatal("resolveModelIO() returned nil error")
			}
		})
	}

	modelInfoReader = func(string) ([]ort.InputOutputInfo, []ort.InputOutputInfo, error) {
		return nil, nil, errors.New("inspect failed")
	}
	if _, err := resolveModelIO("model.onnx", dataShape); err == nil {
		t.Fatal("resolveModelIO() returned nil inspection error")
	}
}

func newFakeModelStage() (*modelStage, *fakeModelTensor, *fakeModelTensor, *fakeModelTensor, *fakeModelTensor, *fakeModelSession) {
	dataInput := &fakeModelTensor{data: make([]float32, 2)}
	stateInput := &fakeModelTensor{data: make([]float32, 2)}
	dataOutput := &fakeModelTensor{data: []float32{3, 4}}
	stateOutput := &fakeModelTensor{data: []float32{5, 6}}
	session := &fakeModelSession{}
	stage := &modelStage{
		session:     session,
		dataInput:   dataInput,
		stateInput:  stateInput,
		dataOutput:  dataOutput,
		stateOutput: stateOutput,
	}

	return stage, dataInput, stateInput, dataOutput, stateOutput, session
}

func TestModelStageRun(t *testing.T) {
	stage, _, stateInput, _, _, session := newFakeModelStage()

	output, err := stage.run([]float32{1, 2})
	if err != nil {
		t.Fatalf("run() returned error: %v", err)
	}

	if session.runCount != 1 {
		t.Fatalf("run count = %d, want 1", session.runCount)
	}
	if output[0] != 3 || output[1] != 4 {
		t.Fatalf("output = %v, want [3 4]", output)
	}
	if stateInput.data[0] != 5 || stateInput.data[1] != 6 {
		t.Fatalf("state input = %v, want [5 6]", stateInput.data)
	}
}

func TestModelStageResetAndClose(t *testing.T) {
	stage, dataInput, stateInput, dataOutput, stateOutput, _ := newFakeModelStage()

	stage.reset()
	if dataInput.zeroCount != 1 || stateInput.zeroCount != 1 || dataOutput.zeroCount != 1 || stateOutput.zeroCount != 1 {
		t.Fatalf("reset counts = %d, %d, %d, %d; want 1 each", dataInput.zeroCount, stateInput.zeroCount, dataOutput.zeroCount, stateOutput.zeroCount)
	}

	if err := stage.close(); err != nil {
		t.Fatalf("close() returned error: %v", err)
	}
	if dataInput.destroyCount != 1 || stateInput.destroyCount != 1 || dataOutput.destroyCount != 1 || stateOutput.destroyCount != 1 {
		t.Fatalf("destroy counts = %d, %d, %d, %d; want 1 each", dataInput.destroyCount, stateInput.destroyCount, dataOutput.destroyCount, stateOutput.destroyCount)
	}
}

func TestModelStageRunErrors(t *testing.T) {
	newStage := func() *modelStage {
		return &modelStage{
			session:     &fakeModelSession{},
			dataInput:   &fakeModelTensor{data: make([]float32, 1)},
			stateInput:  &fakeModelTensor{data: make([]float32, 1)},
			dataOutput:  &fakeModelTensor{data: make([]float32, 1)},
			stateOutput: &fakeModelTensor{data: make([]float32, 1)},
		}
	}

	if _, err := newStage().run(nil); err == nil {
		t.Fatal("run() returned nil length error")
	}

	lengthErrorStage := newFakeModelStageForErrors()
	if _, err := lengthErrorStage.run([]float32{0}); err == nil {
		t.Fatal("run() returned nil input length error")
	}

	inputErrorStage := newStage()
	if _, err := inputErrorStage.run([]float32{float32(math.NaN())}); err == nil {
		t.Fatal("run() returned nil input validation error")
	}

	sessionErrorStage := newStage()
	sessionErrorStage.session.(*fakeModelSession).runError = errors.New("run failed")
	if _, err := sessionErrorStage.run([]float32{0}); err == nil {
		t.Fatal("run() returned nil session error")
	}

	outputErrorStage := newStage()
	outputErrorStage.dataOutput.(*fakeModelTensor).data[0] = float32(math.Inf(1))
	if _, err := outputErrorStage.run([]float32{0}); err == nil {
		t.Fatal("run() returned nil output validation error")
	}

	stateErrorStage := newStage()
	stateErrorStage.stateOutput.(*fakeModelTensor).data[0] = float32(math.NaN())
	if _, err := stateErrorStage.run([]float32{0}); err == nil {
		t.Fatal("run() returned nil state validation error")
	}
}

func newFakeModelStageForErrors() *modelStage {
	stage, _, _, _, _, _ := newFakeModelStage()

	return stage
}

func TestModelStageCloseErrors(t *testing.T) {
	stage, dataInput, stateInput, dataOutput, stateOutput, session := newFakeModelStage()
	closeError := errors.New("destroy failed")
	session.destroyError = closeError
	dataInput.destroyErr = closeError
	stateInput.destroyErr = closeError
	dataOutput.destroyErr = closeError
	stateOutput.destroyErr = closeError

	if err := stage.close(); err == nil {
		t.Fatal("close() returned nil error")
	}
}
