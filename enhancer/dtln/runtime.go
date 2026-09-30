package dtln

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	// runtimeMu protects process wide runtime ownership state
	runtimeMu sync.Mutex
	// activeProcessors counts processors that reserve the runtime
	activeProcessors int
	// isRuntimeReady reports whether ONNX Runtime is initialized
	isRuntimeReady = ort.IsInitialized
	// initializeEnvironment starts ONNX Runtime and can be replaced by tests
	initializeEnvironment = ort.InitializeEnvironment
	// destroyEnvironment stops ONNX Runtime and can be replaced by tests
	destroyEnvironment = ort.DestroyEnvironment
)

// InitializeRuntime initializes the process wide ONNX Runtime environment
//
// The shared library path must point to the native ONNX Runtime library for the current platform
func InitializeRuntime(sharedLibraryPath string) error {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()

	if isRuntimeReady() {
		return nil
	}

	if sharedLibraryPath == "" {
		return fmt.Errorf("ONNX Runtime shared library path is required")
	}

	ort.SetSharedLibraryPath(sharedLibraryPath)

	if err := initializeEnvironment(ort.WithLogLevelWarning()); err != nil {
		return fmt.Errorf("initialize ONNX Runtime: %w", err)
	}

	return nil
}

// ShutdownRuntime destroys the process wide ONNX Runtime environment
//
// All DTLN processors must be closed before this function is called
func ShutdownRuntime() error {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()

	if activeProcessors > 0 {
		return fmt.Errorf(
			"cannot shutdown ONNX Runtime while %d DTLN processor(s) are active",
			activeProcessors,
		)
	}

	if !isRuntimeReady() {
		return nil
	}

	if err := destroyEnvironment(); err != nil {
		return fmt.Errorf("destroy ONNX Runtime: %w", err)
	}

	return nil
}

// acquireRuntime reserves the initialized process wide runtime for one processor
func acquireRuntime() error {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()

	if !isRuntimeReady() {
		return fmt.Errorf("ONNX Runtime is not initialized")
	}

	activeProcessors++

	return nil
}

// releaseRuntime releases a processor runtime reservation
func releaseRuntime() {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()

	if activeProcessors > 0 {
		activeProcessors--
	}
}
