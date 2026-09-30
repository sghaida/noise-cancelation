package dtln

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	runtimeMu             sync.Mutex
	activeProcessors      int
	isRuntimeReady        = ort.IsInitialized
	initializeEnvironment = ort.InitializeEnvironment
	destroyEnvironment    = ort.DestroyEnvironment
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

// acquireRuntime reserves the initialized process-wide runtime for one processor.
func acquireRuntime() error {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()

	if !isRuntimeReady() {
		return fmt.Errorf("ONNX Runtime is not initialized")
	}

	activeProcessors++

	return nil
}

// releaseRuntime releases a processor's runtime reservation.
func releaseRuntime() {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()

	if activeProcessors > 0 {
		activeProcessors--
	}
}
