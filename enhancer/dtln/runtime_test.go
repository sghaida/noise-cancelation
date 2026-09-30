package dtln

import (
	"errors"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestRuntimeValidationWithoutNativeRuntime(t *testing.T) {
	if err := InitializeRuntime(""); err == nil {
		t.Fatal("InitializeRuntime() returned nil error for an empty path")
	}
	if err := InitializeRuntime("does-not-exist.dylib"); err == nil {
		t.Fatal("InitializeRuntime() returned nil error for a missing library")
	}

	if err := ShutdownRuntime(); err != nil {
		t.Fatalf("ShutdownRuntime() returned error before initialization: %v", err)
	}

	if err := acquireRuntime(); err == nil {
		t.Fatal("acquireRuntime() returned nil error before initialization")
	}
}

func TestRuntimeInitializeWhenAlreadyReady(t *testing.T) {
	originalReady := isRuntimeReady
	isRuntimeReady = func() bool { return true }
	defer func() { isRuntimeReady = originalReady }()

	if err := InitializeRuntime(""); err != nil {
		t.Fatalf("InitializeRuntime() returned error for ready runtime: %v", err)
	}
}

func TestRuntimeProcessorReservation(t *testing.T) {
	runtimeMu.Lock()
	activeProcessors = 1
	runtimeMu.Unlock()

	if err := ShutdownRuntime(); err == nil {
		t.Fatal("ShutdownRuntime() returned nil error with an active processor")
	}

	releaseRuntime()

	runtimeMu.Lock()
	defer runtimeMu.Unlock()

	if activeProcessors != 0 {
		t.Fatalf("active processors = %d, want 0", activeProcessors)
	}
}

func TestAcquireRuntimeWhenReady(t *testing.T) {
	original := isRuntimeReady
	isRuntimeReady = func() bool { return true }
	defer func() { isRuntimeReady = original }()

	if err := acquireRuntime(); err != nil {
		t.Fatalf("acquireRuntime() returned error: %v", err)
	}

	releaseRuntime()
}

func TestRuntimeInitializeAndShutdownWithFakes(t *testing.T) {
	originalInitialize := initializeEnvironment
	originalDestroy := destroyEnvironment
	originalReady := isRuntimeReady
	initializeEnvironment = func(...ort.EnvironmentOption) error { return nil }
	destroyEnvironment = func() error { return nil }
	isRuntimeReady = func() bool { return false }
	defer func() {
		initializeEnvironment = originalInitialize
		destroyEnvironment = originalDestroy
		isRuntimeReady = originalReady
	}()

	if err := InitializeRuntime("fake-runtime"); err != nil {
		t.Fatalf("InitializeRuntime() returned error: %v", err)
	}

	isRuntimeReady = func() bool { return true }
	if err := ShutdownRuntime(); err != nil {
		t.Fatalf("ShutdownRuntime() returned error: %v", err)
	}
}

func TestRuntimeShutdownError(t *testing.T) {
	originalDestroy := destroyEnvironment
	originalReady := isRuntimeReady
	destroyEnvironment = func() error { return errors.New("destroy failed") }
	isRuntimeReady = func() bool { return true }
	defer func() {
		destroyEnvironment = originalDestroy
		isRuntimeReady = originalReady
	}()

	if err := ShutdownRuntime(); err == nil {
		t.Fatal("ShutdownRuntime() returned nil error")
	}
}
