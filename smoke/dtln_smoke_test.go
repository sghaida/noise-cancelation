//go:build smoke
// +build smoke

package smoke

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/sghaida/noise-cancelation/enhancer/dtln"
)

const (
	dtlnFixturePath = "fixtures/human_voice_with_noise_with_5s_baseline_16khz.wav"
	dtlnOutputPath  = "output/dtln.out.wav"

	dtlnModel1Path = "../models/model_1.onnx"
	dtlnModel2Path = "../models/model_2.onnx"

	dtlnChunkSize = 320
)

// TestDTLNSmoke processes real 16 kHz PCM audio through both DTLN ONNX stages
func TestDTLNSmoke(t *testing.T) {
	runtimePath := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")

	if runtimePath == "" {
		t.Fatal("ONNXRUNTIME_SHARED_LIBRARY_PATH is required")
	}

	if err := dtln.InitializeRuntime(runtimePath); err != nil {
		t.Fatalf("initialize ONNX Runtime: %v", err)
	}

	defer func() {
		if err := dtln.ShutdownRuntime(); err != nil {
			t.Errorf("shutdown ONNX Runtime: %v", err)
		}
	}()

	sampleRate, channels, bitDepth, samples, err := readPCM16MonoWAV(dtlnFixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	if sampleRate != dtln.SampleRate {
		t.Fatalf("sample rate = %d, want %d", sampleRate, dtln.SampleRate)
	}

	if channels != expectedChannels {
		t.Fatalf("channels = %d, want %d", channels, expectedChannels)
	}

	if bitDepth != expectedBitDepth {
		t.Fatalf("bit depth = %d, want %d", bitDepth, expectedBitDepth)
	}

	processor, err := dtln.New(dtln.Config{
		Model1Path: dtlnModel1Path,
		Model2Path: dtlnModel2Path,
	})
	if err != nil {
		t.Fatalf("create DTLN processor: %v", err)
	}

	defer func() {
		if err := processor.Close(); err != nil {
			t.Errorf("close DTLN processor: %v", err)
		}
	}()

	output := make([]float32, 0, len(samples))

	for start := 0; start < len(samples); start += dtlnChunkSize {
		end := start + dtlnChunkSize

		if end > len(samples) {
			end = len(samples)
		}

		enhanced, err := processor.Process(samples[start:end])
		if err != nil {
			t.Fatalf("process samples at %d: %v", start, err)
		}

		assertDTLNFinite(t, enhanced)
		output = append(output, enhanced...)
	}

	remaining, err := processor.Flush()
	if err != nil {
		t.Fatalf("flush DTLN processor: %v", err)
	}

	assertDTLNFinite(t, remaining)
	output = append(output, remaining...)

	if len(output) != len(samples) {
		t.Fatalf("output contains %d samples, want %d", len(output), len(samples))
	}

	if err := os.MkdirAll(filepath.Dir(dtlnOutputPath), 0o755); err != nil {
		t.Fatalf("create output directory: %v", err)
	}

	if err := writePCM16MonoWAV(dtlnOutputPath, sampleRate, output); err != nil {
		t.Fatalf("write DTLN output: %v", err)
	}

	t.Logf("input samples: %d", len(samples))
	t.Logf("output samples: %d", len(output))
	t.Logf("DTLN output written to %s", dtlnOutputPath)
}

// assertDTLNFinite verifies that enhanced PCM contains no invalid values
func assertDTLNFinite(t *testing.T, samples []float32) {
	t.Helper()

	for i, sample := range samples {
		value := float64(sample)

		if math.IsNaN(value) {
			t.Fatalf("sample %d is NaN", i)
		}

		if math.IsInf(value, 0) {
			t.Fatalf("sample %d is infinite", i)
		}
	}
}
