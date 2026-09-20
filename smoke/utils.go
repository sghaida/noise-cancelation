//go:build smoke
// +build smoke

package smoke

import (
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const (
	fixturePath = "fixtures/human_voice_real_noise_30s_8khz_pcm16.wav"

	expectedSampleRate = 8000
	expectedChannels   = 1
	expectedBitDepth   = 16

	frameDuration = 20
)

// readTestFixture reads and validates the common smoke-test audio fixture
func readTestFixture(t *testing.T) (sampleRate int, samples []float32, durationSeconds float64) {
	t.Helper()

	sampleRate, channels, bitDepth, samples, err := readPCM16MonoWAV(fixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	if sampleRate != expectedSampleRate {
		t.Fatalf("sample rate = %d, want %d", sampleRate, expectedSampleRate)
	}

	if channels != expectedChannels {
		t.Fatalf("channels = %d, want %d", channels, expectedChannels)
	}

	if bitDepth != expectedBitDepth {
		t.Fatalf("bit depth = %d, want %d", bitDepth, expectedBitDepth)
	}

	durationSeconds = float64(len(samples)) / float64(sampleRate)

	if durationSeconds < 29.9 || durationSeconds > 30.1 {
		t.Fatalf(
			"duration = %.3f seconds, want approximately 30 seconds", durationSeconds,
		)
	}

	return sampleRate, samples, durationSeconds
}

// readPCM16MonoWAV reads an uncompressed PCM16 WAV file and converts
// each sample into the normalized float32 range
func readPCM16MonoWAV(path string) (
	sampleRate int, channels int, bitDepth int, samples []float32, err error,
) {
	fileData, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, 0, nil, fmt.Errorf("read file: %w", err)
	}

	if len(fileData) < 12 {
		return 0, 0, 0, nil, fmt.Errorf("WAV header is too short")
	}

	if string(fileData[0:4]) != "RIFF" {
		return 0, 0, 0, nil, fmt.Errorf("missing RIFF header")
	}

	if string(fileData[8:12]) != "WAVE" {
		return 0, 0, 0, nil, fmt.Errorf("missing WAVE header")
	}

	var (
		audioFormat uint16
		pcmData     []byte
		formatFound bool
		dataFound   bool
	)

	offset := 12

	for offset+8 <= len(fileData) {
		chunkID := string(fileData[offset : offset+4])

		chunkSize := int(binary.LittleEndian.Uint32(fileData[offset+4 : offset+8]))

		offset += 8

		if chunkSize < 0 || offset+chunkSize > len(fileData) {
			return 0, 0, 0, nil, fmt.Errorf("invalid %q chunk size %d", chunkID, chunkSize)
		}

		chunkData := fileData[offset : offset+chunkSize]

		switch chunkID {
		case "fmt ":
			if len(chunkData) < 16 {
				return 0, 0, 0, nil, fmt.Errorf("fmt chunk is too short")
			}

			audioFormat = binary.LittleEndian.Uint16(chunkData[0:2])
			channels = int(binary.LittleEndian.Uint16(chunkData[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(chunkData[4:8]))
			bitDepth = int(binary.LittleEndian.Uint16(chunkData[14:16]))
			formatFound = true

		case "data":
			pcmData = chunkData
			dataFound = true
		}

		offset += chunkSize

		if chunkSize%2 != 0 {
			offset++
		}
	}

	if !formatFound {
		return 0, 0, 0, nil, fmt.Errorf(
			"fmt chunk was not found",
		)
	}

	if !dataFound {
		return 0, 0, 0, nil, fmt.Errorf(
			"data chunk was not found",
		)
	}

	const (
		pcmAudioFormat = 1
		monoChannels   = 1
		pcm16BitDepth  = 16
	)

	if audioFormat != pcmAudioFormat {
		return 0, 0, 0, nil, fmt.Errorf("audio format = %d, want uncompressed PCM", audioFormat)
	}

	if channels != monoChannels {
		return 0, 0, 0, nil, fmt.Errorf("channels = %d, want mono audio", channels)
	}

	if bitDepth != pcm16BitDepth {
		return 0, 0, 0, nil, fmt.Errorf("bit depth = %d, want PCM16", bitDepth)
	}

	if len(pcmData)%2 != 0 {
		return 0, 0, 0, nil, fmt.Errorf("PCM data contains an incomplete 16-bit sample")
	}

	samples = make([]float32, len(pcmData)/2)

	for i := range samples {
		byteOffset := i * 2

		pcmSample := int16(
			binary.LittleEndian.Uint16(pcmData[byteOffset : byteOffset+2]),
		)

		samples[i] = float32(pcmSample) / 32768.0
	}

	return sampleRate, channels, bitDepth, samples, nil
}

// writePCM16MonoWAV writes normalized float32 samples into an
// uncompressed mono PCM16 WAV file
func writePCM16MonoWAV(path string, sampleRate int, samples []float32) error {
	const (
		headerSize    = 44
		channels      = uint16(1)
		bitsPerSample = uint16(16)
		audioFormat   = uint16(1)
		fmtChunkSize  = uint32(16)
	)

	outputDirectory := filepath.Dir(path)

	if err := os.MkdirAll(outputDirectory, 0o755); err != nil {
		return fmt.Errorf("create output directory %q: %w", outputDirectory, err)
	}

	const bytesPerSample = bitsPerSample / 8

	blockAlign := channels * bytesPerSample
	byteRate := uint32(sampleRate) * uint32(blockAlign)
	dataSize := uint32(len(samples)) * uint32(blockAlign)
	riffChunkSize := uint32(36) + dataSize

	fileData := make([]byte, headerSize+int(dataSize))

	copy(fileData[0:4], "RIFF")
	binary.LittleEndian.PutUint32(fileData[4:8], riffChunkSize)

	copy(fileData[8:12], "WAVE")
	copy(fileData[12:16], "fmt ")

	binary.LittleEndian.PutUint32(fileData[16:20], fmtChunkSize)
	binary.LittleEndian.PutUint16(fileData[20:22], audioFormat)
	binary.LittleEndian.PutUint16(fileData[22:24], channels)
	binary.LittleEndian.PutUint32(fileData[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(fileData[28:32], byteRate)
	binary.LittleEndian.PutUint16(fileData[32:34], blockAlign)
	binary.LittleEndian.PutUint16(fileData[34:36], bitsPerSample)

	copy(fileData[36:40], "data")
	binary.LittleEndian.PutUint32(fileData[40:44], dataSize)

	for i, sample := range samples {
		if sample > 1 {
			sample = 1
		} else if sample < -1 {
			sample = -1
		}

		pcmValue := math.Round(float64(sample) * 32768.0)

		if pcmValue > math.MaxInt16 {
			pcmValue = math.MaxInt16
		}

		if pcmValue < math.MinInt16 {
			pcmValue = math.MinInt16
		}

		offset := headerSize + i*2

		binary.LittleEndian.PutUint16(fileData[offset:offset+2], uint16(int16(pcmValue)))
	}

	if err := os.WriteFile(path, fileData, 0o644); err != nil {
		return fmt.Errorf("write output file %q: %w", path, err)
	}

	return nil
}

// assertFiniteSamples verifies that processing did not produce NaN
// or infinite sample values
func assertFiniteSamples(t *testing.T, samples []float32) {
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

// assertMCRASmokeFrame verifies one real MCRA output frame
func assertMCRASmokeFrame(
	t *testing.T, frame int, power []float32, noisePSD []float32, floor float32,
) {
	t.Helper()

	if len(noisePSD) != len(power) {
		t.Fatalf("frame %d: noise bins = %d, want %d", frame, len(noisePSD), len(power))
	}

	for k, value := range noisePSD {
		value64 := float64(value)

		if math.IsNaN(value64) {
			t.Fatalf("frame %d bin %d: noise is NaN", frame, k)
		}

		if math.IsInf(value64, 0) {
			t.Fatalf("frame %d bin %d: noise is infinite", frame, k)
		}

		if value < 0 {
			t.Fatalf("frame %d bin %d: noise = %v", frame, k, value)
		}

		if value < floor {
			t.Fatalf("frame %d bin %d: noise = %v, floor = %v", frame, k, value, floor)
		}
	}
}

// averagePower returns the mean power across all frequency bins
func averagePower(power []float32) float64 {
	if len(power) == 0 {
		return 0
	}

	var sum float64

	for _, value := range power {
		sum += float64(value)
	}

	return sum / float64(len(power))
}

// powerRatioDB returns observed power relative to estimated noise
//
// The equation is:
//
//	R = 10 * log10(P / N)
//
// A positive value means observed power is above the estimated noise
func powerRatioDB(power float64, noisePower float64) float64 {
	const epsilon = 1e-12
	ratio := (power + epsilon) / (noisePower + epsilon)
	return 10 * math.Log10(ratio)
}

// writeMCRACSV writes MCRA diagnostic values to a CSV file
func writeMCRACSV(path string, rows [][]string) error {
	dir := filepath.Dir(path)

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create CSV: %w", err)
	}

	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if err := writer.WriteAll(rows); err != nil {
		return fmt.Errorf("write CSV: %w", err)
	}

	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush CSV: %w", err)
	}

	return nil
}
