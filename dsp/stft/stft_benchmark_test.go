package stft

import (
	"math"
	"testing"
)

const (
	stftBenchmarkSampleRate = 8000
	stftBenchmarkFFTSize    = 256
	stftBenchmarkHopSize    = 128
	stftBenchmarkChunkSize  = 160
)

// BenchmarkSTFTProcess measures steady-state STFT processing using
// 20 ms audio chunks matching an 8 kHz Twilio audio stream
func BenchmarkSTFTProcess(b *testing.B) {
	analyzer, err := New(stftBenchmarkFFTSize, stftBenchmarkHopSize)
	if err != nil {
		b.Fatalf("create STFT analyzer: %v", err)
	}

	chunks := makeBenchmarkSTFTChunks(64, stftBenchmarkChunkSize, stftBenchmarkSampleRate)

	// Warm up the analyzer so internal buffering is initialized before
	// benchmark measurements begin
	for _, chunk := range chunks {
		if _, err := analyzer.Process(chunk); err != nil {
			b.Fatalf("warm up STFT analyzer: %v", err)
		}
	}

	b.ReportAllocs()

	// Each input sample is a float32
	b.SetBytes(int64(stftBenchmarkChunkSize * 4))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		chunk := chunks[i%len(chunks)]

		if _, err := analyzer.Process(chunk); err != nil {
			b.Fatalf("process STFT chunk: %v", err)
		}
	}
}

// BenchmarkSTFTProcessHopSize measures STFT processing when input is
// delivered exactly one hop at a time
func BenchmarkSTFTProcessHopSize(b *testing.B) {
	analyzer, err := New(stftBenchmarkFFTSize, stftBenchmarkHopSize)
	if err != nil {
		b.Fatalf("create STFT analyzer: %v", err)
	}

	chunks := makeBenchmarkSTFTChunks(64, stftBenchmarkHopSize, stftBenchmarkSampleRate)

	// Warm up the analyzer before benchmarking
	for _, chunk := range chunks {
		if _, err := analyzer.Process(chunk); err != nil {
			b.Fatalf("warm up STFT analyzer: %v", err)
		}
	}

	b.ReportAllocs()

	b.SetBytes(int64(stftBenchmarkHopSize * 4))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		chunk := chunks[i%len(chunks)]

		if _, err := analyzer.Process(chunk); err != nil {
			b.Fatalf("process STFT chunk: %v", err)
		}
	}
}

// BenchmarkSTFTInitialization measures analyzer construction
func BenchmarkSTFTInitialization(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		analyzer, err := New(stftBenchmarkFFTSize, stftBenchmarkHopSize)
		if err != nil {
			b.Fatalf("create STFT analyzer: %v", err)
		}

		if analyzer == nil {
			b.Fatal("expected STFT analyzer")
		}
	}
}

// makeBenchmarkSTFTChunks creates deterministic PCM audio chunks
// containing multiple frequencies so FFT processing is representative
func makeBenchmarkSTFTChunks(count int, chunkSize int, sampleRate int) [][]float32 {
	chunks := make([][]float32, count)

	sampleIndex := 0

	for chunkIndex := range chunks {
		chunk := make([]float32, chunkSize)

		for i := range chunk {
			t := float64(sampleIndex) / float64(sampleRate)

			// Combine several frequencies to produce a realistic
			// non-trivial spectrum
			sample := 0.5*math.Sin(2*math.Pi*200*t) + 0.3*math.Sin(2*math.Pi*700*t) +
				0.1*math.Sin(2*math.Pi*1500*t)

			chunk[i] = float32(sample)

			sampleIndex++
		}

		chunks[chunkIndex] = chunk
	}

	return chunks
}
