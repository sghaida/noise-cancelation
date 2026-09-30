package benchmark

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/metrics"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sghaida/noise-cancelation/audio"
	"github.com/sghaida/noise-cancelation/codec/mulaw"
	"github.com/sghaida/noise-cancelation/dsp/highpass"
	"github.com/sghaida/noise-cancelation/dsp/interference"
	"github.com/sghaida/noise-cancelation/dsp/noise"
	"github.com/sghaida/noise-cancelation/dsp/snr"
	"github.com/sghaida/noise-cancelation/dsp/stft"
	"github.com/sghaida/noise-cancelation/dsp/suppressor"
	"github.com/sghaida/noise-cancelation/enhancer/dtln"
)

const (
	benchmarkSampleRate       = 8000
	benchmarkCallCount        = 100
	benchmarkCallDuration     = 2 * time.Minute
	benchmarkPacketDuration   = 20 * time.Millisecond
	benchmarkPacketSamples    = 160
	benchmarkFFTSize          = 256
	benchmarkHopSize          = 128
	benchmarkHighPassCutoff   = float32(80)
	benchmarkBaselineDuration = 5 * time.Second
	benchmarkDTLNChunkSamples = 320
	benchmarkDTLNWorkerCount  = 8

	benchmarkPacketsPerCall     = int(benchmarkCallDuration / benchmarkPacketDuration)
	benchmarkDTLNSamplesPerCall = int(benchmarkCallDuration / (time.Second / dtln.SampleRate))
)

// BenchmarkEndToEndPipeline100ConcurrentTwoMinutes measures the complete
// telephony pipeline for 100 independent concurrent calls.
//
// Run this benchmark with -benchtime=1x because one iteration represents the
// complete 100-call, 120-second audio workload.
func BenchmarkEndToEndPipeline100ConcurrentTwoMinutes(b *testing.B) {
	packets := makeBenchmarkPackets(benchmarkPacketsPerCall)
	chunkLatencies := make([]time.Duration, benchmarkCallCount*benchmarkPacketsPerCall)
	totalCodecBytes := int64(benchmarkCallCount * benchmarkPacketsPerCall * benchmarkPacketSamples * 2)

	b.ReportAllocs()
	b.SetBytes(totalCodecBytes)
	b.ResetTimer()

	var (
		totalWallSeconds float64
		totalCPUSeconds  float64
		totalAllocBytes  uint64
		maxHeapBytes     uint64
		maxRuntimeBytes  uint64
	)

	for iteration := 0; iteration < b.N; iteration++ {
		b.StopTimer()
		runtime.GC()
		b.StartTimer()

		before := readResourceMetrics()
		peak := startPeakMemorySampler()
		started := time.Now()

		checksum, err := runConcurrentCalls(packets, chunkLatencies)

		elapsed := time.Since(started)
		peak.stop()
		after := readResourceMetrics()

		if err != nil {
			b.Fatal(err)
		}

		if checksum == 0 {
			b.Fatal("pipeline produced an empty checksum")
		}

		totalWallSeconds += elapsed.Seconds()
		totalCPUSeconds += after.cpuSeconds - before.cpuSeconds
		totalAllocBytes += after.totalAllocBytes - before.totalAllocBytes

		if peak.heapBytes.Load() > maxHeapBytes {
			maxHeapBytes = peak.heapBytes.Load()
		}

		if peak.runtimeBytes.Load() > maxRuntimeBytes {
			maxRuntimeBytes = peak.runtimeBytes.Load()
		}
	}

	b.StopTimer()
	iterations := float64(b.N)
	averageWallSeconds := totalWallSeconds / iterations
	averageCPUSeconds := totalCPUSeconds / iterations
	averageAllocBytes := float64(totalAllocBytes) / iterations

	b.ReportMetric(averageWallSeconds, "wall-sec")
	b.ReportMetric(averageCPUSeconds, "cpu-sec")
	b.ReportMetric(averageCPUSeconds/averageWallSeconds*100, "cpu-util-%")
	b.ReportMetric(float64(maxHeapBytes)/(1024*1024), "peak-heap-MiB")
	b.ReportMetric(float64(maxRuntimeBytes)/(1024*1024), "peak-runtime-MiB")
	b.ReportMetric(averageAllocBytes/(1024*1024), "alloc-MiB")
	reportLatencyMetrics(b, chunkLatencies, "chunk")
}

// BenchmarkEndToEndDTLN100CallsTwoMinutes measures the DTLN enhancer for 100
// independent calls, each containing two minutes of 16 kHz PCM. Eight workers
// keep the native ONNX session count bounded while processing the full workload.
//
// Set ONNXRUNTIME_SHARED_LIBRARY_PATH and run this benchmark with -benchtime=1x
// because one iteration represents the complete 100-call, 120-second workload.
func BenchmarkEndToEndDTLN100CallsTwoMinutes(b *testing.B) {
	shutdownRuntime := initializeDTLNBenchmarkRuntime(b)
	defer shutdownRuntime()

	samples := makeDTLNBenchmarkSamples()
	chunkLatencies := make([]time.Duration, benchmarkCallCount*benchmarkPacketsPerCall)
	totalPCMBytes := int64(benchmarkCallCount * len(samples) * 4 * 2)

	b.ReportAllocs()
	b.SetBytes(totalPCMBytes)
	b.ResetTimer()

	var (
		totalWallSeconds float64
		totalCPUSeconds  float64
		totalAllocBytes  uint64
		maxHeapBytes     uint64
		maxRuntimeBytes  uint64
	)

	for iteration := 0; iteration < b.N; iteration++ {
		b.StopTimer()
		runtime.GC()
		b.StartTimer()

		before := readResourceMetrics()
		peak := startPeakMemorySampler()
		started := time.Now()

		checksum, err := runConcurrentDTLNCalls(samples, chunkLatencies)

		elapsed := time.Since(started)
		peak.stop()
		after := readResourceMetrics()

		if err != nil {
			b.Fatal(err)
		}

		if checksum == 0 {
			b.Fatal("DTLN produced an empty checksum")
		}

		totalWallSeconds += elapsed.Seconds()
		totalCPUSeconds += after.cpuSeconds - before.cpuSeconds
		totalAllocBytes += after.totalAllocBytes - before.totalAllocBytes

		if peak.heapBytes.Load() > maxHeapBytes {
			maxHeapBytes = peak.heapBytes.Load()
		}

		if peak.runtimeBytes.Load() > maxRuntimeBytes {
			maxRuntimeBytes = peak.runtimeBytes.Load()
		}
	}

	b.StopTimer()
	iterations := float64(b.N)
	averageWallSeconds := totalWallSeconds / iterations
	averageCPUSeconds := totalCPUSeconds / iterations
	averageAllocBytes := float64(totalAllocBytes) / iterations

	b.ReportMetric(averageWallSeconds, "wall-sec")
	b.ReportMetric(averageCPUSeconds, "cpu-sec")
	b.ReportMetric(averageCPUSeconds/averageWallSeconds*100, "cpu-util-%")
	b.ReportMetric(float64(maxHeapBytes)/(1024*1024), "peak-heap-MiB")
	b.ReportMetric(float64(maxRuntimeBytes)/(1024*1024), "peak-runtime-MiB")
	b.ReportMetric(averageAllocBytes/(1024*1024), "alloc-MiB")
	reportLatencyMetrics(b, chunkLatencies, "chunk")
}

// BenchmarkEndToEndDTLNScheduled100ConcurrentTwoMinutes measures 100
// independent stateful DTLN streams sharing eight deadline-aware workers.
func BenchmarkEndToEndDTLNScheduled100ConcurrentTwoMinutes(b *testing.B) {
	shutdownRuntime := initializeDTLNBenchmarkRuntime(b)
	defer shutdownRuntime()

	samples := makeDTLNBenchmarkSamples()
	chunkLatencies := make([]time.Duration, benchmarkCallCount*benchmarkPacketsPerCall)
	queueLatencies := make([]time.Duration, benchmarkCallCount*benchmarkPacketsPerCall)
	processingLatency := make([]time.Duration, benchmarkCallCount*benchmarkPacketsPerCall)
	workerCount := min(benchmarkCallCount, runtime.GOMAXPROCS(0))
	totalPCMBytes := int64(benchmarkCallCount * len(samples) * 4 * 2)

	b.ReportAllocs()
	b.SetBytes(totalPCMBytes)
	b.ResetTimer()

	var (
		totalWallSeconds float64
		totalCPUSeconds  float64
		totalAllocBytes  uint64
		maxHeapBytes     uint64
		maxRuntimeBytes  uint64
		deadlineMisses   int
	)

	for iteration := 0; iteration < b.N; iteration++ {
		b.StopTimer()
		runtime.GC()
		b.StartTimer()

		before := readResourceMetrics()
		peak := startPeakMemorySampler()
		started := time.Now()

		checksum, misses, err := runScheduledDTLNCalls(
			samples,
			workerCount,
			chunkLatencies,
			queueLatencies,
			processingLatency,
		)

		elapsed := time.Since(started)
		peak.stop()
		after := readResourceMetrics()

		if err != nil {
			b.Fatal(err)
		}
		if checksum == 0 {
			b.Fatal("scheduled DTLN produced an empty checksum")
		}

		totalWallSeconds += elapsed.Seconds()
		totalCPUSeconds += after.cpuSeconds - before.cpuSeconds
		totalAllocBytes += after.totalAllocBytes - before.totalAllocBytes
		deadlineMisses += misses

		if peak.heapBytes.Load() > maxHeapBytes {
			maxHeapBytes = peak.heapBytes.Load()
		}
		if peak.runtimeBytes.Load() > maxRuntimeBytes {
			maxRuntimeBytes = peak.runtimeBytes.Load()
		}
	}

	b.StopTimer()
	iterations := float64(b.N)
	averageWallSeconds := totalWallSeconds / iterations
	averageCPUSeconds := totalCPUSeconds / iterations
	averageAllocBytes := float64(totalAllocBytes) / iterations

	b.ReportMetric(averageWallSeconds, "wall-sec")
	b.ReportMetric(averageCPUSeconds, "cpu-sec")
	b.ReportMetric(averageCPUSeconds/averageWallSeconds*100, "cpu-util-%")
	b.ReportMetric(float64(maxHeapBytes)/(1024*1024), "peak-heap-MiB")
	b.ReportMetric(float64(maxRuntimeBytes)/(1024*1024), "peak-runtime-MiB")
	b.ReportMetric(averageAllocBytes/(1024*1024), "alloc-MiB")
	b.ReportMetric(float64(workerCount), "workers")
	b.ReportMetric(float64(deadlineMisses)/float64(len(chunkLatencies))*100, "deadline-miss-%")
	reportLatencyMetrics(b, chunkLatencies, "chunk")
	reportLatencyMetrics(b, queueLatencies, "queue")
	reportLatencyMetrics(b, processingLatency, "processing")
}

func runScheduledDTLNCalls(
	samples []float32,
	workerCount int,
	chunkLatencies []time.Duration,
	queueLatencies []time.Duration,
	processingLatencies []time.Duration,
) (uint64, int, error) {
	scheduler, err := dtln.NewScheduler(workerCount)
	if err != nil {
		return 0, 0, err
	}

	processors := make([]*dtln.Processor, 0, benchmarkCallCount)
	streams := make([]*dtln.ScheduledStream, 0, benchmarkCallCount)
	closeResources := func() {
		for index, processor := range processors {
			if index < len(streams) {
				_ = streams[index].Close(context.Background())
				continue
			}

			_ = processor.Close()
		}
		scheduler.Close()
	}

	for range benchmarkCallCount {
		processor, createErr := newDTLNBenchmarkProcessor()
		if createErr != nil {
			closeResources()

			return 0, 0, createErr
		}

		stream, streamErr := scheduler.NewStream(processor)
		if streamErr != nil {
			_ = processor.Close()
			closeResources()

			return 0, 0, streamErr
		}

		processors = append(processors, processor)
		streams = append(streams, stream)
	}

	var (
		waitGroup sync.WaitGroup
		checksums atomic.Uint64
		misses    atomic.Int64
		errors    = make(chan error, benchmarkCallCount)
	)
	started := time.Now()
	waitGroup.Add(benchmarkCallCount)

	for callIndex := range benchmarkCallCount {
		go func() {
			defer waitGroup.Done()

			latencyStart := callIndex * benchmarkPacketsPerCall
			latencyEnd := latencyStart + benchmarkPacketsPerCall
			checksum, deadlineMisses, processErr := processScheduledDTLNCall(
				processors[callIndex],
				streams[callIndex],
				samples,
				started,
				chunkLatencies[latencyStart:latencyEnd],
				queueLatencies[latencyStart:latencyEnd],
				processingLatencies[latencyStart:latencyEnd],
			)
			if processErr != nil {
				errors <- processErr

				return
			}

			checksums.Add(checksum)
			misses.Add(int64(deadlineMisses))
		}()
	}

	waitGroup.Wait()
	close(errors)
	closeResources()

	for processErr := range errors {
		return 0, 0, processErr
	}

	return checksums.Load(), int(misses.Load()), nil
}

func processScheduledDTLNCall(
	processor *dtln.Processor,
	stream *dtln.ScheduledStream,
	samples []float32,
	started time.Time,
	chunkLatencies []time.Duration,
	queueLatencies []time.Duration,
	processingLatencies []time.Duration,
) (uint64, int, error) {
	var checksum uint64
	deadlineMisses := 0
	outputSamples := 0
	outputBuffer := make([]float32, 0, 3*dtln.BlockShift)

	for chunkIndex, start := 0, 0; start < len(samples); chunkIndex, start = chunkIndex+1, start+benchmarkDTLNChunkSamples {
		end := min(start+benchmarkDTLNChunkSamples, len(samples))
		deadline := started.Add(time.Duration(end) * time.Second / dtln.SampleRate)
		callStarted := time.Now()
		output, metrics, err := stream.Process(
			context.Background(),
			deadline,
			outputBuffer[:0],
			samples[start:end],
		)
		chunkLatencies[chunkIndex] = time.Since(callStarted)
		queueLatencies[chunkIndex] = metrics.QueueDuration
		processingLatencies[chunkIndex] = metrics.ProcessingDuration
		if metrics.DeadlineMissed {
			deadlineMisses++
		}
		if err != nil {
			return 0, 0, fmt.Errorf("schedule DTLN samples at %d: %w", start, err)
		}

		outputSamples += len(output)
		checksum = checksumDTLNSamples(checksum, output)
	}

	flushDeadline := started.Add(benchmarkCallDuration + benchmarkPacketDuration)
	output, _, err := stream.Flush(
		context.Background(),
		flushDeadline,
		outputBuffer[:0],
	)
	if err != nil {
		return 0, 0, fmt.Errorf("flush scheduled DTLN processor: %w", err)
	}
	outputSamples += len(output)
	checksum = checksumDTLNSamples(checksum, output)

	if outputSamples != len(samples) {
		return 0, 0, fmt.Errorf(
			"scheduled DTLN produced %d samples, want %d",
			outputSamples,
			len(samples),
		)
	}

	return checksum, deadlineMisses, nil
}

// BenchmarkDTLNProcess20MillisecondChunk compares ONNX thread counts using
// the same 320-sample chunk size as the end-to-end DTLN workload.
func BenchmarkDTLNProcess20MillisecondChunk(b *testing.B) {
	shutdownRuntime := initializeDTLNBenchmarkRuntime(b)
	defer shutdownRuntime()

	samples := makeDTLNBenchmarkSamples()[:benchmarkDTLNChunkSamples]
	tests := []struct {
		name              string
		intraOpNumThreads int
		interOpNumThreads int
	}{
		{name: "ORTDefault", intraOpNumThreads: 0, interOpNumThreads: 0},
		{name: "OneThread", intraOpNumThreads: 1, interOpNumThreads: 1},
		{name: "TwoThreads", intraOpNumThreads: 2, interOpNumThreads: 1},
	}

	for _, test := range tests {
		b.Run(test.name, func(b *testing.B) {
			processor, err := newDTLNBenchmarkProcessorWithThreads(
				test.intraOpNumThreads,
				test.interOpNumThreads,
			)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = processor.Close() }()

			output := make([]float32, 0, 3*dtln.BlockShift)
			b.ReportAllocs()
			b.ResetTimer()

			for b.Loop() {
				if _, err := processor.ProcessInto(output[:0], samples); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func initializeDTLNBenchmarkRuntime(b *testing.B) func() {
	b.Helper()

	runtimePath := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	if runtimePath == "" {
		b.Fatal("ONNXRUNTIME_SHARED_LIBRARY_PATH is required")
	}

	if err := dtln.InitializeRuntime(runtimePath); err != nil {
		b.Fatalf("initialize ONNX Runtime: %v", err)
	}

	return func() {
		if err := dtln.ShutdownRuntime(); err != nil {
			b.Errorf("shutdown ONNX Runtime: %v", err)
		}
	}
}

func makeDTLNBenchmarkSamples() []float32 {
	samples := make([]float32, benchmarkDTLNSamplesPerCall)

	for sampleIndex := range samples {
		timeSeconds := float64(sampleIndex) / dtln.SampleRate
		samples[sampleIndex] = float32(
			0.42*math.Sin(2*math.Pi*220*timeSeconds) +
				0.18*math.Sin(2*math.Pi*700*timeSeconds),
		)
	}

	return samples
}

func runConcurrentDTLNCalls(samples []float32, chunkLatencies []time.Duration) (uint64, error) {
	var (
		waitGroup sync.WaitGroup
		checksums atomic.Uint64
		errors    = make(chan error, benchmarkDTLNWorkerCount)
	)

	workerCount := min(benchmarkCallCount, benchmarkDTLNWorkerCount)
	waitGroup.Add(workerCount)

	for workerIndex := range workerCount {
		go func() {
			defer waitGroup.Done()

			processor, err := newDTLNBenchmarkProcessor()
			if err != nil {
				errors <- err
				return
			}
			defer func() { _ = processor.Close() }()

			for callIndex := workerIndex; callIndex < benchmarkCallCount; callIndex += workerCount {
				processor.Reset()
				latencyStart := callIndex * benchmarkPacketsPerCall
				latencyEnd := latencyStart + benchmarkPacketsPerCall
				checksum, processErr := processDTLNCall(
					processor,
					samples,
					chunkLatencies[latencyStart:latencyEnd],
				)
				if processErr != nil {
					errors <- processErr

					return
				}

				checksums.Add(checksum)
			}
		}()
	}

	waitGroup.Wait()
	close(errors)

	for err := range errors {
		return 0, err
	}

	return checksums.Load(), nil
}

func newDTLNBenchmarkProcessor() (*dtln.Processor, error) {
	config := dtln.DefaultConfig()

	return newDTLNBenchmarkProcessorWithConfig(config)
}

func newDTLNBenchmarkProcessorWithThreads(intraOpNumThreads, interOpNumThreads int) (*dtln.Processor, error) {
	config := dtln.DefaultConfig()
	config.IntraOpNumThreads = intraOpNumThreads
	config.InterOpNumThreads = interOpNumThreads

	return newDTLNBenchmarkProcessorWithConfig(config)
}

func newDTLNBenchmarkProcessorWithConfig(config dtln.Config) (*dtln.Processor, error) {
	config.Model1Path = "../models/model_1.onnx"
	config.Model2Path = "../models/model_2.onnx"

	processor, err := dtln.New(config)
	if err != nil {
		return nil, fmt.Errorf("create DTLN processor: %w", err)
	}

	return processor, nil
}

func processDTLNCall(
	processor *dtln.Processor,
	samples []float32,
	chunkLatencies []time.Duration,
) (checksum uint64, err error) {
	outputSamples := 0
	outputBuffer := make(
		[]float32,
		0,
		((benchmarkDTLNChunkSamples+dtln.BlockShift-1)/dtln.BlockShift)*dtln.BlockShift,
	)

	for chunkIndex, start := 0, 0; start < len(samples); chunkIndex, start = chunkIndex+1, start+benchmarkDTLNChunkSamples {
		end := min(start+benchmarkDTLNChunkSamples, len(samples))
		started := time.Now()
		output, processErr := processor.ProcessInto(outputBuffer[:0], samples[start:end])
		chunkLatencies[chunkIndex] = time.Since(started)
		if processErr != nil {
			return 0, fmt.Errorf("process DTLN samples at %d: %w", start, processErr)
		}

		outputSamples += len(output)
		checksum = checksumDTLNSamples(checksum, output)
	}

	output, flushErr := processor.FlushInto(outputBuffer[:0])
	if flushErr != nil {
		return 0, fmt.Errorf("flush DTLN processor: %w", flushErr)
	}

	outputSamples += len(output)
	checksum = checksumDTLNSamples(checksum, output)

	if outputSamples != len(samples) {
		return 0, fmt.Errorf("DTLN produced %d samples, want %d", outputSamples, len(samples))
	}

	return checksum, nil
}

func checksumDTLNSamples(checksum uint64, samples []float32) uint64 {
	for _, sample := range samples {
		checksum = checksum*131 + uint64(math.Float32bits(sample)) + 1
	}

	return checksum
}

type resourceMetrics struct {
	cpuSeconds      float64
	totalAllocBytes uint64
}

func readResourceMetrics() resourceMetrics {
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)

	return resourceMetrics{
		cpuSeconds:      readProcessCPUSeconds(),
		totalAllocBytes: memory.TotalAlloc,
	}
}

type peakMemorySampler struct {
	heapBytes    atomic.Uint64
	runtimeBytes atomic.Uint64
	stopSignal   chan struct{}
	stopped      chan struct{}
}

func startPeakMemorySampler() *peakMemorySampler {
	sampler := &peakMemorySampler{
		stopSignal: make(chan struct{}),
		stopped:    make(chan struct{}),
	}

	sampler.sample()

	go func() {
		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()
		defer close(sampler.stopped)

		for {
			select {
			case <-ticker.C:
				sampler.sample()
			case <-sampler.stopSignal:
				return
			}
		}
	}()

	return sampler
}

func (s *peakMemorySampler) sample() {
	samples := []metrics.Sample{
		{Name: "/memory/classes/heap/objects:bytes"},
		{Name: "/memory/classes/total:bytes"},
	}

	metrics.Read(samples)
	updatePeak(&s.heapBytes, samples[0].Value.Uint64())
	updatePeak(&s.runtimeBytes, samples[1].Value.Uint64())
}

func (s *peakMemorySampler) stop() {
	close(s.stopSignal)
	<-s.stopped
	s.sample()
}

func updatePeak(target *atomic.Uint64, value uint64) {
	for {
		current := target.Load()

		if value <= current || target.CompareAndSwap(current, value) {
			return
		}
	}
}

func runConcurrentCalls(packets [][]byte, chunkLatencies []time.Duration) (uint64, error) {
	var (
		waitGroup sync.WaitGroup
		checksums atomic.Uint64
		errors    = make(chan error, benchmarkCallCount)
	)

	waitGroup.Add(benchmarkCallCount)

	for callIndex := range benchmarkCallCount {
		go func() {
			defer waitGroup.Done()

			latencyStart := callIndex * benchmarkPacketsPerCall
			latencyEnd := latencyStart + benchmarkPacketsPerCall
			checksum, err := runBenchmarkCall(packets, chunkLatencies[latencyStart:latencyEnd])
			if err != nil {
				errors <- err

				return
			}

			checksums.Add(checksum)
		}()
	}

	waitGroup.Wait()
	close(errors)

	for err := range errors {
		return 0, err
	}

	return checksums.Load(), nil
}

func runBenchmarkCall(packets [][]byte, chunkLatencies []time.Duration) (uint64, error) {
	pipeline, err := newBenchmarkCallPipeline()
	if err != nil {
		return 0, err
	}
	pipeline.chunkLatencies = chunkLatencies[:0]

	return pipeline.process(packets)
}

type benchmarkCallPipeline struct {
	highPass       *highpass.Filter
	analyzer       *stft.Analyzer
	synthesizer    *stft.Synthesizer
	noiseEstimator *noise.SPPMMSEEstimator
	logMMSE        *suppressor.LogMMSE
	toneDetector   *interference.TonalTransientDetector

	decoded         []float32
	encoded         []byte
	baselineSamples int
	frameCount      int
	baselineDone    bool
	suppressedCount int
	checksum        uint64
	chunkLatencies  []time.Duration
}

func newBenchmarkCallPipeline() (*benchmarkCallPipeline, error) {
	highPass := highpass.New(benchmarkSampleRate, benchmarkHighPassCutoff)

	analyzer, err := stft.New(benchmarkFFTSize, benchmarkHopSize)
	if err != nil {
		return nil, fmt.Errorf("create STFT analyzer: %w", err)
	}

	synthesizer, err := stft.NewSynthesizer(benchmarkFFTSize, benchmarkHopSize)
	if err != nil {
		return nil, fmt.Errorf("create ISTFT synthesizer: %w", err)
	}

	noiseEstimator := noise.NewSPPMMSEEstimator(noise.DefaultSPPMMSEConfig())
	noiseEstimator.StartBaseline()

	logMMSE := suppressor.NewLogMMSE(
		suppressor.DefaultLogMMSEConfig(),
		snr.DefaultDecisionDirectedConfig(),
	)
	toneDetector := interference.NewTonalTransientDetector(
		interference.DefaultTonalTransientConfig(),
	)

	return &benchmarkCallPipeline{
		highPass:        highPass,
		analyzer:        analyzer,
		synthesizer:     synthesizer,
		noiseEstimator:  noiseEstimator,
		logMMSE:         logMMSE,
		toneDetector:    toneDetector,
		decoded:         make([]float32, benchmarkPacketSamples),
		encoded:         make([]byte, benchmarkHopSize),
		baselineSamples: int(benchmarkBaselineDuration.Seconds() * benchmarkSampleRate),
	}, nil
}

func (p *benchmarkCallPipeline) process(packets [][]byte) (uint64, error) {
	for _, packet := range packets {
		started := time.Now()
		err := p.processPacket(packet)
		p.chunkLatencies = append(p.chunkLatencies, time.Since(started))
		if err != nil {
			return 0, err
		}
	}

	finalSpectra, err := p.analyzer.Flush()
	if err != nil {
		return 0, fmt.Errorf("flush STFT: %w", err)
	}

	for _, spectrum := range finalSpectra {
		if err := p.processSpectrum(spectrum); err != nil {
			return 0, err
		}
	}

	if !p.baselineDone {
		p.noiseEstimator.FinishBaseline()
	}

	p.consumePCM(p.synthesizer.Flush())

	if p.frameCount == 0 || p.suppressedCount == 0 {
		return 0, fmt.Errorf(
			"pipeline processed %d frames and %d suppressed frames",
			p.frameCount, p.suppressedCount,
		)
	}

	return p.checksum, nil
}

func (p *benchmarkCallPipeline) processPacket(packet []byte) error {
	p.decoded = mulaw.DecodeMuLaw(p.decoded, packet)
	filtered, err := p.highPass.Process(p.decoded)
	if err != nil {
		return fmt.Errorf("process high-pass frame: %w", err)
	}

	spectra, err := p.analyzer.Process(filtered)
	if err != nil {
		return fmt.Errorf("process STFT frame: %w", err)
	}

	for _, spectrum := range spectra {
		if err := p.processSpectrum(spectrum); err != nil {
			return err
		}
	}

	return nil
}

func (p *benchmarkCallPipeline) processSpectrum(spectrum audio.Spectrum) error {
	frameStartSample := p.frameCount * benchmarkHopSize
	isBaselineFrame := frameStartSample+benchmarkFFTSize <= p.baselineSamples

	if !isBaselineFrame && !p.baselineDone {
		p.noiseEstimator.FinishBaseline()
		p.baselineDone = true
	}

	power := spectrum.RecomputePower()
	noisePSD := p.noiseEstimator.Process(power)
	outputSpectrum := spectrum

	if !isBaselineFrame {
		var err error
		outputSpectrum, err = p.logMMSE.Process(spectrum, noisePSD)
		if err != nil {
			return fmt.Errorf("process Log-MMSE frame %d: %w", p.frameCount, err)
		}

		interferenceResult := p.toneDetector.Process(power)
		if err := interference.ApplyGain(&outputSpectrum, interferenceResult.Gain); err != nil {
			return fmt.Errorf("apply interference gain frame %d: %w", p.frameCount, err)
		}

		p.suppressedCount++
	}

	reconstructed, err := p.synthesizer.Process(outputSpectrum)
	if err != nil {
		return fmt.Errorf("process ISTFT frame %d: %w", p.frameCount, err)
	}

	p.consumePCM(reconstructed)
	p.frameCount++

	return nil
}

func (p *benchmarkCallPipeline) consumePCM(samples []float32) {
	p.encoded = mulaw.EncodeMuLaw(p.encoded[:0], samples)

	for _, sample := range p.encoded {
		p.checksum = p.checksum*131 + uint64(sample) + 1
	}
}

func reportLatencyMetrics(b *testing.B, latencies []time.Duration, prefix string) {
	b.Helper()

	if len(latencies) == 0 {
		return
	}

	slices.Sort(latencies)
	var total time.Duration
	for _, latency := range latencies {
		total += latency
	}

	b.ReportMetric(float64(total)/float64(len(latencies))/float64(time.Millisecond), prefix+"-avg-ms")
	b.ReportMetric(float64(latencies[len(latencies)/2])/float64(time.Millisecond), prefix+"-p50-ms")
	b.ReportMetric(float64(latencies[percentileIndex(len(latencies), 95)])/float64(time.Millisecond), prefix+"-p95-ms")
	b.ReportMetric(float64(latencies[percentileIndex(len(latencies), 99)])/float64(time.Millisecond), prefix+"-p99-ms")
}

func percentileIndex(length, percentile int) int {
	return min((length*percentile+99)/100-1, length-1)
}

func makeBenchmarkPackets(packetCount int) [][]byte {
	packets := make([][]byte, packetCount)
	phase := uint32(0x12345678)
	sampleIndex := 0

	for packetIndex := range packets {
		packet := make([]byte, benchmarkPacketSamples)

		for sampleOffset := range packet {
			timeSeconds := float64(sampleIndex) / benchmarkSampleRate
			phase = phase*1664525 + 1013904223
			noiseSample := float64(phase>>8)/float64(1<<24)*2 - 1
			sample := 0.42*math.Sin(2*math.Pi*220*timeSeconds) +
				0.18*math.Sin(2*math.Pi*700*timeSeconds) +
				0.025*noiseSample

			if sample > 1 {
				sample = 1
			} else if sample < -1 {
				sample = -1
			}

			packet[sampleOffset] = mulaw.ToMuLaw(int16(sample * 32767))
			sampleIndex++
		}

		packets[packetIndex] = packet
	}

	return packets
}
