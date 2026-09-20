package benchmark

import (
	"fmt"
	"math"
	"runtime"
	"runtime/metrics"
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

	benchmarkPacketsPerCall = int(benchmarkCallDuration / benchmarkPacketDuration)
)

// BenchmarkEndToEndPipeline100ConcurrentTwoMinutes measures the complete
// telephony pipeline for 100 independent concurrent calls.
//
// Run this benchmark with -benchtime=1x because one iteration represents the
// complete 100-call, 120-second audio workload.
func BenchmarkEndToEndPipeline100ConcurrentTwoMinutes(b *testing.B) {
	packets := makeBenchmarkPackets(benchmarkPacketsPerCall)
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

		checksum, err := runConcurrentCalls(packets)

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
}

type resourceMetrics struct {
	cpuSeconds      float64
	totalAllocBytes uint64
}

func readResourceMetrics() resourceMetrics {
	samples := []metrics.Sample{
		{Name: "/cpu/classes/total:cpu-seconds"},
	}

	metrics.Read(samples)

	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)

	return resourceMetrics{
		cpuSeconds:      samples[0].Value.Float64(),
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

func runConcurrentCalls(packets [][]byte) (uint64, error) {
	var (
		waitGroup sync.WaitGroup
		checksums atomic.Uint64
		errors    = make(chan error, benchmarkCallCount)
	)

	waitGroup.Add(benchmarkCallCount)

	for range benchmarkCallCount {
		go func() {
			defer waitGroup.Done()

			checksum, err := runBenchmarkCall(packets)
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

func runBenchmarkCall(packets [][]byte) (uint64, error) {
	pipeline, err := newBenchmarkCallPipeline()
	if err != nil {
		return 0, err
	}

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
		if err := p.processPacket(packet); err != nil {
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
