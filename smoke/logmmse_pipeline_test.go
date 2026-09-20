//go:build smoke
// +build smoke

package smoke

import (
	"encoding/csv"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/sghaida/noise-cancelation/audio"
	"github.com/sghaida/noise-cancelation/dsp/highpass"
	"github.com/sghaida/noise-cancelation/dsp/noise"
	"github.com/sghaida/noise-cancelation/dsp/snr"
	"github.com/sghaida/noise-cancelation/dsp/stft"
	"github.com/sghaida/noise-cancelation/dsp/suppressor"
)

const (
	logMMSEFixturePath = "fixtures/human_voice_with_noise_with_5s_baseline_8khz_pcm16.wav"

	logMMSEOutputPath    = "output/logmmse.out.wav"
	logMMSECSVOutputPath = "output/logmmse.out.csv"

	logMMSESampleRate = 8000

	logMMSEFFTSize   = 256
	logMMSEHopSize   = 128
	logMMSEChunkSize = 160

	logMMSEHighPassCutoff = float32(80)

	logMMSEBaselineDuration = 5 * time.Second
)

// TestLogMMSEPipelineSmoke processes real audio through the complete
// noise-suppression pipeline
//
// The pipeline is:
//
//	PCM
//	 ↓
//	80 Hz high-pass filter
//	 ↓
//	STFT
//	 ↓
//	power spectrum
//	 ↓
//	MCRA noise estimation
//	 ↓
//	Log-MMSE suppression
//	 ↓
//	ISTFT
//	 ↓
//	reconstructed PCM
//
// first five seconds contain known background noise and are used to
// initialize the MCRA noise baseline
//
// Log MMSE suppression begins only after the known baseline period finishes
// so the suppressor starts with a stable noise PSD estimate
//
// The smoke test verifies that:
//
//   - real PCM audio passes through the high pass filter
//   - STFT produces valid spectra
//   - only complete STFT windows inside the first five seconds are used
//     for MCRA baseline initialization
//   - MCRA produces a valid noise PSD
//   - Log MMSE suppresses post baseline spectra
//   - ISTFT reconstructs time domain PCM
//   - reconstructed samples contain no NaN or infinity
//   - the final reconstructed audio can be written as a WAV file
func TestLogMMSEPipelineSmoke(t *testing.T) {
	sampleRate, channels, bitDepth, samples, err := readPCM16MonoWAV(logMMSEFixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	if sampleRate != logMMSESampleRate {
		t.Fatalf("sample rate = %d, want %d", sampleRate, logMMSESampleRate)
	}

	if channels != expectedChannels {
		t.Fatalf("channels = %d, want %d", channels, expectedChannels)
	}

	if bitDepth != expectedBitDepth {
		t.Fatalf("bit depth = %d, want %d", bitDepth, expectedBitDepth)
	}

	highPass := highpass.New(sampleRate, logMMSEHighPassCutoff)

	analyzer, err := stft.New(logMMSEFFTSize, logMMSEHopSize)
	if err != nil {
		t.Fatalf("create STFT analyzer: %v", err)
	}

	synthesizer, err := stft.NewSynthesizer(logMMSEFFTSize, logMMSEHopSize)
	if err != nil {
		t.Fatalf("create ISTFT synthesizer: %v", err)
	}

	mcraConfig := noise.DefaultMCRAConfig()

	mcraConfig.SampleRate = sampleRate

	mcraConfig.HopSize = logMMSEHopSize

	mcraConfig.BaselineDuration = logMMSEBaselineDuration

	mcraConfig.MinProbabilityFrequency = logMMSEHighPassCutoff

	noiseEstimator := noise.NewMCRAEstimator(mcraConfig)

	noiseEstimator.StartBaseline()

	logMMSEConfig := suppressor.DefaultLogMMSEConfig()
	snrConfig := snr.DefaultDecisionDirectedConfig()

	logMMSE := suppressor.NewLogMMSE(logMMSEConfig, snrConfig)

	diagnosticSNR := snr.NewDecisionDirected(snrConfig)

	baselineSamples := int(logMMSEBaselineDuration.Seconds() * float64(sampleRate))

	output := make([]float32, 0, len(samples))

	if err := os.MkdirAll(filepath.Dir(logMMSECSVOutputPath), 0o755); err != nil {
		t.Fatalf("create output directory: %v", err)
	}

	csvFile, err := os.Create(logMMSECSVOutputPath)
	if err != nil {
		t.Fatalf("create Log-MMSE CSV: %v", err)
	}

	defer func() {
		if err := csvFile.Close(); err != nil {
			t.Errorf("close Log-MMSE CSV: %v", err)
		}
	}()

	csvWriter := csv.NewWriter(csvFile)

	defer csvWriter.Flush()

	header := []string{
		"time_seconds",
		"frame",
		"phase",
		"bin",
		"frequency_hz",
		"observed_power",
		"mcra_noise_power",
		"effective_noise_power",
		"gamma",
		"xi",
		"gain",
		"output_power",
		"attenuation_db",
	}

	if err := csvWriter.Write(header); err != nil {
		t.Fatalf("write Log-MMSE CSV header: %v", err)
	}

	var (
		frameCount           int
		baselineFrameCount   int
		suppressedFrameCount int
		baselineFinished     bool
		diagnosticNoisePSD   []float32
	)

	// processSpectrum contains the common spectral processing path used by
	// both normal STFT output and the final flushed STFT frame
	processSpectrum := func(spectrum audio.Spectrum) {
		t.Helper()

		frameStartSample := frameCount * logMMSEHopSize

		frameEndSample := frameStartSample + logMMSEFFTSize

		// frame belongs to the baseline only when its complete
		// analysis window is contained inside the first five seconds
		isBaselineFrame := frameEndSample <= baselineSamples

		// finish baseline before processing the first frame that crosses
		// or follows the five-second boundary
		if !isBaselineFrame && !baselineFinished {
			noiseEstimator.FinishBaseline()

			baselineFinished = true

			t.Logf(
				"MCRA baseline finished after %d frames at %.3f seconds",
				baselineFrameCount, float64(frameStartSample)/float64(sampleRate),
			)
		}

		power := spectrum.RecomputePower()

		noisePSD := noiseEstimator.Process(power)

		assertPipelineSpectrum(
			t, frameCount, power, noisePSD, mcraConfig.Floor,
		)

		outputSpectrum := spectrum
		var (
			gamma []float32
			xi    []float32
		)

		if isBaselineFrame {
			// during baseline collection the original spectrum is passed
			// through unchanged
			// MCRA uses these frames only to learn the initial noise PSD
			baselineFrameCount++
		} else {
			// Reproduce the same effective noise PSD used internally by
			// Log-MMSE so diagnostic gamma and xi values match the suppressor
			if len(diagnosticNoisePSD) != len(noisePSD) {
				diagnosticNoisePSD = make([]float32, len(noisePSD))
			}

			for k, noisePower := range noisePSD {
				if noisePower < 0 || math.IsNaN(float64(noisePower)) || math.IsInf(float64(noisePower), 0) {
					noisePower = 0
				}
				diagnosticNoisePSD[k] = noisePower * logMMSEConfig.NoiseOverestimation
			}

			gamma, xi, err = diagnosticSNR.Process(power, diagnosticNoisePSD)
			if err != nil {
				t.Fatalf("process diagnostic SNR frame %d: %v", frameCount, err)
			}

			outputSpectrum, err = logMMSE.Process(spectrum, noisePSD)
			if err != nil {
				t.Fatalf("process Log-MMSE frame %d: %v", frameCount, err)
			}

			assertSuppressedSpectrum(t, frameCount, outputSpectrum)

			// Keep the diagnostic SNR estimator synchronized with the actual
			// Log-MMSE estimator using the same clean power and effective noise
			if err := diagnosticSNR.Update(outputSpectrum.Power, diagnosticNoisePSD); err != nil {
				t.Fatalf("update diagnostic SNR frame %d: %v", frameCount, err)
			}

			suppressedFrameCount++
		}

		writeLogMMSECSVFrame(
			t, csvWriter, frameCount, frameStartSample, sampleRate, power,
			noisePSD, diagnosticNoisePSD, gamma, xi,
			outputSpectrum, isBaselineFrame, mcraConfig.Floor,
		)

		reconstructed, err := synthesizer.Process(outputSpectrum)
		if err != nil {
			t.Fatalf("process ISTFT frame %d: %v", frameCount, err)
		}

		assertPCMFinite(t, frameCount, reconstructed)

		output = append(output, reconstructed...)

		frameCount++
	}

	// deed audio using the same 160 sample chunk size used by an
	// 8 kHz Twilio Media Stream
	for start := 0; start < len(samples); start += logMMSEChunkSize {

		end := start + logMMSEChunkSize

		if end > len(samples) {
			end = len(samples)
		}

		filtered, err := highPass.Process(samples[start:end])
		if err != nil {
			t.Fatalf("high-pass samples at %d: %v", start, err)
		}

		spectra, err := analyzer.Process(filtered)
		if err != nil {
			t.Fatalf("process STFT samples at %d: %v", start, err)
		}

		for i := range spectra {
			processSpectrum(spectra[i])
		}
	}

	// flush any remaining PCM samples from the STFT analyzer
	finalSpectra, err := analyzer.Flush()
	if err != nil {
		t.Fatalf("flush STFT analyzer: %v", err)
	}

	for i := range finalSpectra {
		processSpectrum(finalSpectra[i])
	}

	// normally baseline completion occurs when processing crosses the
	// five seconds boundary
	// keep this guard for accidentally shortened fixtures
	if !baselineFinished {
		noiseEstimator.FinishBaseline()
		baselineFinished = true
	}

	// flush the remaining overlap-add samples from ISTFT
	remaining := synthesizer.Flush()

	assertPCMFinite(t, frameCount, remaining)

	output = append(output, remaining...)

	if frameCount == 0 {
		t.Fatal("no STFT frames were processed")
	}

	if baselineFrameCount == 0 {
		t.Fatal("no baseline frames were processed")
	}

	if suppressedFrameCount == 0 {
		t.Fatal("no Log-MMSE frames were processed")
	}

	// STFT flush zero pads the final partial frame and ISTFT emits the
	// remaining overlap samples
	//
	// trim this padding so the output wav has the same duration as the
	// original fixture
	if len(output) > len(samples) {
		output = output[:len(samples)]
	}

	if len(output) == 0 {
		t.Fatal("no reconstructed PCM samples were produced")
	}

	if len(output) != len(samples) {
		t.Fatalf("output contains %d samples, want %d", len(output), len(samples))
	}

	if err := writePCM16MonoWAV(logMMSEOutputPath, sampleRate, output); err != nil {
		t.Fatalf("write Log-MMSE output: %v", err)
	}

	csvWriter.Flush()

	if err := csvWriter.Error(); err != nil {
		t.Fatalf("flush Log-MMSE CSV: %v", err)
	}

	t.Logf("processed %d STFT frames", frameCount)
	t.Logf("baseline frames: %d", baselineFrameCount)
	t.Logf("Log-MMSE suppressed frames: %d", suppressedFrameCount)
	t.Logf("input samples: %d", len(samples))
	t.Logf("output samples: %d", len(output))
	t.Logf("Log-MMSE output written to %s", logMMSEOutputPath)
	t.Logf("Log-MMSE diagnostics written to %s", logMMSECSVOutputPath)

	logMMSE.Reset()
	noiseEstimator.Reset()
	synthesizer.Reset()
	analyzer.Reset()
	highPass.Reset()
}

// assertPipelineSpectrum verifies observed and estimated spectral power
// remain valid throughout the complete suppression pipeline
func assertPipelineSpectrum(
	t *testing.T, frame int, power []float32, noisePSD []float32, floor float32,
) {
	t.Helper()

	if len(power) == 0 {
		t.Fatalf("frame %d contains no power bins", frame)
	}

	if len(noisePSD) != len(power) {
		t.Fatalf(
			"frame %d noise PSD contains %d bins, want %d", frame, len(noisePSD), len(power),
		)
	}

	for k := range power {
		if math.IsNaN(float64(power[k])) {
			t.Fatalf("frame %d power bin %d is NaN", frame, k)
		}

		if math.IsInf(float64(power[k]), 0) {
			t.Fatalf("frame %d power bin %d is infinite", frame, k)
		}

		if math.IsNaN(float64(noisePSD[k])) {
			t.Fatalf("frame %d noise PSD bin %d is NaN", frame, k)
		}

		if math.IsInf(float64(noisePSD[k]), 0) {
			t.Fatalf("frame %d noise PSD bin %d is infinite", frame, k)
		}

		if noisePSD[k] < floor {
			t.Fatalf(
				"frame %d noise PSD bin %d = %v, below floor %v",
				frame, k, noisePSD[k], floor,
			)
		}
	}
}

// assertSuppressedSpectrum verifies Log MMSE output contains valid complex
// bins and power values
func assertSuppressedSpectrum(t *testing.T, frame int, spectrum audio.Spectrum) {
	t.Helper()

	if len(spectrum.Bins) == 0 {
		t.Fatalf("frame %d suppressed spectrum contains no bins", frame)
	}

	if len(spectrum.Power) != len(spectrum.Bins) {
		t.Fatalf(
			"frame %d suppressed power contains %d bins, want %d",
			frame, len(spectrum.Power), len(spectrum.Bins),
		)
	}

	for k, bin := range spectrum.Bins {
		re := float64(real(bin))
		im := float64(imag(bin))

		if math.IsNaN(re) || math.IsNaN(im) {
			t.Fatalf("frame %d suppressed bin %d contains NaN", frame, k)
		}

		if math.IsInf(re, 0) || math.IsInf(im, 0) {
			t.Fatalf("frame %d suppressed bin %d contains infinity", frame, k)
		}

		power := float64(spectrum.Power[k])

		if math.IsNaN(power) {
			t.Fatalf("frame %d suppressed power bin %d is NaN", frame, k)
		}

		if math.IsInf(power, 0) {
			t.Fatalf("frame %d suppressed power bin %d is infinite", frame, k)
		}

		if power < 0 {
			t.Fatalf("frame %d suppressed power bin %d = %v, want non-negative", frame, k, power)
		}
	}
}

// assertPCMFinite verifies reconstructed PCM samples contain no invalid
// floating point values
func assertPCMFinite(t *testing.T, frame int, samples []float32) {
	t.Helper()

	for i, sample := range samples {
		value := float64(sample)

		if math.IsNaN(value) {
			t.Fatalf("frame %d reconstructed sample %d is NaN", frame, i)
		}

		if math.IsInf(value, 0) {
			t.Fatalf("frame %d reconstructed sample %d is infinite", frame, i)
		}
	}
}

// writeLogMMSECSVFrame writes per-frequency-bin diagnostics for one STFT
// frame
//
// Post-baseline rows contain the complete Log-MMSE diagnostic chain:
//
//	observed power
//	MCRA noise PSD
//	effective noise PSD
//	a-posteriori SNR gamma
//	a-priori SNR xi
//	applied spectral gain
//	output power
//	attenuation in dB
//
// Baseline frames are marked as baseline and use a gain of one because
// Log-MMSE suppression is intentionally disabled during baseline learning
func writeLogMMSECSVFrame(
	t *testing.T,
	writer *csv.Writer,
	frame int,
	frameStartSample int,
	sampleRate int,
	power []float32,
	noisePSD []float32,
	effectiveNoisePSD []float32,
	gamma []float32,
	xi []float32,
	outputSpectrum audio.Spectrum,
	isBaseline bool,
	floor float32,
) {
	t.Helper()

	timeSeconds := float64(frameStartSample) / float64(sampleRate)
	binWidth := float64(sampleRate) / float64(logMMSEFFTSize)

	phase := "suppressed"

	if isBaseline {
		phase = "baseline"
	}

	for k := range power {
		frequency := float64(k) * binWidth
		effectiveNoise := float32(0)
		gammaValue := float32(0)
		xiValue := float32(0)
		gain := float32(1)
		outputPower := power[k]

		if !isBaseline {
			if k < len(effectiveNoisePSD) {
				effectiveNoise = effectiveNoisePSD[k]
			}

			if k < len(gamma) {
				gammaValue = gamma[k]
			}

			if k < len(xi) {
				xiValue = xi[k]
			}

			if k < len(outputSpectrum.Power) {
				outputPower = outputSpectrum.Power[k]
			}

			// Log-MMSE applies gain to spectral amplitude, therefore:
			//
			//	outputPower = gain² * inputPower
			//
			// so the actual applied gain can be recovered from the
			// input and output powers
			if power[k] > floor {
				ratio := outputPower / power[k]

				if ratio < 0 {
					ratio = 0
				}

				gain = float32(math.Sqrt(float64(ratio)))
			} else {
				gain = 0
			}
		}

		attenuationDB := float64(0)

		if !isBaseline && gain > 0 {
			attenuationDB = 20 * math.Log10(float64(gain))
		}

		record := []string{
			formatLogMMSECSVFloat(timeSeconds),
			strconv.Itoa(frame),
			phase,
			strconv.Itoa(k),
			formatLogMMSECSVFloat(frequency),
			formatLogMMSECSVFloat(float64(power[k])),
			formatLogMMSECSVFloat(float64(noisePSD[k])),
			formatLogMMSECSVFloat(float64(effectiveNoise)),
			formatLogMMSECSVFloat(float64(gammaValue)),
			formatLogMMSECSVFloat(float64(xiValue)),
			formatLogMMSECSVFloat(float64(gain)),
			formatLogMMSECSVFloat(float64(outputPower)),
			formatLogMMSECSVFloat(attenuationDB),
		}

		if err := writer.Write(record); err != nil {
			t.Fatalf("write Log-MMSE CSV frame %d bin %d: %v", frame, k, err)
		}
	}
}

// formatLogMMSECSVFloat formats floating-point diagnostics using enough
// precision for spectral analysis while keeping the CSV readable
func formatLogMMSECSVFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 8, 64)
}
