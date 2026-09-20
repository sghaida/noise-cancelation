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
	"github.com/sghaida/noise-cancelation/dsp/interference"
	"github.com/sghaida/noise-cancelation/dsp/noise"
	"github.com/sghaida/noise-cancelation/dsp/snr"
	"github.com/sghaida/noise-cancelation/dsp/stft"
	"github.com/sghaida/noise-cancelation/dsp/suppressor"
)

const (
	tonalFixturePath   = "fixtures/natural_voice_office_noise_8khz.wav"
	tonalOutputPath    = "output/tonal_transient.out.wav"
	tonalCSVOutputPath = "output/tonal_transient.out.csv"

	tonalSampleRate = 8000

	tonalFFTSize   = 256
	tonalHopSize   = 128
	tonalChunkSize = 160

	tonalHighPassCutoff = float32(80)

	tonalBaselineDuration = 5 * time.Second
)

// TestTonalTransientPipelineSmoke processes real audio through SPP MMSE, Log MMSE and tonal transient suppression
//
// The pipeline is:
//
//	PCM
//	 ↓
//	80 Hz high pass
//	 ↓
//	STFT
//	 ↓
//	P[t,k] = |X[t,k]|²
//	 ↓
//	SPP MMSE
//	 ↓
//	N[t,k]
//	 ↓
//	Decision directed SNR
//	 ↓
//	Log MMSE
//	 ↓
//	Tonal transient detector
//	 ↓
//	Ginterference[t,k]
//	 ↓
//	ISTFT
//
// The final spectral gain is:
//
//	Gfinal[t,k] = Glogmmse[t,k] * Ginterference[t,k]
//
// The CSV contains one row for every frequency bin and exposes:
//
//	P[t,k]
//	N[t,k]
//	SPP[t,k]
//	γ[t,k]
//	ξ[t,k]
//	Glogmmse[t,k]
//	tonal prominence
//	spectral flux
//	harmonic support
//	movement score
//	interference score
//	Ginterference[t,k]
//	Gfinal[t,k]
func TestTonalTransientPipelineSmoke(t *testing.T) {
	sampleRate, channels, bitDepth, samples, err := readPCM16MonoWAV(tonalFixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	if sampleRate != tonalSampleRate {
		t.Fatalf("sample rate = %d, want %d", sampleRate, tonalSampleRate)
	}

	if channels != expectedChannels {
		t.Fatalf("channels = %d, want %d", channels, expectedChannels)
	}

	if bitDepth != expectedBitDepth {
		t.Fatalf("bit depth = %d, want %d", bitDepth, expectedBitDepth)
	}

	highPass := highpass.New(sampleRate, tonalHighPassCutoff)

	analyzer, err := stft.New(tonalFFTSize, tonalHopSize)
	if err != nil {
		t.Fatalf("create STFT analyzer: %v", err)
	}

	synthesizer, err := stft.NewSynthesizer(tonalFFTSize, tonalHopSize)
	if err != nil {
		t.Fatalf("create ISTFT synthesizer: %v", err)
	}

	sppConfig := noise.DefaultSPPMMSEConfig()
	noiseEstimator := noise.NewSPPMMSEEstimator(sppConfig)

	noiseEstimator.StartBaseline()

	logMMSEConfig := suppressor.DefaultLogMMSEConfig()
	snrConfig := snr.DefaultDecisionDirectedConfig()

	logMMSE := suppressor.NewLogMMSE(logMMSEConfig, snrConfig)

	// diagnosticSNR mirrors the internal Log MMSE SNR estimator only for CSV diagnostics
	diagnosticSNR := snr.NewDecisionDirected(snrConfig)

	tonalConfig := interference.DefaultTonalTransientConfig()
	tonalDetector := interference.NewTonalTransientDetector(tonalConfig)

	baselineSamples := int(tonalBaselineDuration.Seconds() * float64(sampleRate))

	output := make([]float32, 0, len(samples))

	if err := os.MkdirAll(filepath.Dir(tonalCSVOutputPath), 0o755); err != nil {
		t.Fatalf("create output directory: %v", err)
	}

	csvFile, err := os.Create(tonalCSVOutputPath)
	if err != nil {
		t.Fatalf("create tonal transient CSV: %v", err)
	}

	defer func() {
		if err := csvFile.Close(); err != nil {
			t.Errorf("close tonal transient CSV: %v", err)
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
		"spp_noise_power",
		"speech_presence_probability",
		"effective_noise_power",
		"gamma",
		"xi",
		"logmmse_gain",
		"tonal_prominence_db",
		"spectral_flux_db",
		"harmonic_support",
		"movement_score",
		"interference_score",
		"interference_gain",
		"final_gain",
		"output_power",
		"attenuation_db",
	}

	if err := csvWriter.Write(header); err != nil {
		t.Fatalf("write tonal transient CSV header: %v", err)
	}

	var (
		frameCount           int
		baselineFrameCount   int
		suppressedFrameCount int
		baselineFinished     bool

		diagnosticNoisePSD []float32
		logMMSEGain        []float32
	)

	processSpectrum := func(spectrum audio.Spectrum) {
		t.Helper()

		frameStartSample := frameCount * tonalHopSize
		frameEndSample := frameStartSample + tonalFFTSize

		isBaselineFrame := frameEndSample <= baselineSamples

		if !isBaselineFrame && !baselineFinished {
			noiseEstimator.FinishBaseline()

			// Start temporal foreground detection only after the known baseline period
			tonalDetector.Reset()
			diagnosticSNR.Reset()

			baselineFinished = true

			t.Logf(
				"baseline finished after %d frames at %.3f seconds",
				baselineFrameCount,
				float64(frameStartSample)/float64(sampleRate),
			)
		}

		power := spectrum.RecomputePower()

		noisePSD := noiseEstimator.Process(power)

		assertPipelineSpectrum(t, frameCount, power, noisePSD, sppConfig.Floor)

		speechProbability := noiseEstimator.SpeechPresenceProbability()

		outputSpectrum := spectrum

		var (
			gamma            []float32
			xi               []float32
			interferenceData interference.Result
		)

		if isBaselineFrame {
			baselineFrameCount++
		} else {
			if len(diagnosticNoisePSD) != len(noisePSD) {
				diagnosticNoisePSD = make([]float32, len(noisePSD))
			}

			if len(logMMSEGain) != len(power) {
				logMMSEGain = make([]float32, len(power))
			}

			// Reproduce the same effective noise PSD used internally by Log MMSE
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

			// Detect foreground interference from the original spectrum so Log MMSE suppression does not hide
			// tonal, transient or movement evidence from the detector
			interferenceData = tonalDetector.Process(power)

			outputSpectrum, err = logMMSE.Process(spectrum, noisePSD)
			if err != nil {
				t.Fatalf("process Log MMSE frame %d: %v", frameCount, err)
			}

			assertSuppressedSpectrum(t, frameCount, outputSpectrum)

			// Recover the actual Log MMSE amplitude gain before applying the tonal interference gain
			for k := range power {
				logMMSEGain[k] = calculateSmokeGain(power[k], outputSpectrum.Power[k], sppConfig.Floor)
			}

			// Keep the diagnostic estimator synchronized with the real Log MMSE estimator
			//
			// This must happen before tonal attenuation because the real Log MMSE estimator stores its own
			// clean power before the tonal detector modifies the spectrum
			if err := diagnosticSNR.Update(outputSpectrum.Power, diagnosticNoisePSD); err != nil {
				t.Fatalf("update diagnostic SNR frame %d: %v", frameCount, err)
			}

			if err := interference.ApplyGain(&outputSpectrum, interferenceData.Gain); err != nil {
				t.Fatalf("apply tonal interference gain frame %d: %v", frameCount, err)
			}

			assertSuppressedSpectrum(t, frameCount, outputSpectrum)

			suppressedFrameCount++
		}

		writeTonalTransientCSVFrame(
			t,
			csvWriter,
			frameCount,
			frameStartSample,
			sampleRate,
			power,
			noisePSD,
			speechProbability,
			diagnosticNoisePSD,
			gamma,
			xi,
			logMMSEGain,
			interferenceData,
			outputSpectrum,
			isBaselineFrame,
			sppConfig.Floor,
		)

		reconstructed, err := synthesizer.Process(outputSpectrum)
		if err != nil {
			t.Fatalf("process ISTFT frame %d: %v", frameCount, err)
		}

		assertPCMFinite(t, frameCount, reconstructed)

		output = append(output, reconstructed...)

		frameCount++
	}

	// Feed the pipeline using the same 160 sample chunk size used by an 8 kHz Twilio media stream
	for start := 0; start < len(samples); start += tonalChunkSize {
		end := start + tonalChunkSize

		if end > len(samples) {
			end = len(samples)
		}

		filtered, err := highPass.Process(samples[start:end])
		if err != nil {
			t.Fatalf("high pass samples at %d: %v", start, err)
		}

		spectra, err := analyzer.Process(filtered)
		if err != nil {
			t.Fatalf("process STFT samples at %d: %v", start, err)
		}

		for i := range spectra {
			processSpectrum(spectra[i])
		}
	}

	finalSpectra, err := analyzer.Flush()
	if err != nil {
		t.Fatalf("flush STFT analyzer: %v", err)
	}

	for i := range finalSpectra {
		processSpectrum(finalSpectra[i])
	}

	if !baselineFinished {
		noiseEstimator.FinishBaseline()
		baselineFinished = true
	}

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
		t.Fatal("no suppressed frames were processed")
	}

	if len(output) > len(samples) {
		output = output[:len(samples)]
	}

	if len(output) == 0 {
		t.Fatal("no reconstructed PCM samples were produced")
	}

	if len(output) != len(samples) {
		t.Fatalf("output contains %d samples, want %d", len(output), len(samples))
	}

	if err := writePCM16MonoWAV(tonalOutputPath, sampleRate, output); err != nil {
		t.Fatalf("write tonal transient output: %v", err)
	}

	csvWriter.Flush()

	if err := csvWriter.Error(); err != nil {
		t.Fatalf("flush tonal transient CSV: %v", err)
	}

	t.Logf("processed %d STFT frames", frameCount)
	t.Logf("baseline frames: %d", baselineFrameCount)
	t.Logf("suppressed frames: %d", suppressedFrameCount)
	t.Logf("input samples: %d", len(samples))
	t.Logf("output samples: %d", len(output))
	t.Logf("tonal transient WAV written to %s", tonalOutputPath)
	t.Logf("tonal transient CSV written to %s", tonalCSVOutputPath)

	tonalDetector.Reset()
	logMMSE.Reset()
	diagnosticSNR.Reset()
	noiseEstimator.Reset()
	synthesizer.Reset()
	analyzer.Reset()
	highPass.Reset()
}

// writeTonalTransientCSVFrame writes per frequency bin diagnostics for one STFT frame
//
// The final gain is:
//
//	Gfinal[t,k] = Glogmmse[t,k] * Ginterference[t,k]
//
// The attenuation is:
//
//	A[t,k] = 20 * log10(Gfinal[t,k])
func writeTonalTransientCSVFrame(
	t *testing.T,
	writer *csv.Writer,
	frame int,
	frameStartSample int,
	sampleRate int,
	power []float32,
	noisePSD []float32,
	speechProbability []float32,
	effectiveNoisePSD []float32,
	gamma []float32,
	xi []float32,
	logMMSEGain []float32,
	interferenceData interference.Result,
	outputSpectrum audio.Spectrum,
	isBaseline bool,
	floor float32,
) {
	t.Helper()

	timeSeconds := float64(frameStartSample) / float64(sampleRate)
	binWidth := float64(sampleRate) / float64(tonalFFTSize)

	phase := "suppressed"

	if isBaseline {
		phase = "baseline"
	}

	for k := range power {
		frequency := float64(k) * binWidth

		spp := float32(0)
		effectiveNoise := float32(0)
		gammaValue := float32(0)
		xiValue := float32(0)

		logGain := float32(1)
		tonalProminenceDB := float32(0)
		spectralFluxDB := float32(0)
		harmonicSupport := float32(0)
		movementScore := float32(0)
		interferenceScore := float32(0)
		interferenceGain := float32(1)

		outputPower := power[k]
		finalGain := float32(1)

		if k < len(speechProbability) {
			spp = speechProbability[k]
		}

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

			if k < len(logMMSEGain) {
				logGain = logMMSEGain[k]
			}

			if k < len(interferenceData.TonalProminenceDB) {
				tonalProminenceDB = interferenceData.TonalProminenceDB[k]
			}

			if k < len(interferenceData.SpectralFluxDB) {
				spectralFluxDB = interferenceData.SpectralFluxDB[k]
			}

			if k < len(interferenceData.HarmonicSupport) {
				harmonicSupport = interferenceData.HarmonicSupport[k]
			}

			if k < len(interferenceData.MovementScore) {
				movementScore = interferenceData.MovementScore[k]
			}

			if k < len(interferenceData.Score) {
				interferenceScore = interferenceData.Score[k]
			}

			if k < len(interferenceData.Gain) {
				interferenceGain = interferenceData.Gain[k]
			}

			if k < len(outputSpectrum.Power) {
				outputPower = outputSpectrum.Power[k]
			}

			finalGain = calculateSmokeGain(power[k], outputPower, floor)
		}

		attenuationDB := float64(0)

		if !isBaseline && finalGain > 0 {
			attenuationDB = 20 * math.Log10(float64(finalGain))
		}

		record := []string{
			formatTonalCSVFloat(timeSeconds),
			strconv.Itoa(frame),
			phase,
			strconv.Itoa(k),
			formatTonalCSVFloat(frequency),
			formatTonalCSVFloat(float64(power[k])),
			formatTonalCSVFloat(float64(noisePSD[k])),
			formatTonalCSVFloat(float64(spp)),
			formatTonalCSVFloat(float64(effectiveNoise)),
			formatTonalCSVFloat(float64(gammaValue)),
			formatTonalCSVFloat(float64(xiValue)),
			formatTonalCSVFloat(float64(logGain)),
			formatTonalCSVFloat(float64(tonalProminenceDB)),
			formatTonalCSVFloat(float64(spectralFluxDB)),
			formatTonalCSVFloat(float64(harmonicSupport)),
			formatTonalCSVFloat(float64(movementScore)),
			formatTonalCSVFloat(float64(interferenceScore)),
			formatTonalCSVFloat(float64(interferenceGain)),
			formatTonalCSVFloat(float64(finalGain)),
			formatTonalCSVFloat(float64(outputPower)),
			formatTonalCSVFloat(attenuationDB),
		}

		if err := writer.Write(record); err != nil {
			t.Fatalf("write tonal transient CSV frame %d bin %d: %v", frame, k, err)
		}
	}
}

// calculateSmokeGain calculates amplitude gain from input and output spectral power
//
//	Pout[k] = G[k]² * Pin[k]
//
// therefore:
//
//	G[k] = sqrt(Pout[k] / Pin[k])
func calculateSmokeGain(inputPower float32, outputPower float32, floor float32) float32 {
	if inputPower <= floor {
		return 0
	}

	ratio := outputPower / inputPower

	if ratio < 0 {
		ratio = 0
	}

	return float32(math.Sqrt(float64(ratio)))
}

// formatTonalCSVFloat formats floating point diagnostics for CSV analysis
func formatTonalCSVFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 8, 64)
}
