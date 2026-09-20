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
	sppMMSEFixturePath   = "fixtures/natural_voice_office_noise_8khz.wav"
	sppMMSEOutputPath    = "output/sppmmse.out.wav"
	sppMMSECSVOutputPath = "output/sppmmse.out.csv"

	sppMMSESampleRate = 8000

	sppMMSEFFTSize   = 256
	sppMMSEHopSize   = 128
	sppMMSEChunkSize = 160

	sppMMSEHighPassCutoff = float32(80)

	sppMMSEBaselineDuration = 5 * time.Second
)

// TestSPPMMSEPipelineSmoke processes real audio through the complete SPP MMSE and Log MMSE pipeline
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
//	SPP MMSE noise estimation
//	 ↓
//	N[t,k]
//	 ↓
//	Decision directed SNR
//	 ↓
//	γ[t,k] = P[t,k] / N[t,k]
//	 ↓
//	ξ[t,k]
//	 ↓
//	Log MMSE
//	 ↓
//	Y[t,k] = G[t,k] * X[t,k]
//	 ↓
//	ISTFT
//	 ↓
//	reconstructed PCM
//
// first five seconds are treated as a known noise only baseline
//
// generated CSV contains one row per STFT frequency bin and exposes the values required to analyse the
// SPP MMSE noise estimate and Log MMSE suppression behaviour
func TestSPPMMSEPipelineSmoke(t *testing.T) {
	sampleRate, channels, bitDepth, samples, err := readPCM16MonoWAV(sppMMSEFixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	if sampleRate != sppMMSESampleRate {
		t.Fatalf("sample rate = %d, want %d", sampleRate, sppMMSESampleRate)
	}

	if channels != expectedChannels {
		t.Fatalf("channels = %d, want %d", channels, expectedChannels)
	}

	if bitDepth != expectedBitDepth {
		t.Fatalf("bit depth = %d, want %d", bitDepth, expectedBitDepth)
	}

	highPass := highpass.New(sampleRate, sppMMSEHighPassCutoff)

	analyzer, err := stft.New(sppMMSEFFTSize, sppMMSEHopSize)
	if err != nil {
		t.Fatalf("create STFT analyzer: %v", err)
	}

	synthesizer, err := stft.NewSynthesizer(sppMMSEFFTSize, sppMMSEHopSize)
	if err != nil {
		t.Fatalf("create ISTFT synthesizer: %v", err)
	}

	sppConfig := noise.DefaultSPPMMSEConfig()
	noiseEstimator := noise.NewSPPMMSEEstimator(sppConfig)

	noiseEstimator.StartBaseline()

	logMMSEConfig := suppressor.DefaultLogMMSEConfig()
	snrConfig := snr.DefaultDecisionDirectedConfig()

	logMMSE := suppressor.NewLogMMSE(logMMSEConfig, snrConfig)

	// estimator mirrors the internal Log MMSE SNR estimator only for CSV diagnostics
	diagnosticSNR := snr.NewDecisionDirected(snrConfig)

	baselineSamples := int(sppMMSEBaselineDuration.Seconds() * float64(sampleRate))

	output := make([]float32, 0, len(samples))

	if err := os.MkdirAll(filepath.Dir(sppMMSECSVOutputPath), 0o755); err != nil {
		t.Fatalf("create output directory: %v", err)
	}

	csvFile, err := os.Create(sppMMSECSVOutputPath)
	if err != nil {
		t.Fatalf("create SPP MMSE CSV: %v", err)
	}

	defer func() {
		if err := csvFile.Close(); err != nil {
			t.Errorf("close SPP MMSE CSV: %v", err)
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
		"gain",
		"output_power",
		"attenuation_db",
	}

	if err := csvWriter.Write(header); err != nil {
		t.Fatalf("write SPP MMSE CSV header: %v", err)
	}

	var (
		frameCount           int
		baselineFrameCount   int
		suppressedFrameCount int
		baselineFinished     bool
		diagnosticNoisePSD   []float32
	)

	processSpectrum := func(spectrum audio.Spectrum) {
		t.Helper()

		frameStartSample := frameCount * sppMMSEHopSize
		frameEndSample := frameStartSample + sppMMSEFFTSize

		isBaselineFrame := frameEndSample <= baselineSamples

		if !isBaselineFrame && !baselineFinished {
			noiseEstimator.FinishBaseline()
			baselineFinished = true

			t.Logf(
				"SPP MMSE baseline finished after %d frames at %.3f seconds",
				baselineFrameCount, float64(frameStartSample)/float64(sampleRate),
			)
		}

		power := spectrum.RecomputePower()
		noisePSD := noiseEstimator.Process(power)

		assertPipelineSpectrum(t, frameCount, power, noisePSD, sppConfig.Floor)

		speechProbability := noiseEstimator.SpeechPresenceProbability()
		outputSpectrum := spectrum

		var gamma []float32
		var xi []float32

		if isBaselineFrame {
			baselineFrameCount++
		} else {
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
				t.Fatalf("process Log MMSE frame %d: %v", frameCount, err)
			}

			assertSuppressedSpectrum(t, frameCount, outputSpectrum)

			if err := diagnosticSNR.Update(outputSpectrum.Power, diagnosticNoisePSD); err != nil {
				t.Fatalf("update diagnostic SNR frame %d: %v", frameCount, err)
			}

			suppressedFrameCount++
		}

		writeSPPMMSECSVFrame(
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

	// feed the pipeline using the same 160 sample chunk size used by an 8 kHz Twilio media stream
	for start := 0; start < len(samples); start += sppMMSEChunkSize {
		end := start + sppMMSEChunkSize

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
		t.Fatal("no Log MMSE frames were processed")
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

	if err := writePCM16MonoWAV(sppMMSEOutputPath, sampleRate, output); err != nil {
		t.Fatalf("write SPP MMSE output: %v", err)
	}

	csvWriter.Flush()

	if err := csvWriter.Error(); err != nil {
		t.Fatalf("flush SPP MMSE CSV: %v", err)
	}

	t.Logf("processed %d STFT frames", frameCount)
	t.Logf("baseline frames: %d", baselineFrameCount)
	t.Logf("Log MMSE suppressed frames: %d", suppressedFrameCount)
	t.Logf("input samples: %d", len(samples))
	t.Logf("output samples: %d", len(output))
	t.Logf("SPP MMSE output written to %s", sppMMSEOutputPath)
	t.Logf("SPP MMSE diagnostics written to %s", sppMMSECSVOutputPath)

	logMMSE.Reset()
	diagnosticSNR.Reset()
	noiseEstimator.Reset()
	synthesizer.Reset()
	analyzer.Reset()
	highPass.Reset()
}

// writeSPPMMSECSVFrame write one CSV record for every frequency bin in an STFT frame
//
// The output records:
//
//	P[t,k] observed spectral power
//	N[t,k] SPP MMSE noise PSD
//	p[t,k] speech presence probability
//	Ne[t,k] effective noise PSD after Log MMSE noise overestimation
//	γ[t,k] a posteriori SNR
//	ξ[t,k] a priori SNR
//	G[t,k] Log MMSE gain
//
// Attenuation is calculated as:
//
//	A[t,k] = 20 * log10(G[t,k])
func writeSPPMMSECSVFrame(
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
	outputSpectrum audio.Spectrum,
	isBaseline bool,
	floor float32,
) {
	t.Helper()

	timeSeconds := float64(frameStartSample) / float64(sampleRate)
	binWidth := float64(sampleRate) / float64(sppMMSEFFTSize)

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
		gain := float32(1)
		outputPower := power[k]

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

			if k < len(outputSpectrum.Power) {
				outputPower = outputSpectrum.Power[k]
			}

			// Since outputPower = gain² * inputPower the applied amplitude gain is sqrt(outputPower / inputPower)
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
			formatSPPMMSECSVFloat(timeSeconds),
			strconv.Itoa(frame),
			phase,
			strconv.Itoa(k),
			formatSPPMMSECSVFloat(frequency),
			formatSPPMMSECSVFloat(float64(power[k])),
			formatSPPMMSECSVFloat(float64(noisePSD[k])),
			formatSPPMMSECSVFloat(float64(spp)),
			formatSPPMMSECSVFloat(float64(effectiveNoise)),
			formatSPPMMSECSVFloat(float64(gammaValue)),
			formatSPPMMSECSVFloat(float64(xiValue)),
			formatSPPMMSECSVFloat(float64(gain)),
			formatSPPMMSECSVFloat(float64(outputPower)),
			formatSPPMMSECSVFloat(attenuationDB),
		}

		if err := writer.Write(record); err != nil {
			t.Fatalf("write SPP MMSE CSV frame %d bin %d: %v", frame, k, err)
		}
	}
}

// formatSPPMMSECSVFloat formats diagnostic floating point values with enough precision for later analysis
func formatSPPMMSECSVFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 8, 64)
}
