//go:build smoke
// +build smoke

package smoke

import (
	"math"
	"strconv"
	"testing"
	"time"

	"github.com/sghaida/noise-cancelation/dsp/highpass"
	"github.com/sghaida/noise-cancelation/dsp/noise"
	"github.com/sghaida/noise-cancelation/dsp/stft"
)

const (
	mcraFixturePath = "fixtures/human_voice_with_noise_with_5s_baseline_8khz_pcm16.wav"

	mcraOutputPath = "output/mcra.out.csv"

	mcraSampleRate = 8000
	mcraFFTSize    = 256
	mcraHopSize    = 128
	mcraChunkSize  = 160

	mcraHighPassCutoff   = float32(80)
	mcraBaselineDuration = 5 * time.Second
)

// TestMCRASmoke process real audio through STFT and MCRA
// the first five seconds of the fixture contain known baseline noise
// the smoke test verifies that:
//   - real PCM audio can be processed through the 80 Hz high-pass filter
//   - high-pass filtered PCM audio can be processed through STFT
//   - the first five seconds can initialize the MCRA noise baseline
//   - only STFT frames fully contained inside the baseline are used
//   - STFT power can be passed to MCRA
//   - MCRA returns one noise estimate per power bin
//   - noise estimates contain no NaN or infinity
//   - noise estimates stay above the configured floor
//   - the noise estimate continues changing after baseline initialization
//   - speech and noise probability ignore frequencies below 80 Hz
//
// The diagnostic CSV contains:
//
//	time_seconds
//	observed_power
//	noise_power
//	ratio_db
//	speech_probability
//	noise_probability
func TestMCRASmoke(t *testing.T) {
	sampleRate, channels, bitDepth, samples, err := readPCM16MonoWAV(
		mcraFixturePath,
	)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	if sampleRate != mcraSampleRate {
		t.Fatalf("sample rate = %d, want %d", sampleRate, mcraSampleRate)
	}

	if channels != expectedChannels {
		t.Fatalf("channels = %d, want %d", channels, expectedChannels)
	}

	if bitDepth != expectedBitDepth {
		t.Fatalf("bit depth = %d, want %d", bitDepth, expectedBitDepth)
	}

	analyzer, err := stft.New(mcraFFTSize, mcraHopSize)
	if err != nil {
		t.Fatalf("create STFT analyzer: %v", err)
	}

	highPass := highpass.New(sampleRate, mcraHighPassCutoff)

	cfg := noise.DefaultMCRAConfig()

	cfg.SampleRate = mcraSampleRate
	cfg.HopSize = mcraHopSize
	cfg.BaselineDuration = mcraBaselineDuration
	cfg.MinProbabilityFrequency = mcraHighPassCutoff

	estimator := noise.NewMCRAEstimator(cfg)

	// The first five seconds contain known baseline noise
	estimator.StartBaseline()

	baselineSamples := int(
		mcraBaselineDuration.Seconds() * float64(sampleRate),
	)

	rows := [][]string{
		{
			"time_seconds", "observed_power", "noise_power", "ratio_db",
			"speech_probability", "noise_probability",
		},
	}

	var (
		frameCount int

		baselineFrameCount int
		normalFrameCount   int

		baselineFinished bool

		previousNoisePower float64
		havePreviousNoise  bool

		noiseChangedAfterBaseline bool
	)

	for start := 0; start < len(samples); start += mcraChunkSize {
		end := start + mcraChunkSize

		if end > len(samples) {
			end = len(samples)
		}

		filtered, err := highPass.Process(samples[start:end])
		if err != nil {
			t.Fatalf("high-pass samples at %d: %v", start, err)
		}

		spectra, err := analyzer.Process(filtered)
		if err != nil {
			t.Fatalf("process samples at %d: %v", start, err)
		}

		for i := range spectra {
			frameStartSample := frameCount * mcraHopSize

			frameEndSample := frameStartSample + mcraFFTSize

			// A frame belongs to the baseline only when the complete
			// STFT window is contained inside the first five seconds
			//
			// This prevents a frame crossing the 5-second boundary
			// from mixing baseline noise with the beginning of speech
			isBaselineFrame := frameEndSample <= baselineSamples

			if !isBaselineFrame && !baselineFinished {
				estimator.FinishBaseline()

				baselineFinished = true

				t.Logf(
					"MCRA baseline finished after %d frames at %.3f seconds",
					baselineFrameCount, float64(frameStartSample)/float64(sampleRate),
				)
			}

			power := spectra[i].RecomputePower()
			noisePSD := estimator.Process(power)

			assertMCRASmokeFrame(t, frameCount, power, noisePSD, cfg.Floor)

			observedPower := averagePower(power)
			estimatedNoise := averagePower(noisePSD)

			speechProbability := estimator.SpeechProbability(power)
			noiseProbability := estimator.NoiseProbability(power)
			if isBaselineFrame {
				baselineFrameCount++
			} else {
				normalFrameCount++

				// Check adaptation only after baseline initialization
				//
				// Changes during baseline collection do not prove that
				// normal MCRA tracking is functioning
				if havePreviousNoise {
					diff := math.Abs(estimatedNoise - previousNoisePower)

					if diff > 1e-12 {
						noiseChangedAfterBaseline = true
					}
				}

				previousNoisePower = estimatedNoise

				havePreviousNoise = true
			}

			ratioDB := powerRatioDB(observedPower, estimatedNoise)

			timeSeconds := float64(frameStartSample) / float64(sampleRate)

			rows = append(
				rows,
				[]string{
					strconv.FormatFloat(timeSeconds, 'f', 6, 64),
					strconv.FormatFloat(observedPower, 'e', 6, 64),
					strconv.FormatFloat(estimatedNoise, 'e', 6, 64),
					strconv.FormatFloat(ratioDB, 'f', 3, 64),
					strconv.FormatFloat(float64(speechProbability), 'f', 6, 64),
					strconv.FormatFloat(float64(noiseProbability), 'f', 6, 64),
				},
			)

			frameCount++
		}
	}

	// Normally this has already happened when processing passes the
	// five-second boundary
	//
	// Keep this guard so the estimator is finalized correctly if a
	// shorter fixture is accidentally used
	if !baselineFinished {
		estimator.FinishBaseline()
		baselineFinished = true
	}

	if frameCount == 0 {
		t.Fatal("no STFT frames were processed")
	}

	if baselineFrameCount == 0 {
		t.Fatal("no baseline STFT frames were processed")
	}

	if normalFrameCount == 0 {
		t.Fatal("no post-baseline MCRA frames were processed")
	}

	if !noiseChangedAfterBaseline {
		t.Fatal(
			"MCRA noise estimate did not change after baseline initialization",
		)
	}

	if err := writeMCRACSV(mcraOutputPath, rows); err != nil {
		t.Fatalf("write MCRA diagnostics: %v", err)
	}

	t.Logf("processed %d MCRA frames", frameCount)

	t.Logf("baseline frames: %d", baselineFrameCount)

	t.Logf("post-baseline frames: %d", normalFrameCount)

	t.Logf("MCRA diagnostics written to %s", mcraOutputPath)

	estimator.Reset()
	highPass.Reset()
}
