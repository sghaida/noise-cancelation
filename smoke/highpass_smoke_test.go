//go:build smoke
// +build smoke

package smoke

import (
	"math"
	"testing"

	"github.com/sghaida/noise-cancelation/dsp"
	"github.com/sghaida/noise-cancelation/dsp/highpass"
)

const (
	highPassOutputPath = "output/highpass.out.wav"

	cutoffHz       = float32(80)
	lowBandSeconds = 10
)

// TestHighPassSmoke processes the real 30-second speech fixture through
// the high-pass filter using Twilio-sized 20 ms streaming frames
func TestHighPassSmoke(t *testing.T) {
	sampleRate, samples, durationSeconds := readTestFixture(t)

	original := append([]float32(nil), samples...)

	var processor dsp.Processor = highpass.New(sampleRate, cutoffHz)

	frameSize := sampleRate * frameDuration / 1000

	for start := 0; start < len(samples); start += frameSize {
		end := start + frameSize

		if end > len(samples) {
			end = len(samples)
		}

		frame := samples[start:end]

		processedFrame, err := processor.Process(frame)
		if err != nil {
			t.Fatalf(
				"process frame starting at sample %d: %v",
				start, err,
			)
		}

		if len(processedFrame) != len(frame) {
			t.Fatalf(
				"processed frame length = %d, want %d",
				len(processedFrame), len(frame),
			)
		}

		copy(samples[start:end], processedFrame)
	}

	assertFiniteSamples(t, samples)
	assertSignalChanged(t, original, samples)

	beforePower := bandPower(
		original, sampleRate, 0, lowBandSeconds,
		[]float64{20, 30, 40, 50, 60, 70},
	)

	afterPower := bandPower(
		samples, sampleRate, 0, lowBandSeconds,
		[]float64{20, 30, 40, 50, 60, 70},
	)

	if beforePower <= 0 {
		t.Fatal("fixture contains no measurable low-frequency energy")
	}

	attenuationRatio := afterPower / beforePower

	const maximumLowBandRatio = 0.60

	if attenuationRatio >= maximumLowBandRatio {
		t.Fatalf(
			"low-frequency power ratio = %.4f, want less than %.4f",
			attenuationRatio, maximumLowBandRatio,
		)
	}

	attenuationDB := 10 * math.Log10(attenuationRatio)

	if err := writePCM16MonoWAV(highPassOutputPath, sampleRate, samples); err != nil {
		t.Fatalf("write high-pass output: %v", err)
	}

	t.Logf(
		"processed %.2f seconds in %d-sample frames",
		durationSeconds, frameSize,
	)

	t.Logf(
		"20-70 Hz power reduced by %.2f dB with ratio %.4f",
		attenuationDB, attenuationRatio,
	)

	t.Logf(
		"high-pass output written to %s",
		highPassOutputPath,
	)

	processor.Reset()
}

// assertSignalChanged verifies that the output differs from the input
func assertSignalChanged(
	t *testing.T,
	before []float32,
	after []float32,
) {
	t.Helper()

	if len(before) != len(after) {
		t.Fatalf(
			"sample count changed from %d to %d",
			len(before), len(after),
		)
	}

	var absoluteDifference float64

	for i := range before {
		absoluteDifference += math.Abs(float64(after[i] - before[i]))
	}

	const minimumDifference = 1e-6

	if absoluteDifference <= minimumDifference {
		t.Fatal("processed signal is identical to input")
	}
}

// bandPower estimates energy at selected frequencies using direct
// correlation against sine and cosine waves
func bandPower(
	samples []float32,
	sampleRate int,
	startSeconds int,
	durationSeconds int,
	frequencies []float64,
) float64 {
	start := startSeconds * sampleRate
	end := start + durationSeconds*sampleRate

	if start < 0 {
		start = 0
	}

	if end > len(samples) {
		end = len(samples)
	}

	if start >= end {
		return 0
	}

	segment := samples[start:end]

	if len(segment) < 2 {
		return 0
	}

	var totalPower float64

	for _, frequency := range frequencies {
		var (
			realPart      float64
			imaginaryPart float64
		)

		for i, sample := range segment {
			window := 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(len(segment)-1))

			angle := 2 * math.Pi * frequency
			angle *= float64(i) / float64(sampleRate)

			windowedSample := float64(sample) * window

			realPart += windowedSample * math.Cos(angle)
			imaginaryPart -= windowedSample * math.Sin(angle)
		}

		totalPower += realPart*realPart + imaginaryPart*imaginaryPart
	}

	return totalPower
}
