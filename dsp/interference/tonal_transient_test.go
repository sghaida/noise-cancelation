package interference

import (
	"math"
	"testing"

	"github.com/sghaida/noise-cancelation/audio"
)

const tonalTransientTestTolerance = 1e-5

func TestNewTonalTransientDetectorDefaults(t *testing.T) {
	defaults := DefaultTonalTransientConfig()

	detector := NewTonalTransientDetector(
		TonalTransientConfig{
			LocalRadius:              0,
			TonalStartDB:             -1,
			TonalFullDB:              0,
			FluxStartDB:              -1,
			FluxFullDB:               0,
			MovementSearchRadius:     0,
			MovementStartBins:        0,
			MovementFullBins:         0,
			MovementMinRelativePower: 2,
			HarmonicToleranceBins:    -1,
			HarmonicRelativePower:    2,
			Strength:                 -1,
			MinGain:                  -1,
			Attack:                   1,
			Release:                  1,
			SpreadRadius:             -1,
			Floor:                    0,
		},
	)

	if detector.cfg != defaults {
		t.Fatalf("expected config %+v, got %+v", defaults, detector.cfg)
	}
}

func TestNewTonalTransientDetectorDefaultsMinFrequencyBin(t *testing.T) {
	detector := NewTonalTransientDetector(
		TonalTransientConfig{},
	)

	if detector.cfg.MinFrequencyBin != 64 {
		t.Fatalf(
			"expected minimum frequency bin 64, got %d",
			detector.cfg.MinFrequencyBin,
		)
	}
}

func TestNewTonalTransientDetectorCustomConfig(t *testing.T) {
	cfg := TonalTransientConfig{
		LocalRadius:              3,
		TonalStartDB:             5,
		TonalFullDB:              15,
		FluxStartDB:              2,
		FluxFullDB:               12,
		MovementSearchRadius:     8,
		MovementStartBins:        2,
		MovementFullBins:         5,
		MinFrequencyBin:          32,
		MovementMinRelativePower: 0.2,
		HarmonicToleranceBins:    2,
		HarmonicRelativePower:    0.25,
		Strength:                 0.7,
		MinGain:                  0.4,
		Attack:                   0.2,
		Release:                  0.9,
		SpreadRadius:             3,
		Floor:                    1e-10,
	}

	detector := NewTonalTransientDetector(cfg)

	if detector.cfg != cfg {
		t.Fatalf("expected config %+v, got %+v", cfg, detector.cfg)
	}
}

func TestDefaultTonalTransientConfig(t *testing.T) {
	cfg := DefaultTonalTransientConfig()

	if cfg.LocalRadius != 2 {
		t.Fatalf("expected local radius 2, got %d", cfg.LocalRadius)
	}

	assertTonalTransientClose(t, cfg.TonalStartDB, 5)
	assertTonalTransientClose(t, cfg.TonalFullDB, 14)
	assertTonalTransientClose(t, cfg.FluxStartDB, 3)
	assertTonalTransientClose(t, cfg.FluxFullDB, 18)

	if cfg.MovementSearchRadius != 6 {
		t.Fatalf("expected movement search radius 6, got %d", cfg.MovementSearchRadius)
	}

	if cfg.MovementStartBins != 1 {
		t.Fatalf("expected movement start bins 1, got %d", cfg.MovementStartBins)
	}

	if cfg.MovementFullBins != 4 {
		t.Fatalf("expected movement full bins 4, got %d", cfg.MovementFullBins)
	}

	assertTonalTransientClose(t, cfg.MovementMinRelativePower, 0.1)
	assertTonalTransientClose(t, cfg.HarmonicRelativePower, 0.15)
	assertTonalTransientClose(t, cfg.Strength, 0.5)
	assertTonalTransientClose(t, cfg.MinGain, 0.5)
	assertTonalTransientClose(t, cfg.Attack, 0.3)
	assertTonalTransientClose(t, cfg.Release, 0.85)

	if cfg.SpreadRadius != 2 {
		t.Fatalf("expected spread radius 2, got %d", cfg.SpreadRadius)
	}

	assertTonalTransientClose(t, cfg.Floor, 1e-12)
}

func TestTonalTransientDetectsPersistentTone(t *testing.T) {
	cfg := DefaultTonalTransientConfig()
	detector := NewTonalTransientDetector(cfg)

	first := makeFlatPower(129, 1)
	detector.Process(first)

	tone := makeFlatPower(129, 1)
	tone[80] = 100

	firstDetection := detector.Process(tone)

	if firstDetection.Score[80] <= 0 {
		t.Fatalf("expected transient tone detection, got %v", firstDetection.Score[80])
	}

	secondDetection := detector.Process(tone)

	if secondDetection.SpectralFluxDB[80] != 0 {
		t.Fatalf(
			"expected zero spectral flux for persistent tone, got %v",
			secondDetection.SpectralFluxDB[80],
		)
	}

	if secondDetection.Score[80] <= 0 {
		t.Fatalf(
			"expected persistent tone to remain detected, got %v",
			secondDetection.Score[80],
		)
	}
}

func TestTonalTransientFirstFrameInitializesState(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	power := makeFlatPower(129, 1)
	power[80] = 100

	result := detector.Process(power)

	for k, gain := range result.Gain {
		assertTonalTransientClose(t, gain, 1)

		if result.Score[k] != 0 {
			t.Fatalf("expected zero score at bin %d, got %v", k, result.Score[k])
		}
	}
}

func TestTonalTransientDetectsIsolatedTransientTone(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	previous := makeFlatPower(129, 1)
	detector.Process(previous)

	current := makeFlatPower(129, 1)
	current[80] = 100

	result := detector.Process(current)

	if result.Score[80] <= 0.5 {
		t.Fatalf("expected strong interference score, got %v", result.Score[80])
	}

	if result.TonalProminenceDB[80] < 15 {
		t.Fatalf("expected strong tonal prominence, got %v", result.TonalProminenceDB[80])
	}

	if result.SpectralFluxDB[80] < 15 {
		t.Fatalf("expected strong spectral flux, got %v", result.SpectralFluxDB[80])
	}

	if result.Gain[80] >= 0.9 {
		t.Fatalf("expected interference attenuation, got gain %v", result.Gain[80])
	}
}

func TestTonalTransientDoesNotSuppressBroadbandRise(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	previous := makeFlatPower(129, 1)
	detector.Process(previous)

	current := makeFlatPower(129, 1)

	for k := 75; k <= 85; k++ {
		current[k] = 100
	}

	result := detector.Process(current)

	if result.Score[80] != 0 {
		t.Fatalf("expected zero interference score, got %v", result.Score[80])
	}

	assertTonalTransientClose(t, result.Gain[80], 1)
}

func TestTonalTransientTargetGainUsesMinimumGain(t *testing.T) {
	cfg := DefaultTonalTransientConfig()

	cfg.Strength = 1
	cfg.MinGain = 0.5
	cfg.Attack = 0

	detector := NewTonalTransientDetector(cfg)

	detector.Process(makeFlatPower(129, 1))

	current := makeFlatPower(129, 1)
	current[80] = 100

	result := detector.Process(current)

	assertTonalTransientClose(t, result.Gain[80], 0.5)
}

func TestTonalTransientHarmonicSupportProtectsSpeechLikePeak(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	previous := makeFlatPower(129, 1)
	detector.Process(previous)

	current := makeFlatPower(129, 1)

	current[20] = 30
	current[80] = 100

	result := detector.Process(current)

	if result.HarmonicSupport[80] < 0.9 {
		t.Fatalf("expected strong harmonic support, got %v", result.HarmonicSupport[40])
	}

	if result.Score[80] > 0.1 {
		t.Fatalf(
			"expected harmonic support to protect the peak, got score %v",
			result.Score[80],
		)
	}

	if result.Gain[80] < 0.95 {
		t.Fatalf(
			"expected speech like peak to remain protected, got gain %v",
			result.Gain[80],
		)
	}
}

func TestTonalTransientIgnoresBinsBelowMinimumFrequency(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	previous := makeFlatPower(129, 1)
	detector.Process(previous)

	current := makeFlatPower(129, 1)
	current[40] = 100

	result := detector.Process(current)

	assertTonalTransientClose(t, result.Score[40], 0)
	assertTonalTransientClose(t, result.Gain[40], 1)
}

func TestTonalTransientHarmonicSupportIgnoresDuplicateTargets(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	power := makeFlatPower(32, 1)
	power[4] = 10

	got := detector.calculateHarmonicSupport(power, 4, 10)

	want := float32(1) / (detector.cfg.HarmonicRelativePower * 10)

	assertTonalTransientClose(t, got, want)
}
func TestTonalTransientDetectsMovingPeak(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())
	previous := makeFlatPower(129, 1)

	previous[80] = 100
	previous[83] = 80

	detector.Process(previous)

	current := makeFlatPower(129, 1)
	current[80] = 80
	current[83] = 100

	result := detector.Process(current)

	if result.SpectralFluxDB[83] >= 3 {
		t.Fatalf(
			"expected movement test to avoid strong flux, got %v dB",
			result.SpectralFluxDB[83],
		)
	}

	if result.MovementScore[83] <= 0 {
		t.Fatalf("expected positive movement score, got %v", result.MovementScore[83])
	}

	if result.Score[83] <= 0 {
		t.Fatalf("expected moving tonal peak to be detected, got %v", result.Score[83])
	}
}

func TestTonalTransientSpreadsGainToNeighboringBins(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	detector.Process(
		makeFlatPower(129, 1),
	)

	current := makeFlatPower(129, 1)
	current[80] = 100

	result := detector.Process(current)

	if result.Gain[80] >= 1 {
		t.Fatal("expected center bin attenuation")
	}

	if result.Gain[79] >= 1 {
		t.Fatal("expected neighboring bin attenuation")
	}

	if result.Gain[78] >= 1 {
		t.Fatal("expected spread attenuation two bins away")
	}

	if result.Gain[77] != 1 {
		t.Fatalf(
			"expected bin outside spread radius to remain unchanged, got %v", result.Gain[77],
		)
	}

	if result.Gain[80] >= result.Gain[79] {
		t.Fatalf(
			"expected center gain %v to be lower than neighbor gain %v", result.Gain[80], result.Gain[79],
		)
	}
}

func TestTonalTransientReleaseRecoversGradually(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	detector.Process(makeFlatPower(129, 1))

	interference := makeFlatPower(129, 1)
	interference[80] = 100
	detected := detector.Process(interference)
	detectedGain := detected.Gain[80]

	if detectedGain >= 1 {
		t.Fatal("expected interference gain below one")
	}

	clean := makeFlatPower(129, 1)

	recovering := detector.Process(clean)

	if recovering.Gain[80] <= detectedGain {
		t.Fatalf("expected gain to recover above %v, got %v", detectedGain, recovering.Gain[80])
	}

	if recovering.Gain[80] >= 1 {
		t.Fatalf("expected release smoothing below one, got %v", recovering.Gain[80])
	}
}

func TestTonalTransientHarmonicSupportPowerAtFloor(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	power := makeFlatPower(32, 1)

	got := detector.calculateHarmonicSupport(power, 4, detector.cfg.Floor)

	assertTonalTransientClose(t, got, 0)
}

func TestPositiveSpectralFlux(t *testing.T) {
	got := positiveSpectralFlux(100, 1, 1e-12)
	assertTonalTransientClose(t, got, 20)

	got = positiveSpectralFlux(1, 100, 1e-12)
	assertTonalTransientClose(t, got, 0)
}

func TestTonalTransientInvalidPower(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	first := makeFlatPower(129, 1)
	detector.Process(first)

	second := makeFlatPower(129, 1)

	second[50] = float32(math.NaN())
	second[51] = float32(math.Inf(1))
	second[52] = -1

	result := detector.Process(second)

	for k, gain := range result.Gain {
		if math.IsNaN(float64(gain)) || math.IsInf(float64(gain), 0) {
			t.Fatalf("gain bin %d contains invalid value %v", k, gain)
		}
	}
}

func TestTonalTransientResize(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	detector.Process(makeFlatPower(129, 1))

	result := detector.Process(makeFlatPower(65, 1))

	if len(result.Gain) != 65 {
		t.Fatalf("expected 65 gain bins, got %d", len(result.Gain))
	}

	for k, gain := range result.Gain {
		if gain != 1 {
			t.Fatalf("expected reset gain one at bin %d, got %v", k, gain)
		}
	}
}

func TestTonalTransientReset(t *testing.T) {
	detector := NewTonalTransientDetector(DefaultTonalTransientConfig())

	detector.Process(makeFlatPower(129, 1))
	detector.Reset()

	if detector.previousPower != nil {
		t.Fatalf("expected nil previous power, got %v", detector.previousPower)
	}

	if detector.score != nil {
		t.Fatalf("expected nil score, got %v", detector.score)
	}

	if detector.gain != nil {
		t.Fatalf("expected nil gain, got %v", detector.gain)
	}

	if detector.initialized {
		t.Fatal("expected detector to be uninitialized")
	}
}

func TestApplyGain(t *testing.T) {
	spectrum := audio.Spectrum{
		Bins:  []complex64{complex(2, 0), complex(4, 0)},
		Power: []float32{4, 16},
	}

	err := ApplyGain(&spectrum, []float32{0.5, 1})
	if err != nil {
		t.Fatalf("apply interference gain: %v", err)
	}

	assertTonalTransientClose(t, real(spectrum.Bins[0]), 1)
	assertTonalTransientClose(t, spectrum.Power[0], 1)
	assertTonalTransientClose(t, real(spectrum.Bins[1]), 4)
	assertTonalTransientClose(t, spectrum.Power[1], 16)
}

func TestApplyGainSizeMismatch(t *testing.T) {
	spectrum := audio.Spectrum{
		Bins: []complex64{complex(1, 0), complex(1, 0)},
	}

	err := ApplyGain(&spectrum, []float32{1})

	if err == nil {
		t.Fatal("expected gain size mismatch error")
	}
}

func makeFlatPower(size int, value float32) []float32 {
	power := make([]float32, size)

	for k := range power {
		power[k] = value
	}

	return power
}

func TestClampFloat32(t *testing.T) {
	tests := []struct {
		name    string
		value   float32
		minimum float32
		maximum float32
		want    float32
	}{
		{
			name:    "below minimum",
			value:   -1,
			minimum: 0,
			maximum: 1,
			want:    0,
		},
		{
			name:    "inside range",
			value:   0.5,
			minimum: 0,
			maximum: 1,
			want:    0.5,
		},
		{
			name:    "above maximum",
			value:   2,
			minimum: 0,
			maximum: 1,
			want:    1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := clampFloat32(tt.value, tt.minimum, tt.maximum)
			assertTonalTransientClose(t, got, tt.want)
		})
	}
}

func TestMinInt(t *testing.T) {
	tests := []struct {
		name string
		a    int
		b    int
		want int
	}{
		{
			name: "first smaller",
			a:    1,
			b:    2,
			want: 1,
		},
		{
			name: "second smaller",
			a:    2,
			b:    1,
			want: 1,
		},
		{
			name: "equal",
			a:    1,
			b:    1,
			want: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := minInt(tt.a, tt.b)
			if got != tt.want {
				t.Fatalf("expected %d, got %d", tt.want, got)
			}
		})
	}
}

func TestMaxInt(t *testing.T) {
	tests := []struct {
		name string
		a    int
		b    int
		want int
	}{
		{
			name: "first greater",
			a:    2,
			b:    1,
			want: 2,
		},
		{
			name: "second greater",
			a:    1,
			b:    2,
			want: 2,
		},
		{
			name: "equal",
			a:    1,
			b:    1,
			want: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := maxInt(tt.a, tt.b)
			if got != tt.want {
				t.Fatalf("expected %d, got %d", tt.want, got)
			}
		})
	}
}

func assertTonalTransientClose(t *testing.T, got float32, want float32) {
	t.Helper()

	diff := math.Abs(float64(got - want))

	if diff > tonalTransientTestTolerance {
		t.Fatalf("expected %v, got %v", want, got)
	}
}
