package mulaw

import (
	"math/bits"

	"github.com/sghaida/noise-cancelation/codec"
)

const (
	muLawBias = 0x84 // 132
	muLawClip = 32635
)

// MuLaw implements the G.711 µlaw audio codec
//
// G.711 µlaw represents each audio sample using one byte
// this implementation assumes an 8 kHz mono stream, which is the format commonly used by
// telephony systems such as Twilio Media Streams
//
// encoded µlaw samples are converted internally to normalized float32 PCM
// samples in the range [-1.0, +1.0] for use by DSP components
type MuLaw struct{}

// Ensure MuLaw implements Codec
var _ codec.Codec = (*MuLaw)(nil)

// NewMuLaw creates a new G.711 µlaw codec
// codec operates at 8 kHz with a single audio channel
func NewMuLaw() *MuLaw {
	return &MuLaw{}
}

// Name returns the name of the codec
func (m *MuLaw) Name() string {
	return "mulaw"
}

// SampleRate returns the sample rate used by G.711 µlaw
// this implementation operates at 8000 samples per second
func (m *MuLaw) SampleRate() int {
	return 8000
}

// Channels returns the number of audio channels used by the codec
// this implementation operates on mono audio so it returns 1
func (m *MuLaw) Channels() int {
	return 1
}

// Decode converts G.711 µlaw encoded audio into normalized float32 PCM
// each input byte represents exactly one audio sample
// the resulting PCM samples are normalized to approximately the range [-1.0, +1.0]
// for example, a 20 ms µlaw frame at 8 kHz contains 160 bytes and produces
// 160 float32 PCM samples
//
// decode allocates a new output slice
// for realtime processing where allocations should be minimized
// use DecodeMuLaw instead
func (m *MuLaw) Decode(data []byte) []float32 {
	out := make([]float32, len(data))

	for i, sample := range data {
		// int16 PCM sample ranges from -32768 ... 32767
		// using 32768.0 the result will map ~ -1 ... 0.99997
		out[i] = float32(ToPCM16(sample)) / 32768.0
	}

	return out
}

// Encode converts normalized float32 PCM samples into G.711 µlaw
// input samples are expected to be in the range [-1.0, +1.0]
// values outside that range are
// clamped before being converted to signed 16-bit PCM and then encoded as µlaw
// each input PCM sample produces exactly one encoded µlaw byte
//
// encode allocates a new output slice
// for realtime processing where allocations should be minimized
// use EncodeMuLaw instead
func (m *MuLaw) Encode(samples []float32) []byte {
	out := make([]byte, len(samples))

	for i, sample := range samples {
		sample = clampSample(sample)

		pcm := int16(sample * 32767.0)

		out[i] = ToMuLaw(pcm)
	}

	return out
}

// ToPCM16 decodes a single G.711 µlaw encoded byte into a signed
// 16-bit linear PCM sample
//
// G.711 µlaw stores the sample using:
//
//	sign | exponent | mantissa
//	  1       3          4
//
// the encoded byte uses one's complement representation, so the bits are
// first inverted before reconstructing the linear PCM magnitude
//
// the returned value is linear PCM and is not normalized
// example 11001110
// 1 - invert  ^ = 00110001 => 0 | 011 | 0001
// 2 - mask the sign 00110001 & 10000000 => 00000000
// 3 - extract exponent => shift by 4 => 00000011
//
//	mask the first 3 digits => 00000111 => 00000011 & 00000111 => 00000011
//
// 4 - extract the mantissa 00110001 & 00001111 = 00000001
func ToPCM16(encoded byte) int16 {
	// G.711 µlaw stores the encoded value inverted
	encoded = ^encoded

	sign := encoded & 0x80
	exponent := (encoded >> 4) & 0x07
	mantissa := encoded & 0x0F

	// Reconstruct the linear magnitude
	// mantissa contains the fine-grained value within the logarithmic
	// segment selected by the exponent
	magnitude := ((int(mantissa) << 3) + muLawBias) << exponent

	// Remove the bias that was introduced during encoding
	magnitude -= muLawBias

	if sign != 0 {
		return pcm16(-magnitude)
	}

	return pcm16(magnitude)
}

func pcm16(value int) int16 {
	if value > 32767 {
		return 32767
	}

	if value < -32768 {
		return -32768
	}

	return int16(value) //nolint:gosec // value is clamped to int16's range
}

// ToMuLaw encodes a single signed 16-bit linear PCM sample as a G.711 µlaw byte
//
// the PCM magnitude is clipped to the maximum value representable by
// G.711 µlaw, biased, divided into an exponent and mantissa and
// stored using one's complement representation
//
// ToMuLaw performs no sample rate conversion
// it encodes one PCM sample into exactly one µlaw byte
func ToMuLaw(sample int16) byte {
	pcm := int(sample)

	var sign byte

	if pcm < 0 {
		sign = 0x80
		pcm = -pcm
	}

	// limit magnitude to the maximum value that G.711 µlaw can encode
	if pcm > muLawClip {
		pcm = muLawClip
	}

	// add G.711 µlaw bias before determining the logarithmic segment
	pcm += muLawBias

	// determine which logarithmic segment contains the sample
	exponent := bits.Len16(uint16(pcm)) - 8 //nolint:gosec // pcm is non-negative and clipped below 65536

	// extract the four-bit mantissa within the selected segment
	mantissa := byte((pcm >> (exponent + 3)) & 0x0F) //nolint:gosec // mask limits the value to four bits

	encoded := sign | byte(exponent<<4) | mantissa //nolint:gosec // exponent is limited to the seven G.711 segments

	// G.711 µlaw transmits the one's complement of the encoded value
	return ^encoded
}

// DecodeMuLaw decodes G.711 µlaw data into normalized float32 PCM while
// allowing the caller to reuse an existing destination buffer
//
// each byte in src represents one µlaw audio sample. The resulting samples
// are normalized to approximately the range [-1.0, +1.0]
//
// if dst has sufficient capacity, its backing array is reused. Otherwise,
// a new slice is allocated
//
// the returned slice always has len(src) elements
//
// this function is preferable to MuLaw.Decode in realtime audio processing
// loops because it can avoid repeated allocations
func DecodeMuLaw(dst []float32, src []byte) []float32 {
	if cap(dst) < len(src) {
		dst = make([]float32, len(src))
	} else {
		dst = dst[:len(src)]
	}

	for i, sample := range src {
		dst[i] = float32(ToPCM16(sample)) / 32768.0
	}

	return dst
}

// EncodeMuLaw encodes normalized float32 PCM samples as G.711 µlaw while
// allowing the caller to reuse an existing destination buffer
//
// input samples are clamped to the range [-1.0, +1.0] before being converted
// to signed 16-bit PCM and encoded as µlaw
//
// if dst has sufficient capacity, its backing array is reused. Otherwise,
// a new slice is allocated
//
// the returned slice always has len(src) elements
//
// this function is preferable to MuLaw.Encode in realtime audio processing
// loops because it can avoid repeated allocations
func EncodeMuLaw(dst []byte, src []float32) []byte {
	if cap(dst) < len(src) {
		dst = make([]byte, len(src))
	} else {
		dst = dst[:len(src)]
	}

	for i, sample := range src {
		sample = clampSample(sample)

		dst[i] = ToMuLaw(int16(sample * 32767.0))
	}

	return dst
}

// clampSample constrains a normalized PCM sample to the range [-1.0, +1.0]
func clampSample(sample float32) float32 {
	if sample > 1.0 {
		return 1.0
	}

	if sample < -1.0 {
		return -1.0
	}

	return sample
}
