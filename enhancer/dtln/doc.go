// Package dtln implements streaming DTLN speech enhancement using ONNX Runtime
//
// # DTLN uses two neural separation stages
//
// Stage one estimates a spectral magnitude mask
//
//	M[k] = model1(|X[k]|)
//
// The mask is applied while preserving the original phase
//
//	Y1[k] = M[k] X[k]
//
// Stage two refines the reconstructed time domain frame
//
//	y2[n] = model2(IFFT(Y1[k]))
//
// Streaming reconstruction uses overlap add with hop H
//
//	y[n] = Σm ym[n - mH]
//
// # The pretrained ONNX model uses 16 kHz audio, 512 sample blocks and 128 sample hops
//
// # References
//
// https://arxiv.org/abs/2005.07551
// https://github.com/breizhn/DTLN
package dtln
