package noise

// Estimator estimates the noise power spectral density for each frequency bin
// for each STFT frame t and frequency bin k, the estimator receives the
// observed power spectrum:
//
//	P[t,k] = |X[t,k]|²
//
// where:
//
//	X[t,k] is the complex STFT coefficient
//	P[t,k] is the observed signal power
//	t is the frame index
//	k is the frequency-bin index
//
// the estimator produces N[t,k]
//
// where:
//
//	N[t,k] is the estimated noise power spectral density
//
// the resulting noise PSD can later be used to estimate the posterior SNR:
//
//	γ[t,k] = P[t,k] / (N[t,k] + ε)
//
// where ε is a small positive value used to avoid division by zero
type Estimator interface {
	// Process estimates the noise power spectral density for the current frame
	// power contains the observed power spectrum:
	//	P[k] = |X[k]|²
	//
	// the returned slice contains the estimated noise PSD N[k]
	// the returned slice may be reused by the implementation and should not be
	// retained or modified by the caller unless the concrete implementation
	// explicitly documents otherwise
	Process(power []float32) []float32

	// Reset clears the internal estimator state
	Reset()
}
