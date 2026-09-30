//go:build !darwin && !linux

package benchmark

import "runtime/metrics"

func readProcessCPUSeconds() float64 {
	samples := []metrics.Sample{{Name: "/cpu/classes/total:cpu-seconds"}}
	metrics.Read(samples)

	return samples[0].Value.Float64()
}
