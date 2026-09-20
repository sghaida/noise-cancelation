APP_NAME := noise-cancelation
COVERAGE_FILE := coverage.out
COVERAGE_MIN := 95
GOLANGCI_LINT := $(shell go env GOPATH)/bin/golangci-lint
COVERAGE_PACKAGES := $(filter-out %/smoke,$(shell go list ./...))

.PHONY: test test-race cover cover-race clean bench bench-all bench-e2e golintcli lint smoke-highpass smoke-stft smoke-mcra smoke-sppmmse smoke-tonal-transient

test:
	go test ./...

test-race:
	go test -race ./...

cover:
	go test \
		-coverprofile=$(COVERAGE_FILE) \
		-covermode=atomic \
		$(COVERAGE_PACKAGES)

	go tool cover -func=$(COVERAGE_FILE)

cover-race:
	go test \
		-race \
		-coverprofile=$(COVERAGE_FILE) \
		-covermode=atomic \
		$(COVERAGE_PACKAGES)

	go tool cover -func=$(COVERAGE_FILE)

cover-check: cover
	@coverage=$$(go tool cover \
		-func=$(COVERAGE_FILE) \
		| awk '/^total:/ {gsub(/%/, "", $$3); print $$3}'); \
	echo "Total coverage: $$coverage%"; \
	awk \
		-v coverage="$$coverage" \
		-v minimum="$(COVERAGE_MIN)" \
		'BEGIN { \
			if (coverage + 0 < minimum + 0) { \
				printf "Coverage %.2f%% is below required %.2f%%\n", coverage, minimum; \
				exit 1; \
			} \
			printf "Coverage %.2f%% meets required %.2f%%\n", coverage, minimum; \
		}'

clean:
	rm -f $(COVERAGE_FILE)

bench:
	go test \
		-run=^$$ \
		-bench=. \
		-benchmem \
		./...

bench-all:
	go test \
		-run=^$$ \
		-bench=. \
		-benchmem \
		-benchtime=3s \
		-count=3 \
		./...

bench-e2e:
	go test \
		./benchmark \
		-run=^$$ \
		-bench='^BenchmarkEndToEndPipeline100ConcurrentTwoMinutes$$' \
		-benchmem \
		-benchtime=1x

golintcli:
	go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest

lint:
	$(GOLANGCI_LINT) run --config .golangci.yml

coverage:
	go test $(COVERAGE_PACKAGES) -coverprofile=$(COVERAGE_FILE)
	go tool cover -func=$(COVERAGE_FILE)

smoke-highpass:
	go test -tags=smoke ./smoke -run TestHighPassSmoke -v

smoke-stft:
	go test -tags=smoke ./smoke -run TestSTFTISTFTSmoke -v

smoke-mcra:
	go test -tags=smoke ./smoke -run TestMCRASmoke -v

smoke-logmmse:
	go test -tags smoke ./smoke -run TestLogMMSEPipelineSmoke -v

smoke-sppmmse:
	go test -tags smoke ./smoke -run TestSPPMMSEPipelineSmoke -v

smoke-tonal-transient:
	go test -tags smoke ./smoke -run TestTonalTransientPipelineSmoke -v