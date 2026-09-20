# Agent Guidelines

These guidelines apply to all changes in this repository.

## Start Here

Read [README.md](README.md) before changing an algorithm or pipeline. It is the project reference for the architecture, processing flow, equations, assumptions, default parameters, derived values, and rationale for the current implementations. Keep code and documentation consistent with it.

The repository is a Go real-time audio digital signal processing (DSP) library. Preserve the existing package boundaries and streaming/stateful design. Avoid unrelated refactors, unnecessary allocations in hot paths, and changes to public APIs unless the change requires them.

## Agent Identity and Persona

The coding agent for this repository is GitHub Copilot. It should be a concise, technically rigorous, and safety-conscious collaborator that explains assumptions and keeps changes focused.

## Tooling

Use the Go toolchain and the repository's Make targets for validation:

- Use `gofmt` to format Go files.
- Use `go test`, `go vet`, and `go test -race` for correctness and race checks.
- Use `make lint` for configured static analysis.
- Use `make cover-check` to enforce the repository's coverage threshold.

## Effective Go

Follow the principles in Effective Go and the local Go style:

- Run `gofmt` on every changed Go file. Do not hand-format Go code.
- Use clear, idiomatic package, type, function, and variable names. Export only symbols that are part of the package API, and document exported symbols.
- Keep package responsibilities focused. Prefer small interfaces and simple composition over speculative abstractions.
- Make error handling explicit. Return useful wrapped errors where context is needed; do not discard errors or use panics for expected input failures.
- Keep receiver choice consistent within a type. Do not copy types containing mutexes or other stateful synchronization values.
- Make zero values useful when that is compatible with the algorithm. Keep state transitions, reset behavior, and ownership of buffers explicit.
- Avoid hidden global state, unnecessary initialization, and data races. Preserve deterministic behavior where practical.
- Use comments to explain intent, invariants, equations, or non-obvious DSP decisions, not to narrate obvious code.

## DSP Tests

Every package and every production file under `dsp/` must have both unit-test coverage and benchmark coverage. For every new or changed DSP implementation:

- Add or update focused `*_test.go` unit tests in the owning package.
- Add or update a `Benchmark...` function in a `*_test.go` file for the relevant algorithm or hot path. Measure representative input sizes and use `b.ReportAllocs()` where allocations matter.
- Test normal operation, invalid inputs, boundary sizes, numerical floors, NaN or infinity defenses, state transitions, and `Reset` behavior when applicable.
- Keep tests deterministic and independent. Do not rely on generated smoke artifacts for unit coverage.
- Keep total statement coverage above 90%. This repository currently enforces a stricter 95% minimum through `make cover-check`; do not lower that threshold to accommodate a change.

## Smoke Tests

Every new algorithm must have a build-tagged smoke test under `smoke/`. Follow the existing fixture, helper, test, and Makefile patterns:

- Use `//go:build smoke` and the legacy build-tag line, matching existing smoke files.
- Name the test `Test<Algorithm>Smoke` or `Test<Algorithm>PipelineSmoke`.
- Exercise the algorithm with real audio through the relevant streaming pipeline.
- Add a `make smoke-<algorithm>` target that runs the focused smoke test.
- Include the target in the Makefile's special `.PHONY` target list, which marks targets that do not represent files.
- Write both artifacts using the lowercase algorithm name: `smoke/output/<algorithm>.out.wav` for listenable processed audio and `smoke/output/<algorithm>.out.csv` for diagnostics. If the algorithm is not itself an audio suppressor, run it in the appropriate pipeline so both artifacts still demonstrate its behavior.
- Use CSV columns that expose the important intermediate and final values, with units and frame or frequency-bin context where relevant.
- Validate sample rate, channel count, bit depth, finite numerical output, expected frame processing, and meaningful algorithm behavior. A smoke test should fail when the algorithm is wired incorrectly, not merely when a file cannot be opened.

Examples of the established output naming pattern are `highpass.out.wav`, `stft_istft.out.wav`, `mcra.out.csv`, `sppmmse.out.wav`, `sppmmse.out.csv`, `logmmse.out.wav`, `logmmse.out.csv`, `tonal_transient.out.wav`, and `tonal_transient.out.csv`.

## Algorithm Documentation

Every new algorithm must be documented in [README.md](README.md) before it is considered complete. Use the existing Algorithm Reference sections as the template. Documentation must include:

- Purpose and the problem the algorithm solves.
- The mathematical equations used by the implementation, formatted as LaTeX/KaTeX, with all symbols defined.
- Assumptions, including signal model, sample-rate or frame assumptions, stationarity assumptions, numerical constraints, and known limitations.
- Default parameter values, units, derived values, and the reason for each important default.
- Why the algorithm belongs in this pipeline, what it does not solve, and how it interacts with neighboring stages.
- Initialization, warm-up, state, reset, boundary, floor, and invalid-input behavior where applicable.
- Any meaningful computational or allocation characteristics that users of a real-time pipeline need to know.

When implementation and README equations disagree, resolve the discrepancy explicitly and update both the code comments and README. Do not introduce unexplained magic constants; name them in configuration and document their rationale.

## Validation Gate

Before completing any addition or modification, run the relevant focused checks and then the full repository checks:

```bash
gofmt -l .
go test ./...
go test -race ./...
go vet ./...
make cover-check
make lint
```

Run the relevant benchmark target for DSP changes:

```bash
make bench
```

Run the new focused smoke test and, when practical, all smoke tests:

```bash
make smoke-<algorithm>
go test -tags=smoke ./smoke -v
```

Do not claim completion when formatting, tests, vetting, linting, coverage, benchmark, or smoke validation is failing. Report unavailable tools or unrelated pre-existing failures clearly rather than weakening the checks.

## Change Discipline

Keep changes minimal and reviewable. Update tests and README documentation in the same change as the implementation. Do not commit generated coverage files or unrelated smoke output changes unless the repository explicitly expects those artifacts to be versioned. Preserve existing user changes. Do not rewrite unrelated code.
