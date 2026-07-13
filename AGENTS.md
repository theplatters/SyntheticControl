# Repository Guidelines

## Project Structure & Module Organization

This is a Julia package named `SyntheticControl`.

- `src/SyntheticControl.jl`: main package module, public types, solver implementation, and internal optimization caches.
- `test/runtests.jl`: package test suite using Julia `Test`.
- `test/data_generator.jl`: synthetic data generator used by tests and local benchmarks.
- `docs/make.jl`: Documenter.jl build script for the package documentation.
- `docs/src/`: Markdown source pages for installation, concepts, examples, solver configuration, result interpretation, API reference, and internal implementation details.
- `Project.toml` and `Manifest.toml`: package dependencies and resolved versions.
- `test/Project.toml`: additional test-only dependencies, including `Optimization`, `OptimizationOptimJL`, and AD tooling.
- `docs/Project.toml` and `docs/Manifest.toml`: documentation-only dependencies, including `Documenter`.

Keep source changes concentrated in `src/SyntheticControl.jl` unless adding reusable test fixtures or generated data helpers.

## Build, Test, and Development Commands

Run commands from the repository root.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
Installs package dependencies from the manifest.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
Runs the full test suite in `test/runtests.jl`.

```bash
julia --project=. -e 'using SyntheticControl'
```
Checks that the package loads in the active environment.

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
```
Installs documentation dependencies.

```bash
julia --project=docs docs/make.jl
```
Builds the Documenter.jl static HTML site in `docs/build/`, runs doctests, and fails on missing docstrings, invalid cross-references, and doctest failures.

For local performance checks, prefer BenchmarkTools in the REPL if available, e.g. `@btime solve($problem)` after a warm-up solve.

## Coding Style & Naming Conventions

Use two-space indentation, as in the existing source. Prefer descriptive names over abbreviations for new internals, especially in solver code: `predictor_weights`, `donor_weights`, `active_set`, and `mspe` are preferred patterns.

Functions that mutate preallocated buffers should end in `!`, for example `normalize_weights!` or `optimize_donor_weights!`. Keep hot solver paths allocation-conscious; avoid views, comprehensions, or temporary arrays in repeated inner loops unless measured.

Every new or changed package-defined type, constructor, function, and callable method must include a Julia docstring. Docstrings should describe arguments, return values, relevant constraints, possible errors, and side effects. Include a concrete minimal Julia example for each documented method; use `jldoctest` examples in documentation pages when practical and plain `julia` examples in docstrings when output would be brittle.

## Testing Guidelines

Tests use Julia’s standard `Test` framework. Add regression tests for:

- dimension and input validation,
- solver numerical behavior,
- allocation-sensitive or performance-sensitive code paths,
- generated-data edge cases such as `K > T_pre`.

Keep tests deterministic by passing explicit seeds when randomness matters. Run `Pkg.test()` before submitting changes.

Documentation examples are tested by Documenter. Run `julia --project=docs docs/make.jl` after changing docstrings or files under `docs/src/`; the build is configured to fail on missing docstrings, invalid references, and doctest failures.

## Commit & Pull Request Guidelines

Commit messages must use the format `<feat> (<scope>): <change>`, where `<feat>` is the change category, `<scope>` names the affected package area, and `<change>` is a concise imperative summary. For example: `docs (api): add Documenter reference pages`.

Pull requests should include:

- a short description of behavior changed,
- test commands run and results,
- performance numbers when solver runtime or allocations change,
- notes on API changes such as constructor keywords or renamed types.
