# Repository Guidelines

## Project Structure & Module Organization

This is a Julia package named `SyntheticControl`.

- `src/SyntheticControl.jl`: main package module, public types, solver implementation, and internal optimization caches.
- `src/visualization.jl`: backend-independent visualization data containers, statistical helpers, and public Makie plotting entry points.
- `ext/SyntheticControlMakieExt.jl`: Makie weak-dependency extension defining full recipes and `Makie.plot!` implementations.
- `test/runtests.jl`: package test suite using Julia `Test`.
- `test/visualization_tests.jl`: statistical visualization-helper tests and Makie recipe structure tests.
- `test/data_generator.jl`: synthetic data generator used by tests and local benchmarks.
- `docs/make.jl`: Documenter.jl build script for the package documentation.
- `docs/serve.jl`: small local HTTP server for previewing `docs/build/` without `file://` asset-loading issues.
- `docs/src/`: Markdown source pages for installation, concepts, examples, solver configuration, result interpretation, visualization, API reference, and internal implementation details.
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

```bash
julia --project=docs docs/serve.jl
```
Serves the generated documentation at `http://127.0.0.1:8000/` for local preview. Prefer this over opening `docs/build/index.html` directly, because Documenter runtime assets can fail under `file://` URLs in some browsers.

For interactive visualization work, load GLMakie before package plotting calls:

```julia
using GLMakie
using SyntheticControl
GLMakie.activate!()
pathplot(result)
```

For local performance checks, prefer BenchmarkTools in the REPL if available, e.g. `@btime solve($problem)` after a warm-up solve.

## Coding Style & Naming Conventions

Use two-space indentation, as in the existing source. Prefer descriptive names over abbreviations for new internals, especially in solver code: `predictor_weights`, `donor_weights`, `active_set`, and `mspe` are preferred patterns.

Functions that mutate preallocated buffers should end in `!`, for example `normalize_weights!` or `optimize_donor_weights!`. Keep hot solver paths allocation-conscious; avoid views, comprehensions, or temporary arrays in repeated inner loops unless measured.

Every new or changed package-defined type, constructor, function, and callable method must include a Julia docstring. Docstrings should describe arguments, return values, relevant constraints, possible errors, and side effects. Include a concrete minimal Julia example for each documented method; use `jldoctest` examples in documentation pages when practical and plain `julia` examples in docstrings when output would be brittle.

Visualization architecture is split deliberately: statistical helpers and validated path/placebo data containers live in `src/visualization.jl`, while Makie rendering lives in `ext/SyntheticControlMakieExt.jl`. Do not put estimation or placebo-filtering calculations inside `Makie.plot!`; recipes should consume existing result objects or prepared plotting data and compose Makie primitives such as `lines!`, `vlines!`, `hlines!`, `scatter!`, and `text!`.

The public Makie recipes are `pathplot`, `gapplot`, `placeboplot`, and `placebodistribution`, each with mutating `!` forms. `pathplot` compares observed and synthetic paths; `gapplot` shows actual-minus-synthetic gaps; `placeboplot` compares treated and placebo gaps after documented filtering; `placebodistribution` shows post/pre RMSPE ratios and the finite-sample randomization p-value.

## Testing Guidelines

Tests use Julia’s standard `Test` framework. Add regression tests for:

- dimension and input validation,
- solver numerical behavior,
- allocation-sensitive or performance-sensitive code paths,
- generated-data edge cases such as `K > T_pre`.

Keep tests deterministic by passing explicit seeds when randomness matters. Run `Pkg.test()` before submitting changes.

Documentation examples are tested by Documenter. Run `julia --project=docs docs/make.jl` after changing docstrings or files under `docs/src/`; the build is configured to fail on missing docstrings, invalid references, and doctest failures.

Visualization changes need tests for both layers: numerical tests for gaps, RMSPE values, ratios, placebo filtering, and p-values; recipe tests for mutating/non-mutating Makie calls, child plot structure, marker locations, and attribute propagation. Prefer inspecting Makie plot objects and converted arguments over pixel comparisons. Run at least a lightweight GLMakie smoke test when changing the extension, and use CairoMakie for headless documentation rendering.

## Commit & Pull Request Guidelines

Commit messages must use the format `<feat> (<scope>): <change>`, where `<feat>` is the change category, `<scope>` names the affected package area, and `<change>` is a concise imperative summary. For example: `docs (api): add Documenter reference pages`.

Pull requests should include:

- a short description of behavior changed,
- test commands run and results,
- performance numbers when solver runtime or allocations change,
- notes on API changes such as constructor keywords or renamed types.
