# Repository Guidelines

## Project Structure & Module Organization

This is a Julia package named `SyntheticControl`.

- `src/SyntheticControl.jl`: main package module, public types, solver implementation, and internal optimization caches.
- `src/visualization.jl`: backend-independent visualization data containers, statistical helpers, and public Makie plotting entry points.
- `src/robustness.jl`: panel metadata, robustness result types, refitting logic, failure records, RMSPE inference, and public robustness APIs.
- `ext/SyntheticControlMakieExt.jl`: Makie weak-dependency extension defining full recipes and `Makie.plot!` implementations.
- `ext/SyntheticControlTablesExt.jl`: Tables.jl weak-dependency extension for long-format panel input and table-shaped result output. It activates only after both `using SyntheticControl` and `using Tables`.
- `test/runtests.jl`: package test suite using Julia `Test`.
- `test/visualization_tests.jl`: statistical visualization-helper tests and Makie recipe structure tests.
- `test/data_generator.jl`: synthetic data generator used by tests and local benchmarks.
- `docs/make.jl`: Documenter.jl build script for the package documentation.
- `docs/serve.jl`: small local HTTP server for previewing `docs/build/` without `file://` asset-loading issues.
- `docs/src/`: Markdown source pages for installation, concepts, examples, solver configuration, result interpretation, visualization, API reference, and internal implementation details.
- `Project.toml` and `Manifest.toml`: package dependencies and resolved versions.
- `test/Project.toml`: additional test-only dependencies, including `Optimization`, `OptimizationOptimJL`, and AD tooling.
- `docs/Project.toml` and `docs/Manifest.toml`: documentation-only dependencies, including `Documenter`.

Keep estimator source changes concentrated in `src/SyntheticControl.jl` and
the estimator-specific included files. Robustness refitting and inference
belong in `src/robustness.jl`; visualization statistics remain in
`src/visualization.jl`.

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

Makie recipe customization follows Makie's standard split between recipe, Axis, Figure, and Legend attributes. Recipe attributes control plotted content and appearance only. Axis attributes such as `title`, `xlabel`, `ylabel`, ticks, limits, scales, grids, and tick formatting belong on `Axis(...)` or the non-mutating `axis=(; ...)` keyword. Figure attributes belong on `Figure(...)` or `figure=(; ...)`. Legend placement, orientation, titles, columns, and styling belong to `axislegend` or `Legend`; recipes should expose child `label` attributes and should not implement recipe-specific legend layout.

Public recipe attributes include:

- `pathplot`: `actual_color`, `actual_linewidth`, `actual_linestyle`, `actual_label`, `synthetic_color`, `synthetic_linewidth`, `synthetic_linestyle`, `synthetic_label`, `treatment_color`, `treatment_linewidth`, `treatment_linestyle`, `treatment_label`, and `show_treatment`.
- `gapplot`: `gap_color`, `gap_linewidth`, `gap_linestyle`, `gap_label`, `zero_color`, `zero_linewidth`, `zero_linestyle`, `zero_label`, `show_zero`, `treatment_color`, `treatment_linewidth`, `treatment_linestyle`, `treatment_label`, and `show_treatment`.
- `placeboplot`: `treated_color`, `treated_linewidth`, `treated_linestyle`, `treated_label`, `placebo_color`, `placebo_alpha`, `placebo_linewidth`, `placebo_linestyle`, `placebo_label`, `show_placebos`, `show_excluded`, `excluded_color`, `excluded_alpha`, `excluded_linewidth`, `excluded_linestyle`, `excluded_label`, `pre_rmspe_threshold`, `zero_color`, `zero_linewidth`, `zero_linestyle`, `zero_label`, `show_zero`, `treatment_color`, `treatment_linewidth`, `treatment_linestyle`, `treatment_label`, and `show_treatment`.
- `placebodistribution`: `pre_rmspe_threshold`, `placebo_color`, `placebo_alpha`, `placebo_markersize`, `placebo_marker`, `placebo_label`, `treated_color`, `treated_alpha`, `treated_markersize`, `treated_marker`, `treated_label`, `treated_line_color`, `treated_linewidth`, `treated_linestyle`, `treated_line_label`, `show_treated_line`, `show_p_value`, `p_value_label`, `p_value_color`, `p_value_fontsize`, and `p_value_align`.

Makie `plot!` methods must not create `Figure` or `Axis` objects and must not overwrite user-provided axis titles, labels, ticks, limits, or scales. Non-mutating calls may provide default SCM axis labels through Makie's axis-hint mechanisms, but explicit `axis=(; ...)` values and mutating calls into an existing `Axis` must win. Propagate recipe attributes to child plots as observables, use `@inherit` for Makie-wide defaults such as `linewidth`, `markersize`, `marker`, and `fontsize`, and validate only recipe-specific constraints such as non-negative line widths and opacity values in `[0, 1]`.

Tables.jl integration must stay in `ext/SyntheticControlTablesExt.jl`; the core package only declares generic public hooks. The public table API is `from_table`, `weights_table`, `balance_table`, and `path_table`. `from_table` accepts Tables.jl-compatible long panels with unit, time, outcome, and predictor columns selected by `Symbol` or `String`. It materializes the complete panel once, rejects duplicate `(unit, time)` observations, unbalanced panels, missing/non-finite numeric values, non-numeric outcomes or predictors, unknown columns, absent treated units, empty donor pools, and treatment times outside observed values. It sorts time values and donor identifiers explicitly so row order does not affect results.

Tables matrix orientation conventions are fixed: `X1` is length `K`, `X0` is `K × J`, `Y1` is length `T_pre`, and `Y0` is `T_pre × J`. Predictor columns are mean-aggregated over pre-treatment observations only, where times before `treatment_time` are pre-treatment and `treatment_time` is the first post-treatment period. `weights_table` returns `donor, weight`; `balance_table` returns `predictor, treated, synthetic, difference`; `path_table` returns `time, actual, synthetic, gap, post_treatment`. Return lightweight Tables.jl-compatible objects, not DataFrames or other concrete sink types.

## Testing Guidelines

Tests use Julia’s standard `Test` framework. Add regression tests for:

- dimension and input validation,
- solver numerical behavior,
- allocation-sensitive or performance-sensitive code paths,
- generated-data edge cases such as `K > T_pre`.

Keep tests deterministic by passing explicit seeds when randomness matters. Run `Pkg.test()` before submitting changes.

Documentation examples are tested by Documenter. Run `julia --project=docs docs/make.jl` after changing docstrings or files under `docs/src/`; the build is configured to fail on missing docstrings, invalid references, and doctest failures.

Visualization changes need tests for both layers: numerical tests for gaps, RMSPE values, ratios, placebo filtering, and p-values; recipe tests for mutating/non-mutating Makie calls, child plot structure, marker locations, and attribute propagation. Prefer inspecting Makie plot objects and converted arguments over pixel comparisons. Run at least a lightweight GLMakie smoke test when changing the extension, and use CairoMakie for headless documentation rendering.

Recipe customization changes also need tests that every documented recipe attribute is accepted, attributes propagate to the intended child plot, independent styling works for actual, synthetic, treated, placebo, treatment, zero, marker, and distribution elements, `nothing` labels suppress legend entries, `axislegend(ax)` discovers labeled child elements, explicit `Axis` settings are preserved, non-mutating `axis=(; ...)` values work, `with_theme` and observable updates affect inherited or observable-backed attributes, and invalid recipe-specific values fail clearly. Update `docs/src/visualization.md` and recipe docstrings when adding, removing, or renaming public recipe attributes.

Tables extension changes need tests for activation with Tables.jl, named-tuple column tables, row tables, minimal custom Tables.jl sources, single-pass schema-less sources, shuffled input rows, multiple predictors, matrix orientations and values, output schemas, solver integration, non-string identifiers, supported time types, and validation failures. Update `docs/src/tables.md` and run the Documenter build after changing public table behavior or docstrings.

Robustness APIs are `in_space_placebos`, `leave_one_out`, and
`in_time_placebos`, returning `InSpacePlaceboResult`, `LeaveOneOutResult`, and
`InTimePlaceboResult`. Full panel storage uses `SyntheticControlPanelData` and
retains generic identifier and ordered time types. Refits must clone the
original estimator configuration, own independent solver caches, preserve
deterministic assignment order, and record rather than discard failures under
`on_failure=:record`.

The treatment period is the first post-treatment period. RMSPE ratios use
post/pre RMSPE; `0/0` is `1` and positive/zero is `Inf`. In-space inference
includes the treated assignment, counts ties with `>=`, and excludes failed,
filtered, or non-finite placebo ratios. Relative filtering never removes
stored refits. A placebo is excluded from its own donors, and the original
treated unit is excluded unless `include_treated=true`. Leave-one-out removes
exactly one donor and has no p-value. In-time fitting uses only observations
before the pseudo-date and, by default, evaluation stops before real
treatment; windows count observed periods rather than calendar distance.

Tables robustness methods are `placebo_summary`, `leave_one_out_summary`, and
`in_time_summary`. Makie consumes stored results through `placeboplot`,
`placebodistribution`, `leaveoneoutplot`, and `intimeplaceboplot`; recipes must
never refit or calculate statistics. Future robustness changes require tests
for reassignment, donor composition, windows, failures, zero/non-finite RMSPE,
exact ranks and ties, generic times, table schemas, recipe construction, and
serial/parallel equivalence. Update `docs/src/robustness.md`, relevant
docstrings, Tables docs, and visualization docs, then run full tests and docs.

## Commit & Pull Request Guidelines

Commit messages must use the format `<feat> (<scope>): <change>`, where `<feat>` is the change category, `<scope>` names the affected package area, and `<change>` is a concise imperative summary. For example: `docs (api): add Documenter reference pages`.

Pull requests should include:

- a short description of behavior changed,
- test commands run and results,
- performance numbers when solver runtime or allocations change,
- notes on API changes such as constructor keywords or renamed types.
