# Unified Robustness Suite

[`robustness_suite`](@ref) is a thin orchestration layer over the package's
standalone robustness APIs. It retains each component's native result type,
calculates baseline diagnostics once, and never replaces or mutates the
baseline fit.

```jldoctest suite
julia> using SyntheticControl, CommonSolve, Tables

julia> panel = (
           unit=repeat([:treated, :a, :b], inner=5),
           time=repeat([1, 2, 3, 4, 5], 3),
           outcome=[2., 3, 4, 7, 9, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7],
           x=[2., 3, 4, 5, 6, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7],
       );

julia> problem = from_table(panel; unit=:unit, time=:time, outcome=:outcome,
                            predictors=[:x], treated=:treated,
                            treatment_time=4);

julia> fit = solve(problem);

julia> suite = robustness_suite(
           fit; in_space=true, leave_one_out=:active, placebo_times=[3],
       );

julia> suite.states.in_space.status
:success

julia> suite.in_space isa InSpacePlaceboResult
true

julia> robustness_summary(suite).analysis
4-element Vector{Symbol}:
 :in_space
 :leave_one_out
 :in_time
 :specification_sensitivity
```

The fit-only form works for results returned by [`solve`](@ref). For a fit
constructed manually, pass both the original problem and fit:

```julia
robustness_suite(problem, fit; in_space=true)
```

## Configuration

The concise forms map to documented defaults:

- `in_space=true` runs all eligible donor-unit assignments.
- `leave_one_out=true` omits every donor; `:active` omits only donors with a
  positive baseline weight under the component API's tolerance.
- `placebo_times=times` enables in-time placebos.
- `specifications=specs` enables specification sensitivity.
- `false` disables a component.

Use [`InSpaceOptions`](@ref), [`LeaveOneOutOptions`](@ref),
[`InTimeOptions`](@ref), and [`SpecificationSensitivityOptions`](@ref) when a
component needs advanced settings. These options map directly to the
standalone APIs, including their refit-level failure and parallel policies.

```jldoctest suite
julia> configured = robustness_suite(
           problem,
           fit;
           in_space=InSpaceOptions(units=[:a], rmspe_cutoff=5.0),
           leave_one_out=LeaveOneOutOptions(active_only=true),
           in_time=InTimeOptions(placebo_times=[3], post_window=1),
       );

julia> length(configured.in_space.refits)
1
```

## States and failures

Every component has a [`RobustnessAnalysisState`](@ref):

- `:success` means the component API returned a result. That result can still
  contain recorded individual refit failures.
- `:disabled` means the analysis was not requested.
- `:skipped` means it was requested without required placebo times or
  specifications.
- `:failed` means the component itself threw and `suite_failure=:record`
  isolated the error.

Set `suite_failure=:error` to rethrow component-level errors. This does not
change the `on_failure` behavior inside any component.

[`diagnostics_table`](@ref) reads `suite.diagnostics` without recalculating
it. [`robustness_summary`](@ref) returns four deterministic component-status
rows. Component path, summary, inference, and plotting APIs consume the
stored fields directly; none of those reporting operations refits a model.
