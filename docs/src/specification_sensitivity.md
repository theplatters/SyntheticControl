# Specification Sensitivity

[`specification_sensitivity`](@ref) completely refits SCM under explicit
[`SCMSpecification`](@ref) objects. It uses the full balanced panel retained
by [`from_table`](@ref), preserves the estimator configuration, and never
mutates the baseline problem or fit.

## Specification dimensions

Each specification independently controls four dimensions:

- `pre_periods`: outcome observations used to fit donor weights.
- `omitted_predictors`: predictors removed before fitting all weights.
- `predictor_periods`: pre-treatment observations used to average retained predictors.
- `donors`: the donor pool.

`nothing` preserves a baseline dimension. Changing `pre_periods` does not
implicitly change predictor aggregation periods.

```jldoctest specification
julia> using SyntheticControl, CommonSolve, Tables

julia> panel = (
           unit=repeat([:treated, :a, :b, :c], inner=6),
           time=repeat(1:6, 4),
           outcome=[10.,12,15,19,25,30, 8,10,12,14,16,18,
                    12,13,16,18,20,22, 17,20,22,24,26,28],
           x=[2.,2.5,3,4,5,6, 1,1.2,1.4,1.6,1.8,2,
              3,3.2,3.6,4,4.4,4.8, 5,5.5,6,6.5,7,7.5],
           z=[9.,8,7,6,5,4, 4,4.5,5,5.5,6,6.5,
              10,9.5,9,8.5,8,7.5, 14,13,12,11,10,9],
       );

julia> problem = from_table(
           panel; unit=:unit, time=:time, outcome=:outcome,
           predictors=[:x, :z], treated=:treated, treatment_time=5,
       );

julia> baseline = solve(problem);

julia> specifications = [
           SCMSpecification(:short_pre; pre_periods=[2, 3, 4]),
           SCMSpecification(:without_x; omitted_predictors=["x"]),
           SCMSpecification(:late_predictors; predictor_periods=[3, 4]),
           SCMSpecification(:restricted; donors=[:a, :c]),
       ];

julia> sensitivity = specification_sensitivity(problem, baseline, specifications);

julia> [refit.solver_status for refit in sensitivity.refits]
4-element Vector{Symbol}:
 :success
 :success
 :success
 :success
```

Every success stores its specification, identifiers and predictors, full
solver result, full observed path, donor weights, classic predictor weights
where applicable, [`FitDiagnostics`](@ref), and
[`PredictorDiagnostics`](@ref). Penalized SCM preserves `lambda` and reports
`predictor_weights === nothing` because it does not estimate classic ``V``
weights.

Classic SCM preserves stopping thresholds and configured predictor starts.
When predictors are omitted, starts are projected onto retained predictors,
renormalized, and deduplicated deterministically. All successes are evaluated
on the same full panel, so paths remain comparable when fitting windows differ.

## Convenience grids

```jldoctest specification
julia> length(preperiod_specifications([[1, 2], [1, 2, 3]]))
2

julia> [spec.omitted_predictors for spec in leave_one_predictor_out(problem)]
2-element Vector{Vector{String}}:
 ["x"]
 ["z"]

julia> length(aggregation_period_specifications([[1, 2], [3, 4]]))
2

julia> length(donor_pool_specifications([[:a, :b], [:a, :c]]))
2
```

Generated identifiers are deterministic and can be replaced with `ids=`.

## Validation and failures

The batch rejects empty batches, duplicate identifiers, duplicate
definitions, and heterogeneous identifier types. Individual specifications
validate empty or too-short fitting windows, unknown or post-treatment
periods, removal of the last predictor, duplicate or unknown predictors, and
empty, duplicate, unknown, or treated-unit donor pools.

`minimum_pre_periods=2` is the default. With `on_failure=:record`, invalid
specifications and solver errors produce retained failed refits with
`solver_status=:failed`, `included=false`, `exclusion_reason=:failed`, and a
failure message. `on_failure=:error` rethrows immediately.

## Reporting tables

```jldoctest specification
julia> Tables.columnnames(specification_definitions(sensitivity))
(:specification_id, :pre_periods, :omitted_predictors, :predictor_periods, :donors, :solver_status, :included, :failure_reason)

julia> Tables.columnnames(specification_paths(sensitivity))
(:specification_id, :time, :actual, :synthetic, :gap, :is_post_treatment, :included, :solver_status, :failure_reason)

julia> Tables.columnnames(specification_weights(sensitivity))
(:specification_id, :donor, :weight, :included, :solver_status, :failure_reason)
```

[`specification_diagnostics`](@ref) returns one scalar diagnostic row per
specification. [`specification_balance`](@ref) returns predictor balance and
classic predictor weights. Failed specifications receive one sentinel row in
long tables and typed `missing` diagnostic values; they are never dropped.
