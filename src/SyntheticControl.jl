module SyntheticControl

using CommonSolve, LinearAlgebra
import CommonSolve: solve
using Base.Threads: @threads, nthreads
using Statistics
export SyntheticControlData, SyntheticControlProblem, SyntheticControlResult
export PenalizedSyntheticControlProblem, PenalizedSyntheticControlResult
export SyntheticControlPathData, SyntheticControlPlaceboResult
export SyntheticControlPanelData, RobustnessRefit
export InSpacePlaceboResult, LeaveOneOutResult, InTimePlaceboResult
export PointwisePlaceboInferenceResult, AggregatePlaceboInferenceResult
export SCMSpecification, SpecificationSensitivityRefit, SpecificationSensitivityResult
export InSpaceOptions, LeaveOneOutOptions, InTimeOptions, SpecificationSensitivityOptions
export RobustnessAnalysisState, RobustnessSuiteResult
export FitDiagnostics, PredictorDiagnostics
export actual_outcome, synthetic_outcome, outcome_gap, placebo_gaps
export pre_treatment_rmspe, post_treatment_rmspe, rmspe_ratio
export placebo_rmspe_ratios, filter_placebos, randomization_p_value
export in_space_placebos, leave_one_out, in_time_placebos, robustness_counts
export pointwise_placebo_inference, aggregate_placebo_inference
export specification_sensitivity
export robustness_suite
export preperiod_specifications, leave_one_predictor_out
export aggregation_period_specifications, donor_pool_specifications
export pathplot, pathplot!, gapplot, gapplot!, placeboplot, placeboplot!
export placebodistribution, placebodistribution!
export leaveoneoutplot, leaveoneoutplot!, intimeplaceboplot, intimeplaceboplot!
export from_table, weights_table, balance_table, path_table
export fit_diagnostics, predictor_diagnostics
export diagnostics_table, predictor_diagnostics_table
export placebo_summary, leave_one_out_summary, in_time_summary
export placebo_paths, leave_one_out_paths, in_time_paths
export pointwise_placebo_summary, aggregate_placebo_summary
export specification_definitions, specification_diagnostics
export specification_paths, specification_weights, specification_balance
export robustness_summary
export solve

const DEFAULT_MAX_PAIR_STARTS = 0
const DEFAULT_MAX_OUTER_ITERS = 50
const DEFAULT_MAX_OUTER_EVALUATIONS = 200
const DEFAULT_MAX_TRANSFER_DONORS = 4
const DEFAULT_MAX_TRANSFER_RECEIVERS = 8
const DEFAULT_MIN_RELATIVE_MSPE_IMPROVEMENT = 0.01

const SOLUTION_PROBLEM_REGISTRY = IdDict{Any,Any}()

"""
    _register_solution_problem!(solution, problem)

Associate a mutable solver result with the problem that produced it so
fit-only orchestration APIs can recover estimator configuration. Returns the
solution and replaces an older association for the same result object.

# Examples

```julia
isdefined(SyntheticControl, :_register_solution_problem!)
```
"""
function _register_solution_problem!(solution, problem)
  SOLUTION_PROBLEM_REGISTRY[solution] = problem
  return solution
end

"""
    _problem_for_solution(solution)

Return the registered problem that produced `solution`. Throws
`ArgumentError` for manually constructed or otherwise unregistered fits;
callers may then use an explicit `(problem, solution)` API.

# Examples

```julia
isdefined(SyntheticControl, :_problem_for_solution)
```
"""
function _problem_for_solution(solution)
  problem = get(SOLUTION_PROBLEM_REGISTRY, solution, nothing)
  problem === nothing && throw(ArgumentError(
    "the fit is not associated with a solved problem; call robustness_suite(problem, fit; ...)"
  ))
  return problem
end

"""
    SyntheticControlData(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)
    SyntheticControlData{T<:AbstractFloat}

Holds the cleanly aligned matrices and vectors used by synthetic control
estimators. Solver-specific problem types, such as the classic
`SyntheticControlProblem`, wrap this shared data interface.

`X1` is the treated-unit predictor vector with length `K`, `Y1` is the
treated-unit pre-treatment outcome vector with length `T_pre`, `X0` is the
`K × J` donor predictor matrix, and `Y0` is the `T_pre × J` donor outcome
matrix. `predictor_names` and `donor_ids` must have lengths `K` and `J`.
All numeric inputs must have the same concrete floating-point element type,
be finite, and every predictor row must have non-zero cross-unit variation.

The constructor returns a `SyntheticControlData{T}` and stores standardized
predictors in `X1_normalized` and `X0_normalized`. It throws
`DimensionMismatch` for inconsistent shapes and `ArgumentError` for
non-finite values or near-zero predictor variance. Construction has no side
effects.

# Examples

```julia
using SyntheticControl

data = SyntheticControlData(
  [1.0, 2.0],
  [10.0, 11.0, 12.0],
  [0.8 1.4 2.0; 1.6 2.2 2.8],
  [9.0 10.5 12.0; 10.0 11.5 13.0; 11.0 12.5 14.0],
  ["level", "trend"],
  ["A", "B", "C"],
  "treated",
)
size(data.X0) == (2, 3)
```
"""
struct SyntheticControlData{T<:AbstractFloat}
  # 1. Target Data (Treated Unit)
  X1::Vector{T}      # K x 1 vector of pre-treatment predictors
  X1_normalized::Vector{T}      # K x 1 vector of pre-treatment predictors
  Y1::Vector{T}      # T_pre x 1 vector of pre-treatment outcomes

  # 2. Donor Pool Data (Control Units)
  X0::Matrix{T}      # K x J matrix of predictors for J control units
  X0_normalized::Matrix{T}      # K x J matrix of predictors for J control units
  Y0::Matrix{T}      # T_pre x J matrix of outcomes for J control units

  # 3. Metadata for Tracking & Validation
  predictor_names::Vector{String}
  donor_ids::Vector{String}
  treated_id::String

  # Inner constructor for shape and dimension verification
  function SyntheticControlData(
    X1::Vector{T}, Y1::Vector{T},
    X0::Matrix{T}, Y0::Matrix{T},
    predictor_names::Vector{String}, donor_ids::Vector{String}, treated_id::String
  ) where {T<:AbstractFloat}

    K, J = size(X0)
    T_pre = length(Y1)

    length(X1) == K || throw(DimensionMismatch("X1 length must match rows of X0 (K)"))
    size(Y0, 1) == T_pre || throw(DimensionMismatch("Y0 rows must match length of Y1 (T_pre)"))
    size(Y0, 2) == J || throw(DimensionMismatch("Y0 columns must match columns of X0 (J)"))
    length(predictor_names) == K || throw(DimensionMismatch("Must provide K predictor names"))
    length(donor_ids) == J || throw(DimensionMismatch("Must provide J donor IDs"))
    all(isfinite, X1) || throw(ArgumentError("X1 contains Infs or NaNs"))
    all(isfinite, Y1) || throw(ArgumentError("Y1 contains Infs or NaNs"))
    all(isfinite, X0) || throw(ArgumentError("X0 contains Infs or NaNs"))
    all(isfinite, Y0) || throw(ArgumentError("Y0 contains Infs or NaNs"))

    std_divisor = vec(std(hcat(X0, X1), dims=2))
    all(isfinite, std_divisor) || throw(ArgumentError("Predictor standard deviations contain Infs or NaNs"))
    any(<(eps(T)), std_divisor) && throw(ArgumentError("Predictors have almost zero variance"))
    return new{T}(X1, X1 ./ std_divisor, Y1, X0, X0 ./ std_divisor, Y0, predictor_names, donor_ids, treated_id)
  end
end

include("classic_scm.jl")
include("penalized_scm.jl")
include("visualization.jl")
include("diagnostics.jl")
include("robustness.jl")
include("specification.jl")
include("suite.jl")

"""
    from_table(table; unit, time, outcome, predictors, treated, treatment_time, kwargs...)

Construct a `SyntheticControlProblem` from a Tables.jl-compatible long-format
panel after loading Tables.jl. The Tables extension materializes the source
once, aggregates predictor columns over pre-treatment rows with means, and
uses pre-treatment outcomes as `Y1` and `Y0`.

This fallback method throws unless `using Tables` has activated the package
extension. It has no side effects.

# Examples

```julia
using SyntheticControl

isdefined(SyntheticControl, :from_table)
```
"""
function from_table(args...; kwargs...)
  throw(ArgumentError("from_table requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    weights_table(result)

Return donor weights as a Tables.jl-compatible table after loading Tables.jl.
The stable schema is `donor, weight`. This fallback method throws unless
`using Tables` has activated the package extension.

# Examples

```julia
using SyntheticControl

isdefined(SyntheticControl, :weights_table)
```
"""
function weights_table(args...; kwargs...)
  throw(ArgumentError("weights_table requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    balance_table(problem, result)

Return predictor balance as a Tables.jl-compatible table after loading
Tables.jl. The stable schema is `predictor, treated, synthetic, difference`.
This fallback method throws unless `using Tables` has activated the package
extension.

# Examples

```julia
using SyntheticControl

isdefined(SyntheticControl, :balance_table)
```
"""
function balance_table(args...; kwargs...)
  throw(ArgumentError("balance_table requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    path_table(problem, result)

Return actual, synthetic, gap, and post-treatment flags as a
Tables.jl-compatible table after loading Tables.jl. The stable schema is
`time, actual, synthetic, gap, post_treatment`. Problems built by
`from_table` retain full panel paths; matrix-built problems return their
pre-treatment fitted paths. This fallback method throws unless `using Tables`
has activated the package extension.

# Examples

```julia
using SyntheticControl

isdefined(SyntheticControl, :path_table)
```
"""
function path_table(args...; kwargs...)
  throw(ArgumentError("path_table requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    diagnostics_table(fit; weight_sum_tolerance=nothing)

Return a one-row Tables.jl-compatible table of stable fit diagnostics after
loading Tables.jl. See [`fit_diagnostics`](@ref) for definitions and weight
validation. This fallback throws unless the Tables extension is active.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :diagnostics_table)
```
"""
function diagnostics_table(args...; kwargs...)
  throw(ArgumentError("diagnostics_table requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    predictor_diagnostics_table(fit; weight_sum_tolerance=nothing)

Return one Tables.jl-compatible row per predictor after loading Tables.jl.
See [`predictor_diagnostics`](@ref) for the stable statistical definitions.
This fallback throws unless the Tables extension is active.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :predictor_diagnostics_table)
```
"""
function predictor_diagnostics_table(args...; kwargs...)
  throw(ArgumentError("predictor_diagnostics_table requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    robustness_summary(suite::RobustnessSuiteResult)

Return one Tables.jl-compatible status row for each suite component after
loading Tables.jl. The fallback throws unless the Tables extension is active.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :robustness_summary)
```
"""
function robustness_summary(args...; kwargs...)
  throw(ArgumentError("robustness_summary requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    placebo_summary(result::InSpacePlaceboResult)

Return the stable Tables.jl summary for an in-space placebo result after
loading Tables.jl. The fallback throws when the Tables extension is absent.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :placebo_summary)
```
"""
function placebo_summary(args...; kwargs...)
  throw(ArgumentError("placebo_summary requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    leave_one_out_summary(result::LeaveOneOutResult)

Return the stable Tables.jl leave-one-out summary after loading Tables.jl.
The fallback throws when the Tables extension is absent.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :leave_one_out_summary)
```
"""
function leave_one_out_summary(args...; kwargs...)
  throw(ArgumentError("leave_one_out_summary requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    in_time_summary(result::InTimePlaceboResult)

Return the stable Tables.jl in-time placebo summary after loading Tables.jl.
The fallback throws when the Tables extension is absent.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :in_time_summary)
```
"""
function in_time_summary(args...; kwargs...)
  throw(ArgumentError("in_time_summary requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    placebo_paths(result::InSpacePlaceboResult)

Return the stored treated and in-space placebo paths as a long
Tables.jl-compatible table after loading Tables.jl. Successful assignments
retain every stored period; failed assignments retain one sentinel row.
This fallback throws unless the Tables extension is active and never refits.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :placebo_paths)
```
"""
function placebo_paths(args...; kwargs...)
  throw(ArgumentError("placebo_paths requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    leave_one_out_paths(result::LeaveOneOutResult)

Return stored leave-one-out refit paths as a long Tables.jl-compatible table
after loading Tables.jl. Failed assignments retain one sentinel row. The
baseline path remains available as `result.original_path`. This fallback
throws unless the Tables extension is active and never refits.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :leave_one_out_paths)
```
"""
function leave_one_out_paths(args...; kwargs...)
  throw(ArgumentError("leave_one_out_paths requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    in_time_paths(result::InTimePlaceboResult)

Return stored in-time placebo paths as a long Tables.jl-compatible table
after loading Tables.jl. Failed pseudo-dates retain one sentinel row. This
fallback throws unless the Tables extension is active and never refits.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :in_time_paths)
```
"""
function in_time_paths(args...; kwargs...)
  throw(ArgumentError("in_time_paths requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    pointwise_placebo_summary(result::PointwisePlaceboInferenceResult)

Return the stable Tables.jl pointwise-inference summary after loading
Tables.jl. This fallback throws unless the Tables extension is active and
never recalculates inference.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :pointwise_placebo_summary)
```
"""
function pointwise_placebo_summary(args...; kwargs...)
  throw(ArgumentError("pointwise_placebo_summary requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    aggregate_placebo_summary(result::AggregatePlaceboInferenceResult)

Return the stable one-row Tables.jl aggregate-inference summary after loading
Tables.jl. This fallback throws unless the Tables extension is active and
never recalculates inference.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :aggregate_placebo_summary)
```
"""
function aggregate_placebo_summary(args...; kwargs...)
  throw(ArgumentError("aggregate_placebo_summary requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end

"""
    specification_definitions(result::SpecificationSensitivityResult)

Return specification definitions as a Tables.jl-compatible table after
loading Tables.jl. This fallback throws unless the extension is active.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :specification_definitions)
```
"""
function specification_definitions(args...; kwargs...)
  throw(ArgumentError("specification_definitions requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end


"""
    specification_diagnostics(result::SpecificationSensitivityResult)

Return one diagnostic row per specification after loading Tables.jl. Failed
specifications use typed missing diagnostic values. This fallback throws
unless the extension is active.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :specification_diagnostics)
```
"""
function specification_diagnostics(args...; kwargs...)
  throw(ArgumentError("specification_diagnostics requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end


"""
    specification_paths(result::SpecificationSensitivityResult)

Return stored specification paths in long Tables.jl form after loading
Tables.jl. Failed specifications contribute one sentinel row. This fallback
throws unless the extension is active and never refits.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :specification_paths)
```
"""
function specification_paths(args...; kwargs...)
  throw(ArgumentError("specification_paths requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end


"""
    specification_weights(result::SpecificationSensitivityResult)

Return stored donor weights in long Tables.jl form after loading Tables.jl.
Failed specifications contribute one sentinel row. This fallback throws
unless the extension is active and never refits.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :specification_weights)
```
"""
function specification_weights(args...; kwargs...)
  throw(ArgumentError("specification_weights requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end


"""
    specification_balance(result::SpecificationSensitivityResult)

Return stored predictor balance and predictor weights in long Tables.jl form
after loading Tables.jl. Failed specifications contribute one sentinel row.
This fallback throws unless the extension is active and never refits.

# Examples

```julia
using SyntheticControl
isdefined(SyntheticControl, :specification_balance)
```
"""
function specification_balance(args...; kwargs...)
  throw(ArgumentError("specification_balance requires loading Tables.jl: run `using SyntheticControl, Tables`"))
end


end # module SyntheticControl
