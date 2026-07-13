const DiagnosticResultLike = Union{SyntheticControlResult,PenalizedSyntheticControlResult}

"""
    FitDiagnostics{T}

Typed scalar diagnostics for a fitted SCM. Outcome diagnostics use every row
of `fit.data.Y1` and `fit.data.Y0`, which are exactly the estimator's
pre-treatment observations. Predictor imbalance is measured on the original,
unstandardized predictor scale.

`solver_success`, `termination_status`, `iteration_count`,
`runtime_seconds`, and `objective_evaluations` are `missing` because the
current package solvers do not retain those metadata. `objective_value` is
the stored MSPE objective for classic SCM and the stored penalized objective
for penalized SCM.

# Examples

```julia
using SyntheticControl, CommonSolve
problem = SyntheticControlProblem([1.0], [1.0, 2.0], reshape([0.0, 2.0], 1, 2),
                                  [0.0 2.0; 1.0 3.0], ["x"], ["a", "b"], "t")
diagnostics = fit_diagnostics(solve(problem))
diagnostics.pre_rmspe >= 0
```
"""
struct FitDiagnostics{T<:AbstractFloat}
  pre_rmspe::T
  pre_mae::T
  max_absolute_pre_gap::T
  effective_donor_count::T
  largest_donor_weight::T
  weight_sum::T
  weight_sum_error::T
  predictor_mae::T
  predictor_max_absolute_imbalance::T
  predictor_mean_relative_imbalance::T
  predictor_max_relative_imbalance::T
  solver::Symbol
  solver_success::Union{Missing,Bool}
  termination_status::Union{Missing,Symbol}
  iteration_count::Union{Missing,Int}
  objective_value::T
  runtime_seconds::Union{Missing,T}
  objective_evaluations::Union{Missing,Int}
end

"""
    PredictorDiagnostics{T}

Typed predictor-level fit diagnostics. `difference` is treated minus
synthetic, `absolute_difference` is its absolute value, and
`relative_difference` divides that absolute difference by the absolute
treated value. Relative difference is zero for `0 / 0` and `Inf` for a
nonzero difference relative to a zero treated value.

# Examples

```julia
using SyntheticControl
data = SyntheticControlData([2.0], [1.0], reshape([1.0, 3.0], 1, 2), [0.0 2.0],
                            ["x"], ["a", "b"], "t")
fit = SyntheticControl.SyntheticControlResult(data, [0.5, 0.5], [1.0], 0.0)
predictor_diagnostics(fit).difference == [0.0]
```
"""
struct PredictorDiagnostics{T<:AbstractFloat}
  predictor::Vector{String}
  treated::Vector{T}
  synthetic::Vector{T}
  difference::Vector{T}
  absolute_difference::Vector{T}
  relative_difference::Vector{T}
end

"""
    _validate_diagnostic_weights(fit, weight_sum_tolerance)

Validate diagnostic donor weights without changing them and return the
resolved sum tolerance. Weights must be nonempty, finite, nonnegative, and
sum to one within `weight_sum_tolerance`. When the keyword is `nothing`, the
tolerance is `sqrt(eps(T))` for weight type `T`.

# Examples

```julia
data = SyntheticControlData([1.0], [1.0], reshape([0.0, 2.0], 1, 2), [0.0 2.0],
                            ["x"], ["a", "b"], "t")
fit = SyntheticControl.SyntheticControlResult(data, [0.5, 0.5], [1.0], 0.0)
SyntheticControl._validate_diagnostic_weights(fit, nothing) > 0
```
"""
function _validate_diagnostic_weights(fit::DiagnosticResultLike, weight_sum_tolerance)
  weights = fit.W
  T = eltype(weights)
  isempty(weights) && throw(ArgumentError("diagnostic donor weights must not be empty"))
  all(isfinite, weights) || throw(ArgumentError("diagnostic donor weights must be finite"))
  all(>=(zero(T)), weights) || throw(ArgumentError("diagnostic donor weights must be nonnegative"))
  tolerance = weight_sum_tolerance === nothing ? sqrt(eps(T)) : T(weight_sum_tolerance)
  isfinite(tolerance) && tolerance >= zero(T) ||
    throw(ArgumentError("weight_sum_tolerance must be finite and nonnegative"))
  abs(sum(weights) - one(T)) <= tolerance ||
    throw(ArgumentError("diagnostic donor weights must sum to one within tolerance $tolerance"))
  return tolerance
end

"""
    _relative_predictor_difference(difference, treated)

Return `abs(difference) / abs(treated)`, with `0 / 0` defined as zero and a
nonzero difference over a zero treated value defined as `Inf`.

# Examples

```julia
SyntheticControl._relative_predictor_difference(1.0, 0.0) == Inf
```
"""
function _relative_predictor_difference(difference::T, treated::T) where {T<:AbstractFloat}
  treated == zero(T) && return difference == zero(T) ? zero(T) : T(Inf)
  return abs(difference) / abs(treated)
end

"""
    predictor_diagnostics(fit; weight_sum_tolerance=nothing)

Calculate predictor-level diagnostics from a classic or penalized SCM fit.
The returned [`PredictorDiagnostics`](@ref) uses original predictor units and
raw, unnormalized donor weights. Weights are validated as described by
[`fit_diagnostics`](@ref). Throws `ArgumentError` for invalid weights or
non-finite calculated predictor values and does not mutate `fit`.

# Examples

```julia
using SyntheticControl
data = SyntheticControlData([2.0], [1.0], reshape([1.0, 3.0], 1, 2), [0.0 2.0],
                            ["x"], ["a", "b"], "t")
fit = SyntheticControl.SyntheticControlResult(data, [0.5, 0.5], [1.0], 0.0)
predictor_diagnostics(fit).synthetic == [2.0]
```
"""
function predictor_diagnostics(fit::DiagnosticResultLike; weight_sum_tolerance=nothing)
  _validate_diagnostic_weights(fit, weight_sum_tolerance)
  data = fit.data
  isempty(data.X1) && throw(ArgumentError("at least one predictor is required for diagnostics"))
  synthetic = data.X0 * fit.W
  all(isfinite, synthetic) || throw(ArgumentError("synthetic predictor values must be finite"))
  treated = copy(data.X1)
  difference = treated .- synthetic
  all(isfinite, difference) || throw(ArgumentError("predictor differences must be finite"))
  absolute_difference = abs.(difference)
  relative_difference = similar(difference)
  for index in eachindex(relative_difference, difference, treated)
    relative_difference[index] = _relative_predictor_difference(difference[index], treated[index])
  end
  return PredictorDiagnostics(
    copy(data.predictor_names), treated, synthetic, difference,
    absolute_difference, relative_difference,
  )
end

"""
    fit_diagnostics(fit; weight_sum_tolerance=nothing)

Calculate stable scalar diagnostics for a classic or penalized SCM fit.
Every `Y1`/`Y0` row is pre-treatment. Missing values cannot occur in package
fit arrays; non-finite inputs, derived paths, and weights are rejected.

Weights are never normalized. They must be nonempty, finite, nonnegative,
and sum to one within `sqrt(eps(T))` by default, or within the supplied
nonnegative `weight_sum_tolerance`. Effective donor count is computed from
the raw accepted weights as `1 / sum(abs2, W)`.

The result includes pre-treatment RMSPE, MAE, maximum absolute gap, donor
concentration, aggregate absolute and relative predictor imbalance, and
available solver metadata. Metadata not retained by a supported solver is
represented as `missing`. Throws `ArgumentError` for invalid diagnostic
inputs and does not mutate `fit`.

# Examples

```julia
using SyntheticControl
data = SyntheticControlData([2.0], [1.0, 3.0], reshape([1.0, 3.0], 1, 2),
                            [0.0 2.0; 2.0 4.0], ["x"], ["a", "b"], "t")
fit = SyntheticControl.SyntheticControlResult(data, [0.5, 0.5], [1.0], 0.0)
fit_diagnostics(fit).pre_rmspe == 0.0
```
"""
function fit_diagnostics(fit::DiagnosticResultLike; weight_sum_tolerance=nothing)
  _validate_diagnostic_weights(fit, weight_sum_tolerance)
  data = fit.data
  actual = data.Y1
  isempty(actual) && throw(ArgumentError("at least one pre-treatment observation is required for diagnostics"))
  synthetic = data.Y0 * fit.W
  all(isfinite, synthetic) || throw(ArgumentError("synthetic pre-treatment outcomes must be finite"))
  gap = outcome_gap(actual, synthetic)
  all(isfinite, gap) || throw(ArgumentError("pre-treatment gaps must be finite"))

  predictor = predictor_diagnostics(fit; weight_sum_tolerance=weight_sum_tolerance)
  weights = fit.W
  T = eltype(weights)
  weight_sum = sum(weights)
  sum_squared_weights = sum(abs2, weights)
  sum_squared_weights > zero(T) || throw(ArgumentError("at least one diagnostic donor weight must be positive"))
  objective_value = fit isa SyntheticControlResult ? fit.mspe : fit.objective
  isfinite(objective_value) || throw(ArgumentError("stored solver objective must be finite"))

  return FitDiagnostics(
    _rmspe(gap),
    sum(abs, gap) / T(length(gap)),
    maximum(abs, gap),
    one(T) / sum_squared_weights,
    maximum(weights),
    weight_sum,
    abs(weight_sum - one(T)),
    sum(predictor.absolute_difference) / T(length(predictor.absolute_difference)),
    maximum(predictor.absolute_difference),
    sum(predictor.relative_difference) / T(length(predictor.relative_difference)),
    maximum(predictor.relative_difference),
    fit isa SyntheticControlResult ? :classic : :penalized,
    missing,
    missing,
    missing,
    objective_value,
    missing,
    missing,
  )
end
