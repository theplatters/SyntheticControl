"""
    SyntheticControlPanelData(time, treatment_time, unit_ids, treated,
                              outcomes, predictors, predictor_names)
    SyntheticControlPanelData{T,UI,TT}

Validated balanced-panel storage used by robustness refits. `outcomes` is a
`periods × units` matrix and `predictors` is a
`periods × predictors × units` array. Times and unit identifiers retain their
input types, must be unique and deterministically ordered, and
`treatment_time` is the first post-treatment observation. Numeric data must
be finite. Construction copies all inputs and has no side effects.

# Examples

```julia
panel = SyntheticControlPanelData(
  1:4, 3, [:treated, :a, :b], :treated,
  [2.0 1.0 3.0; 3.0 2.0 4.0; 5.0 3.0 5.0; 7.0 4.0 6.0],
  reshape([2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0,
           3.0, 4.0, 5.0, 6.0], 4, 1, 3),
  ["x"],
)
panel.treatment_index == 3
```
"""
struct SyntheticControlPanelData{T<:AbstractFloat,UI,TT}
  time::Vector{TT}
  treatment_index::Int
  treatment_time::TT
  unit_ids::Vector{UI}
  treated_index::Int
  outcomes::Matrix{T}
  predictors::Array{T,3}
  predictor_names::Vector{String}

  function SyntheticControlPanelData(
    time::AbstractVector{TT},
    treatment_time::TT,
    unit_ids::AbstractVector{UI},
    treated,
    outcomes::AbstractMatrix{T},
    predictors::AbstractArray{T,3},
    predictor_names::AbstractVector{<:AbstractString}
  ) where {T<:AbstractFloat,UI,TT}
    resolved_time = collect(time)
    resolved_units = collect(unit_ids)
    !isempty(resolved_time) || throw(ArgumentError("time must contain at least one observation"))
    length(unique(resolved_time)) == length(resolved_time) || throw(ArgumentError("time values must be unique"))
    issorted(resolved_time) || throw(ArgumentError("time values must be sorted"))
    length(unique(resolved_units)) == length(resolved_units) || throw(ArgumentError("unit identifiers must be unique"))
    treated_index = findfirst(isequal(treated), resolved_units)
    treated_index === nothing && throw(ArgumentError("treated unit is absent from unit_ids"))
    treatment_index = findfirst(isequal(treatment_time), resolved_time)
    treatment_index === nothing && throw(ArgumentError("treatment_time must be an observed time value"))
    treatment_index > 1 || throw(ArgumentError("at least one pre-treatment observation is required"))
    size(outcomes) == (length(resolved_time), length(resolved_units)) ||
      throw(DimensionMismatch("outcomes must have size periods × units"))
    size(predictors, 1) == length(resolved_time) || throw(DimensionMismatch("predictor periods must match time"))
    size(predictors, 2) == length(predictor_names) || throw(DimensionMismatch("predictor columns must match predictor_names"))
    size(predictors, 3) == length(resolved_units) || throw(DimensionMismatch("predictor units must match unit_ids"))
    !isempty(predictor_names) || throw(ArgumentError("at least one predictor is required"))
    _assert_finite_outcomes("panel outcomes", outcomes)
    _assert_finite_outcomes("panel predictors", predictors)
    return new{T,UI,TT}(
      resolved_time,
      treatment_index,
      treatment_time,
      resolved_units,
      treated_index,
      Matrix(outcomes),
      Array(predictors),
      String.(predictor_names)
    )
  end
end

const ROBUSTNESS_PANEL_DATA = IdDict{SyntheticControlData,Any}()

function _register_panel_data!(data::SyntheticControlData, panel::SyntheticControlPanelData)
  ROBUSTNESS_PANEL_DATA[data] = panel
  return data
end

function _panel_data(problem)
  panel = get(ROBUSTNESS_PANEL_DATA, problem.data, nothing)
  panel === nothing && throw(ArgumentError(
    "robustness refits require full panel data; construct the problem with from_table"
  ))
  return panel
end

"""
    RobustnessRefit{T,L,UI,TT}

One attempted robustness assignment. It records its assignment label, donor
pool, effective time path, complete solver result when successful, outcome
and gap paths, RMSPE statistics, solver status, failure reason, and inference
inclusion state. Failed fits use empty paths and `NaN` statistics rather than
being dropped.

# Examples

```julia
isdefined(SyntheticControl, :RobustnessRefit)
```
"""
mutable struct RobustnessRefit{T<:AbstractFloat,L,UI,TT}
  assignment::L
  donor_ids::Vector{UI}
  time::Vector{TT}
  treatment_index::Int
  result::Union{Nothing,SyntheticControlResult,PenalizedSyntheticControlResult}
  actual::Vector{T}
  synthetic::Vector{T}
  gap::Vector{T}
  pre_rmspe::T
  post_rmspe::T
  rmspe_ratio::T
  mean_post_gap::T
  cumulative_post_gap::T
  max_path_deviation::T
  solver_status::Symbol
  failure_reason::Union{Nothing,String}
  included::Bool
  exclusion_reason::Union{Nothing,Symbol}
end

"""
    InSpacePlaceboResult{T,UI,TT}

Result of complete SCM refits assigning treatment to eligible donor units.
`treated` summarizes the original fit, while `refits` retains every selected
assignment, including failures and filtered fits. `p_value` is `nothing` if
the treated ratio is non-finite or no placebo assignment is eligible.

# Examples

```julia
isdefined(SyntheticControl, :InSpacePlaceboResult)
```
"""
struct InSpacePlaceboResult{T<:AbstractFloat,UI,TT}
  original::Union{SyntheticControlResult,PenalizedSyntheticControlResult}
  panel::SyntheticControlPanelData{T,UI,TT}
  treated::RobustnessRefit
  refits::Vector{RobustnessRefit}
  include_treated::Bool
  rmspe_cutoff::Union{Nothing,T}
  p_value::Union{Nothing,T}
end

"""
    LeaveOneOutResult{T,UI,TT}

Result of removing selected donors one at a time and completely refitting the
treated SCM. `refits` remains in original donor order and retains failures.
This object contains no p-value because leave-one-out paths are sensitivity
checks, not independent randomization assignments.

# Examples

```julia
isdefined(SyntheticControl, :LeaveOneOutResult)
```
"""
struct LeaveOneOutResult{T<:AbstractFloat,UI,TT}
  original::Union{SyntheticControlResult,PenalizedSyntheticControlResult}
  panel::SyntheticControlPanelData{T,UI,TT}
  original_path::RobustnessRefit
  refits::Vector{RobustnessRefit}
  weight_tol::T
  active_only::Bool
end

"""
    InTimePlaceboResult{T,UI,TT}

Result of assigning pseudo-treatment dates before the real intervention.
Every attempted date records its effective fitting and evaluation periods,
complete fit, statistics, or failure. In-time placebos are diagnostics and
do not carry a randomization p-value.

# Examples

```julia
isdefined(SyntheticControl, :InTimePlaceboResult)
```
"""
struct InTimePlaceboResult{T<:AbstractFloat,UI,TT}
  original::Union{SyntheticControlResult,PenalizedSyntheticControlResult}
  panel::SyntheticControlPanelData{T,UI,TT}
  refits::Vector{RobustnessRefit}
  stop_before_treatment::Bool
  minimum_pre_periods::Int
  minimum_post_periods::Int
end

"""
    PointwisePlaceboInferenceResult{T,UI,TT}

Typed finite-sample inference results for individual post-treatment periods.
`source` retains every original assignment and exclusion record;
`included_assignments` identifies the placebo columns of
`placebo_statistics`. `assignment_count` includes the treated assignment.
These p-values are pointwise and do not provide simultaneous inference.

# Examples

```julia
isdefined(SyntheticControl, :PointwisePlaceboInferenceResult)
```
"""
struct PointwisePlaceboInferenceResult{T<:AbstractFloat,UI,TT}
  source::InSpacePlaceboResult{T,UI,TT}
  statistic::Symbol
  alternative::Symbol
  time::Vector{TT}
  treated_statistic::Vector{T}
  placebo_statistics::Matrix{T}
  included_assignments::Vector{UI}
  extreme_count::Vector{Int}
  assignment_count::Int
  p_value::Vector{T}
end

"""
    AggregatePlaceboInferenceResult{T,UI,TT,S}

Typed finite-sample inference for one post-treatment aggregate. `source`
retains all attempted and excluded assignments. `specification` is the
requested statistic symbol or callable, `statistic` is its stable reporting
name, and `periods` is the selected observed-time window.

# Examples

```julia
isdefined(SyntheticControl, :AggregatePlaceboInferenceResult)
```
"""
struct AggregatePlaceboInferenceResult{T<:AbstractFloat,UI,TT,S}
  source::InSpacePlaceboResult{T,UI,TT}
  specification::S
  statistic::Symbol
  alternative::Symbol
  periods::Vector{TT}
  treated_statistic::T
  placebo_statistics::Vector{T}
  included_assignments::Vector{UI}
  extreme_count::Int
  assignment_count::Int
  p_value::T
end

"""
    _validate_placebo_alternative(alternative)

Validate and return one of `:greater`, `:less`, or `:two_sided`.

# Examples

```julia
SyntheticControl._validate_placebo_alternative(:greater) == :greater
```
"""
function _validate_placebo_alternative(alternative::Symbol)
  alternative in (:greater, :less, :two_sided) ||
    throw(ArgumentError("alternative must be :greater, :less, or :two_sided"))
  return alternative
end

"""
    _validate_inference_path(refit, label)

Validate a successful stored robustness path for inference and return it.
The path must have aligned time, actual, synthetic, and gap vectors and a
valid first post-treatment index. No values are recalculated.

# Examples

```julia
isdefined(SyntheticControl, :_validate_inference_path)
```
"""
function _validate_inference_path(refit::RobustnessRefit, label::AbstractString)
  refit.solver_status === :success || throw(ArgumentError("$label must be a successful stored fit"))
  path_length = length(refit.time)
  length(refit.actual) == path_length || throw(DimensionMismatch("$label actual path must match time"))
  length(refit.synthetic) == path_length || throw(DimensionMismatch("$label synthetic path must match time"))
  length(refit.gap) == path_length || throw(DimensionMismatch("$label gap path must match time"))
  1 <= refit.treatment_index <= path_length ||
    throw(ArgumentError("$label treatment_index must identify the first post-treatment period"))
  return refit
end

"""
    _eligible_placebo_inference_refits(result)

Return stored placebo refits eligible for time-specific inference. Failed and
filtered assignments and assignments with non-finite gaps are excluded.
Every otherwise eligible placebo must have the treated assignment's exact
time vector and treatment index. Throws when the treated path is invalid or
no placebo remains.

# Examples

```julia
isdefined(SyntheticControl, :_eligible_placebo_inference_refits)
```
"""
function _eligible_placebo_inference_refits(result::InSpacePlaceboResult)
  treated = _validate_inference_path(result.treated, "treated assignment")
  all(isfinite, treated.gap) || throw(ArgumentError("treated gaps must be finite for inference"))
  eligible = RobustnessRefit[]
  for refit in result.refits
    refit.solver_status === :success || continue
    refit.included || continue
    _validate_inference_path(refit, "placebo assignment $(refit.assignment)")
    refit.time == treated.time ||
      throw(ArgumentError("included placebo assignment $(refit.assignment) does not share the treated time window"))
    refit.treatment_index == treated.treatment_index ||
      throw(ArgumentError("included placebo assignment $(refit.assignment) does not share the treated treatment index"))
    all(isfinite, refit.gap) || continue
    push!(eligible, refit)
  end
  isempty(eligible) && throw(ArgumentError("no eligible placebo assignments remain for inference"))
  return eligible
end

"""
    _placebo_extreme_count(placebo_values, treated_value, alternative)

Count the treated assignment plus placebo statistics at least as extreme as
the treated statistic. `:greater` uses `>=`, `:less` uses `<=`, and
`:two_sided` compares absolute values with `>=`, so ties always count.

# Examples

```julia
SyntheticControl._placebo_extreme_count([1.0, 2.0], 2.0, :greater) == 2
```
"""
function _placebo_extreme_count(placebo_values, treated_value, alternative::Symbol)
  alternative = _validate_placebo_alternative(alternative)
  if alternative === :greater
    return 1 + count(value -> value >= treated_value, placebo_values)
  elseif alternative === :less
    return 1 + count(value -> value <= treated_value, placebo_values)
  end
  threshold = abs(treated_value)
  return 1 + count(value -> abs(value) >= threshold, placebo_values)
end

"""
    _pointwise_statistic(gap, statistic)

Return `gap` for `statistic=:gap` or `abs(gap)` for
`statistic=:absolute_gap`. Throws for unsupported statistics.

# Examples

```julia
SyntheticControl._pointwise_statistic(-2.0, :absolute_gap) == 2.0
```
"""
function _pointwise_statistic(gap::T, statistic::Symbol) where {T<:AbstractFloat}
  statistic === :gap && return gap
  statistic === :absolute_gap && return abs(gap)
  throw(ArgumentError("statistic must be :gap or :absolute_gap"))
end

"""
    pointwise_placebo_inference(result; statistic=:gap, alternative=:greater)

Calculate exact finite-sample placebo p-values for every post-treatment
period stored in an [`InSpacePlaceboResult`](@ref). `statistic=:gap` ranks
signed gaps and `:absolute_gap` ranks absolute gaps. Alternatives are
`:greater`, `:less`, and `:two_sided`; two-sided inference ranks the absolute
value of the selected statistic.

The denominator contains the treated assignment and every successful,
inference-included placebo with the identical evaluation window and finite
gaps. Ties count as extreme. Failed, filtered, and non-finite placebos remain
stored in `result` but are excluded. The minimum p-value is
`1 / assignment_count`. Returned p-values are pointwise, not multiplicity
adjusted or simultaneous. Throws `ArgumentError` when the inference set or
stored windows are invalid and does not mutate or refit `result`.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=4), time=repeat(1:4, 3),
         y=[2.,3,5,7, 1,2,3,4, 3,4,5,6], x=[2.,3,4,5, 1,2,3,4, 3,4,5,6])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=3)
inference = pointwise_placebo_inference(in_space_placebos(problem, solve(problem)))
length(inference.p_value) == 2
```
"""
function pointwise_placebo_inference(
  result::InSpacePlaceboResult{T,UI,TT};
  statistic::Symbol=:gap,
  alternative::Symbol=:greater
) where {T,UI,TT}
  statistic in (:gap, :absolute_gap) || throw(ArgumentError("statistic must be :gap or :absolute_gap"))
  alternative = _validate_placebo_alternative(alternative)
  treated = result.treated
  eligible = _eligible_placebo_inference_refits(result)
  post_indices = collect(treated.treatment_index:length(treated.time))
  time = collect(treated.time[post_indices])
  treated_statistic = T[_pointwise_statistic(treated.gap[index], statistic) for index in post_indices]
  placebo_statistics = Matrix{T}(undef, length(post_indices), length(eligible))
  for (column, refit) in pairs(eligible)
    for (row, index) in pairs(post_indices)
      placebo_statistics[row, column] = _pointwise_statistic(refit.gap[index], statistic)
    end
  end
  assignment_count = 1 + length(eligible)
  extreme_count = Vector{Int}(undef, length(post_indices))
  p_value = Vector{T}(undef, length(post_indices))
  for row in eachindex(post_indices)
    extreme_count[row] = _placebo_extreme_count(
      @view(placebo_statistics[row, :]), treated_statistic[row], alternative,
    )
    p_value[row] = T(extreme_count[row] / assignment_count)
  end
  return PointwisePlaceboInferenceResult(
    result, statistic, alternative, time, treated_statistic, placebo_statistics,
    UI[refit.assignment for refit in eligible], extreme_count, assignment_count, p_value,
  )
end

"""
    _selected_aggregate_period_indices(treated, periods)

Resolve an aggregate window to unique post-treatment indices in observed-time
order. `periods=nothing` selects every post-treatment period. Explicit values
must be known post-treatment observations and need not be consecutive.

# Examples

```julia
isdefined(SyntheticControl, :_selected_aggregate_period_indices)
```
"""
function _selected_aggregate_period_indices(treated::RobustnessRefit, periods)
  post_indices = collect(treated.treatment_index:length(treated.time))
  periods === nothing && return post_indices
  selected = collect(periods)
  isempty(selected) && throw(ArgumentError("periods must select at least one post-treatment observation"))
  length(unique(selected)) == length(selected) || throw(ArgumentError("periods contains duplicates"))
  indices = Int[]
  for index in post_indices
    treated.time[index] in selected && push!(indices, index)
  end
  length(indices) == length(selected) ||
    throw(ArgumentError("periods must contain only observed post-treatment values"))
  return indices
end

"""
    _aggregate_placebo_statistic(gaps, times, statistic, T)

Evaluate a predefined or callable aggregate and return a finite value of type
`T`. Predefined symbols are `:mean_gap`, `:cumulative_gap`,
`:mean_absolute_gap`, and `:rmspe`. A callable receives copies of `(gaps,
times)` when applicable, otherwise a copy of `gaps`, so it cannot mutate
stored result paths.

# Examples

```julia
SyntheticControl._aggregate_placebo_statistic([1.0, 3.0], [2, 5], :mean_gap, Float64) == 2.0
```
"""
function _aggregate_placebo_statistic(gaps, times, statistic, ::Type{T}) where {T<:AbstractFloat}
  value = if statistic === :mean_gap
    sum(gaps) / T(length(gaps))
  elseif statistic === :cumulative_gap
    sum(gaps)
  elseif statistic === :mean_absolute_gap
    sum(abs, gaps) / T(length(gaps))
  elseif statistic === :rmspe
    _rmspe(gaps)
  elseif !(statistic isa Symbol)
    gap_copy = collect(gaps)
    time_copy = collect(times)
    if applicable(statistic, gap_copy, time_copy)
      statistic(gap_copy, time_copy)
    elseif applicable(statistic, gap_copy)
      statistic(gap_copy)
    else
      throw(ArgumentError("custom aggregate statistic must accept gaps or gaps and times"))
    end
  else
    throw(ArgumentError("statistic must be :mean_gap, :cumulative_gap, :mean_absolute_gap, :rmspe, or a callable"))
  end
  value isa Real || throw(ArgumentError("aggregate statistic must return a real scalar"))
  converted = T(value)
  isfinite(converted) || throw(ArgumentError("aggregate statistic must return a finite scalar"))
  return converted
end

"""
    aggregate_placebo_inference(result; statistic=:mean_gap, periods=nothing,
                                alternative=:greater)

Calculate exact finite-sample placebo inference for one post-treatment
aggregate. Predefined statistics are `:mean_gap`, `:cumulative_gap`,
`:mean_absolute_gap`, and `:rmspe`. A callable may accept `(gaps, times)` or
`gaps`; it receives copies. `periods` selects observed post-treatment values
without assuming consecutive or integer time.

The treated assignment and all eligible placebo assignments use the same
selected window. Alternatives and ties follow
[`pointwise_placebo_inference`](@ref). Failed, filtered, non-finite, and
incomparable assignments do not silently enter the denominator; mismatched
stored windows throw. Returns an [`AggregatePlaceboInferenceResult`](@ref),
does not refit, and does not mutate stored paths.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=4), time=repeat(1:4, 3),
         y=[2.,3,5,7, 1,2,3,4, 3,4,5,6], x=[2.,3,4,5, 1,2,3,4, 3,4,5,6])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=3)
inference = aggregate_placebo_inference(
  in_space_placebos(problem, solve(problem)); statistic=:cumulative_gap,
)
0 < inference.p_value <= 1
```
"""
function aggregate_placebo_inference(
  result::InSpacePlaceboResult{T,UI,TT};
  statistic=:mean_gap,
  periods=nothing,
  alternative::Symbol=:greater
) where {T,UI,TT}
  alternative = _validate_placebo_alternative(alternative)
  treated = result.treated
  eligible = _eligible_placebo_inference_refits(result)
  indices = _selected_aggregate_period_indices(treated, periods)
  selected_times = collect(treated.time[indices])
  treated_statistic = _aggregate_placebo_statistic(
    @view(treated.gap[indices]), selected_times, statistic, T,
  )
  placebo_statistics = Vector{T}(undef, length(eligible))
  for (position, refit) in pairs(eligible)
    placebo_statistics[position] = _aggregate_placebo_statistic(
      @view(refit.gap[indices]), selected_times, statistic, T,
    )
  end
  extreme_count = _placebo_extreme_count(placebo_statistics, treated_statistic, alternative)
  assignment_count = 1 + length(eligible)
  statistic_name = statistic isa Symbol ? statistic : :custom
  return AggregatePlaceboInferenceResult(
    result, statistic, statistic_name, alternative, selected_times, treated_statistic,
    placebo_statistics, UI[refit.assignment for refit in eligible], extreme_count,
    assignment_count, T(extreme_count / assignment_count),
  )
end


function _validate_original(problem, result)
  result.data === problem.data || throw(ArgumentError("problem and solution must reference the same SyntheticControlData"))
  all(isfinite, result.W) || throw(ArgumentError("solution donor weights must be finite"))
  isfinite(result.mspe) || throw(ArgumentError("solution appears unsolved or has non-finite MSPE"))
  return nothing
end

function _validate_failure_policy(on_failure::Symbol)
  on_failure in (:record, :error) || throw(ArgumentError("on_failure must be :record or :error"))
  return on_failure
end

function _selection_indices(values, selected, label::AbstractString)
  chosen = selected === nothing ? collect(values) : collect(selected)
  isempty(chosen) && throw(ArgumentError("$label selection must not be empty"))
  length(unique(chosen)) == length(chosen) || throw(ArgumentError("$label selection contains duplicates"))
  indices = Int[]
  for value in chosen
    index = findfirst(isequal(value), values)
    index === nothing && throw(ArgumentError("unknown $label identifier: $value"))
    push!(indices, index)
  end
  return indices
end

function _data_for_assignment(panel, target_index::Int, donor_indices::Vector{Int}, pre_indices::Vector{Int})
  isempty(donor_indices) && throw(ArgumentError("donor pool is empty after reassignment or omission"))
  target_index in donor_indices && throw(ArgumentError("a treated or placebo unit cannot appear in its own donor pool"))
  isempty(pre_indices) && throw(ArgumentError("at least one fitting period is required"))
  T = eltype(panel.outcomes)
  K = length(panel.predictor_names)
  J = length(donor_indices)
  X1 = Vector{T}(undef, K)
  X0 = Matrix{T}(undef, K, J)
  for predictor in 1:K
    X1[predictor] = sum(panel.predictors[pre_indices, predictor, target_index]) / T(length(pre_indices))
    for (column, donor_index) in pairs(donor_indices)
      X0[predictor, column] = sum(panel.predictors[pre_indices, predictor, donor_index]) / T(length(pre_indices))
    end
  end
  Y1 = collect(panel.outcomes[pre_indices, target_index])
  Y0 = Matrix(panel.outcomes[pre_indices, donor_indices])
  donor_ids = string.(panel.unit_ids[donor_indices])
  return SyntheticControlData(X1, Y1, X0, Y0, panel.predictor_names, donor_ids, string(panel.unit_ids[target_index]))
end

function _problem_like(template::SyntheticControlProblem{T}, data::SyntheticControlData{T}) where {T}
  problem = SyntheticControlProblem(
    data;
    target_mspe=template.target_mspe,
    min_relative_mspe_improvement=template.min_relative_mspe_improvement
  )
  problem.starts = copy(template.starts)
  problem.nstarts = template.nstarts
  problem.inner_caches = [InnerWeightCache(data) for _ in 1:template.nstarts]
  problem.search_results = OuterSearchResults(
    Matrix{T}(undef, length(data.predictor_names), template.nstarts),
    Matrix{T}(undef, length(data.donor_ids), template.nstarts),
    Vector{T}(undef, template.nstarts)
  )
  return problem
end

function _problem_like(template::PenalizedSyntheticControlProblem{T}, data::SyntheticControlData{T}) where {T}
  return PenalizedSyntheticControlProblem(data; lambda=template.lambda)
end

function _successful_refit(
  problem,
  panel::SyntheticControlPanelData{T,UI,TT},
  assignment,
  target_index::Int,
  donor_indices::Vector{Int},
  pre_indices::Vector{Int},
  post_indices::Vector{Int};
  original_synthetic=nothing
) where {T,UI,TT}
  data = _data_for_assignment(panel, target_index, donor_indices, pre_indices)
  refit_problem = _problem_like(problem, data)
  fit = solve(refit_problem)
  all(isfinite, fit.W) || throw(ErrorException("solver returned non-finite donor weights"))
  isfinite(fit.mspe) || throw(ErrorException("solver returned non-finite MSPE"))
  path_indices = vcat(pre_indices, post_indices)
  actual = collect(panel.outcomes[path_indices, target_index])
  synthetic = collect(panel.outcomes[path_indices, donor_indices] * fit.W)
  gap = outcome_gap(actual, synthetic)
  treatment_index = length(pre_indices) + 1
  pre = pre_treatment_rmspe(gap, treatment_index)
  post = post_treatment_rmspe(gap, treatment_index)
  ratio = rmspe_ratio(pre, post)
  post_gap = @view gap[treatment_index:end]
  maximum_deviation = original_synthetic === nothing ? T(NaN) : maximum(abs.(synthetic .- original_synthetic[path_indices]))
  return RobustnessRefit{T,typeof(assignment),UI,TT}(
    assignment,
    collect(panel.unit_ids[donor_indices]),
    collect(panel.time[path_indices]),
    treatment_index,
    fit,
    actual,
    synthetic,
    gap,
    pre,
    post,
    ratio,
    sum(post_gap) / T(length(post_gap)),
    sum(post_gap),
    maximum_deviation,
    :success,
    nothing,
    true,
    nothing
  )
end

function _failed_refit(
  panel::SyntheticControlPanelData{T,UI,TT}, assignment, donor_indices, pre_indices, post_indices, reason
) where {T,UI,TT}
  path_indices = vcat(pre_indices, post_indices)
  return RobustnessRefit{T,typeof(assignment),UI,TT}(
    assignment,
    collect(panel.unit_ids[donor_indices]),
    collect(panel.time[path_indices]),
    length(pre_indices) + 1,
    nothing,
    T[], T[], T[],
    T(NaN), T(NaN), T(NaN), T(NaN), T(NaN), T(NaN),
    :failed,
    sprint(showerror, reason),
    false,
    :failed
  )
end

function _attempt_refit(problem, panel, assignment, target, donors, pre, post, on_failure; original_synthetic=nothing)
  try
    return _successful_refit(
      problem, panel, assignment, target, donors, pre, post;
      original_synthetic=original_synthetic
    )
  catch err
    on_failure === :error && rethrow()
    return _failed_refit(panel, assignment, donors, pre, post, err)
  end
end

function _batch_refits(operation, count::Int, parallel::Bool)
  refits = Vector{RobustnessRefit}(undef, count)
  if parallel && nthreads() > 1
    @threads for index in 1:count
      refits[index] = operation(index)
    end
  else
    for index in 1:count
      refits[index] = operation(index)
    end
  end
  return refits
end

function _original_refit(problem, result, panel)
  target = panel.treated_index
  donors = [index for index in eachindex(panel.unit_ids) if index != target]
  length(donors) == length(result.W) || throw(DimensionMismatch("panel donor order does not match solution weights"))
  actual = collect(panel.outcomes[:, target])
  synthetic = collect(panel.outcomes[:, donors] * result.W)
  gap = outcome_gap(actual, synthetic)
  pre = pre_treatment_rmspe(gap, panel.treatment_index)
  post = post_treatment_rmspe(gap, panel.treatment_index)
  post_gap = @view gap[panel.treatment_index:end]
  T = eltype(panel.outcomes)
  UI = eltype(panel.unit_ids)
  TT = eltype(panel.time)
  return RobustnessRefit{T,UI,UI,TT}(
    panel.unit_ids[target], collect(panel.unit_ids[donors]), copy(panel.time), panel.treatment_index,
    result, actual, synthetic, gap, pre, post, rmspe_ratio(pre, post),
    sum(post_gap) / T(length(post_gap)), sum(post_gap), zero(T),
    :success, nothing, true, nothing
  )
end

"""
    in_space_placebos(problem, solution; units=nothing, include_treated=false,
                      rmspe_cutoff=nothing, on_failure=:record, parallel=false)
    in_space_placebos(problem; kwargs...)

Completely refit the SCM after assigning treatment to each selected donor.
The placebo is excluded from its own pool; the originally treated unit is
excluded by default and is included only with `include_treated=true`.
`rmspe_cutoff=c` retains inference assignments whose pre-treatment RMSPE is
at most `c` times the treated RMSPE. Failures are retained with
`on_failure=:record` or thrown immediately with `:error`.

The exact upper-tail p-value includes the treated unit, counts ties with
`>=`, and excludes failed, filtered, and non-finite placebo ratios. It is
`nothing` when no placebo remains or the treated ratio is non-finite.
`parallel=true` uses independent problem caches and preserves selection
order; the package's solvers are deterministic and expose no RNG setting.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=4), time=repeat(1:4, 3),
         y=[2.,3,5,7, 1,2,3,4, 3,4,5,6],
         x=[2.,3,4,5, 1,2,3,4, 3,4,5,6])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=3)
placebos = in_space_placebos(problem, solve(problem))
length(placebos.refits) == 2
```
"""
function in_space_placebos(
  problem,
  solution;
  units=nothing,
  include_treated::Bool=false,
  rmspe_cutoff=nothing,
  on_failure::Symbol=:record,
  parallel::Bool=false
)
  _validate_original(problem, solution)
  _validate_failure_policy(on_failure)
  panel = _panel_data(problem)
  cutoff = rmspe_cutoff === nothing ? nothing : convert(eltype(panel.outcomes), rmspe_cutoff)
  cutoff === nothing || (isfinite(cutoff) && cutoff >= zero(cutoff)) ||
    throw(ArgumentError("rmspe_cutoff must be finite and non-negative"))
  eligible = [index for index in eachindex(panel.unit_ids) if index != panel.treated_index]
  chosen_positions = _selection_indices(panel.unit_ids[eligible], units, "placebo unit")
  chosen = eligible[chosen_positions]
  pre = collect(1:(panel.treatment_index - 1))
  post = collect(panel.treatment_index:length(panel.time))
  refits = _batch_refits(length(chosen), parallel && on_failure === :record) do position
    target = chosen[position]
    donors = [index for index in eachindex(panel.unit_ids) if index != target && (include_treated || index != panel.treated_index)]
    _attempt_refit(problem, panel, panel.unit_ids[target], target, donors, pre, post, on_failure)
  end
  treated = _original_refit(problem, solution, panel)
  for refit in refits
    refit.solver_status === :success || continue
    if !isfinite(refit.rmspe_ratio)
      refit.included = false
      refit.exclusion_reason = :nonfinite_ratio
    elseif cutoff !== nothing && refit.pre_rmspe > cutoff * treated.pre_rmspe
      refit.included = false
      refit.exclusion_reason = :poor_pre_fit
    end
  end
  eligible_refits = [refit for refit in refits if refit.included]
  p_value = if isfinite(treated.rmspe_ratio) && !isempty(eligible_refits)
    numerator = 1 + count(refit -> refit.rmspe_ratio >= treated.rmspe_ratio, eligible_refits)
    eltype(panel.outcomes)(numerator / (1 + length(eligible_refits)))
  else
    nothing
  end
  return InSpacePlaceboResult(solution, panel, treated, refits, include_treated, cutoff, p_value)
end

function in_space_placebos(problem; kwargs...)
  return in_space_placebos(problem, solve(problem); kwargs...)
end

"""
    randomization_p_value(result::InSpacePlaceboResult)

Return the stored exact, one-sided upper-tail placebo p-value. The treated
unit is included in numerator and denominator, ties count with `>=`, and
failed, filtered, or non-finite placebo fits are excluded. Throws
`ArgumentError` when no valid inference set exists.

# Examples

```julia
isdefined(SyntheticControl, :randomization_p_value)
```
"""
function randomization_p_value(result::InSpacePlaceboResult)
  result.p_value === nothing && throw(ArgumentError("no finite non-empty placebo inference set is available"))
  return result.p_value
end

"""
    leave_one_out(problem, solution; donors=nothing, active_only=false,
                  weight_tol=sqrt(eps(T)), on_failure=:record, parallel=false)
    leave_one_out(problem; kwargs...)

Remove exactly one selected donor at a time and completely refit the
original treated unit with the same estimator configuration. With
`active_only=true`, only donors whose original weight exceeds `weight_tol`
are used. All attempted refits, metrics, statuses, and failures are retained
in original donor order. No p-value is computed.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=4), time=repeat(1:4, 3),
         y=[2.,3,5,7, 1,2,3,4, 3,4,5,6],
         x=[2.,3,4,5, 1,2,3,4, 3,4,5,6])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=3)
loo = leave_one_out(problem, solve(problem))
length(loo.refits) == 2
```
"""
function leave_one_out(
  problem,
  solution;
  donors=nothing,
  active_only::Bool=false,
  weight_tol=sqrt(eps(eltype(solution.W))),
  on_failure::Symbol=:record,
  parallel::Bool=false
)
  _validate_original(problem, solution)
  _validate_failure_policy(on_failure)
  panel = _panel_data(problem)
  T = eltype(panel.outcomes)
  tolerance = T(weight_tol)
  isfinite(tolerance) && tolerance >= zero(T) || throw(ArgumentError("weight_tol must be finite and non-negative"))
  donor_indices = [index for index in eachindex(panel.unit_ids) if index != panel.treated_index]
  chosen_positions = _selection_indices(panel.unit_ids[donor_indices], donors, "donor")
  if active_only
    chosen_positions = [position for position in chosen_positions if solution.W[position] > tolerance]
    isempty(chosen_positions) && throw(ArgumentError("no selected donors exceed weight_tol"))
  end
  chosen = donor_indices[chosen_positions]
  pre = collect(1:(panel.treatment_index - 1))
  post = collect(panel.treatment_index:length(panel.time))
  original_path = _original_refit(problem, solution, panel)
  refits = _batch_refits(length(chosen), parallel && on_failure === :record) do position
    omitted = chosen[position]
    remaining = [index for index in donor_indices if index != omitted]
    _attempt_refit(
      problem, panel, panel.unit_ids[omitted], panel.treated_index, remaining, pre, post, on_failure;
      original_synthetic=original_path.synthetic
    )
  end
  return LeaveOneOutResult(solution, panel, original_path, refits, tolerance, active_only)
end

function leave_one_out(problem; kwargs...)
  return leave_one_out(problem, solve(problem); kwargs...)
end

function _post_indices(panel, pseudo_index::Int, post_window, stop_before_treatment::Bool)
  last_index = stop_before_treatment ? panel.treatment_index - 1 : length(panel.time)
  available = collect(pseudo_index:last_index)
  if post_window === nothing
    return available
  elseif post_window isa Integer
    post_window > 0 || throw(ArgumentError("integer post_window must be positive"))
    return available[1:min(Int(post_window), length(available))]
  end
  available_times = copy(panel.time[available])
  selected = applicable(post_window, available_times) ? post_window(available_times) : collect(post_window)
  if selected isa AbstractVector{Bool}
    length(selected) == length(available) || throw(DimensionMismatch("callable post_window mask must match available periods"))
    return available[selected]
  end
  selected_values = collect(selected)
  length(unique(selected_values)) == length(selected_values) || throw(ArgumentError("post_window contains duplicate times"))
  indices = Int[]
  for value in selected_values
    index = findfirst(isequal(value), panel.time)
    index === nothing && throw(ArgumentError("post_window contains an unknown time: $value"))
    index in available || throw(ArgumentError("post_window time $value is outside the allowed evaluation interval"))
    push!(indices, index)
  end
  sort!(indices)
  return indices
end

"""
    in_time_placebos(problem, solution; placebo_times, post_window=nothing,
                     stop_before_treatment=true, minimum_pre_periods=2,
                     minimum_post_periods=1, on_failure=:record,
                     parallel=false)
    in_time_placebos(problem; placebo_times, kwargs...)

Completely refit the treated SCM at each observed pseudo-treatment time.
Times before the pseudo date form the fitting period and the pseudo date is
the first evaluation period. By default evaluation stops before the real
treatment. `post_window` may be a positive number of observed periods, an
explicit collection of observed times, or a callable receiving available
times and returning a time collection or Boolean mask. No calendar
arithmetic or equal spacing is assumed.

Insufficient windows and solver errors are recorded or thrown according to
`on_failure`. Independent refits may run with `parallel=true` while result
ordering remains the order of `placebo_times`.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=5), time=repeat([1,3,6,10,15], 3),
         y=[2.,3,4,6,8, 1,2,3,4,5, 3,4,5,6,7],
         x=[2.,3,4,5,6, 1,2,3,4,5, 3,4,5,6,7])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=10)
timing = in_time_placebos(problem, solve(problem); placebo_times=[6])
timing.refits[1].time == [1, 3, 6]
```
"""
function in_time_placebos(
  problem,
  solution;
  placebo_times,
  post_window=nothing,
  stop_before_treatment::Bool=true,
  minimum_pre_periods::Int=2,
  minimum_post_periods::Int=1,
  on_failure::Symbol=:record,
  parallel::Bool=false
)
  _validate_original(problem, solution)
  _validate_failure_policy(on_failure)
  minimum_pre_periods >= 1 || throw(ArgumentError("minimum_pre_periods must be positive"))
  minimum_post_periods >= 1 || throw(ArgumentError("minimum_post_periods must be positive"))
  panel = _panel_data(problem)
  time_indices = _selection_indices(panel.time, placebo_times, "placebo time")
  donors = [index for index in eachindex(panel.unit_ids) if index != panel.treated_index]
  refits = _batch_refits(length(time_indices), parallel && on_failure === :record) do position
    pseudo_index = time_indices[position]
    assignment = panel.time[pseudo_index]
    pre = collect(1:(pseudo_index - 1))
    post = Int[]
    try
      stop_before_treatment && pseudo_index >= panel.treatment_index &&
        throw(ArgumentError("placebo time must occur before the actual treatment"))
      length(pre) >= minimum_pre_periods || throw(ArgumentError("placebo time leaves fewer than minimum_pre_periods"))
      post = _post_indices(panel, pseudo_index, post_window, stop_before_treatment)
      length(post) >= minimum_post_periods || throw(ArgumentError("placebo time leaves fewer than minimum_post_periods"))
      _attempt_refit(problem, panel, assignment, panel.treated_index, donors, pre, post, on_failure)
    catch err
      on_failure === :error && rethrow()
      _failed_refit(panel, assignment, donors, pre, post, err)
    end
  end
  return InTimePlaceboResult(
    solution, panel, refits, stop_before_treatment, minimum_pre_periods, minimum_post_periods
  )
end

function in_time_placebos(problem; placebo_times, kwargs...)
  return in_time_placebos(problem, solve(problem); placebo_times=placebo_times, kwargs...)
end

"""
    robustness_counts(result)

Return counts of successful, failed, filtered, and inference-eligible refits.
For leave-one-out and in-time results, every successful refit is eligible as
a diagnostic but no p-value is implied.

# Examples

```julia
isdefined(SyntheticControl, :robustness_counts)
```
"""
function robustness_counts(result::InSpacePlaceboResult)
  successful = count(refit -> refit.solver_status === :success, result.refits)
  failed = length(result.refits) - successful
  eligible = count(refit -> refit.included, result.refits)
  return (successful=successful, failed=failed, filtered=successful - eligible, inference_eligible=eligible)
end

function robustness_counts(result::Union{LeaveOneOutResult,InTimePlaceboResult})
  successful = count(refit -> refit.solver_status === :success, result.refits)
  return (successful=successful, failed=length(result.refits) - successful, filtered=0, inference_eligible=successful)
end

"""
    SyntheticControlPlaceboResult(result::InSpacePlaceboResult)

Convert successful in-space refits to the package's plot-ready placebo data.
Failed attempts remain available on `result.refits` but cannot supply paths
and are omitted from this rendering adapter.

# Examples

```julia
isdefined(SyntheticControl, :SyntheticControlPlaceboResult)
```
"""
function SyntheticControlPlaceboResult(result::InSpacePlaceboResult{T}) where {T}
  successful = [refit for refit in result.refits if refit.solver_status === :success]
  isempty(successful) && throw(ArgumentError("no successful placebo paths are available for plotting"))
  actual = reduce(hcat, (refit.actual for refit in successful))
  synthetic = reduce(hcat, (refit.synthetic for refit in successful))
  return SyntheticControlPlaceboResult(
    result.panel.time,
    result.panel.treatment_index,
    result.panel.treatment_time,
    string(result.panel.unit_ids[result.panel.treated_index]),
    string.([refit.assignment for refit in successful]),
    result.treated.actual,
    result.treated.synthetic,
    Matrix{T}(actual),
    Matrix{T}(synthetic)
  )
end
