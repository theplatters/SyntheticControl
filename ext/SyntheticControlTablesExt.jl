module SyntheticControlTablesExt

using SyntheticControl
using Tables

const ResultLike = Union{
  SyntheticControl.SyntheticControlResult,
  SyntheticControl.PenalizedSyntheticControlResult
}

struct PanelMetadata{T<:AbstractFloat,TT}
  time::Vector{TT}
  treatment_time::TT
  treatment_index::Int
  actual::Vector{T}
  donor_outcomes::Matrix{T}
end

const PANEL_METADATA = IdDict{SyntheticControl.SyntheticControlData,Any}()

_column_key(name::Symbol) = name
_column_key(name::AbstractString) = Symbol(name)
_column_key(name) = throw(ArgumentError("column selectors must be Symbols or Strings; got $(typeof(name))"))

function _selected_columns(unit, time, outcome, predictors)
  predictor_keys = [_column_key(predictor) for predictor in predictors]
  isempty(predictor_keys) && throw(ArgumentError("predictors must contain at least one column"))
  return _column_key(unit), _column_key(time), _column_key(outcome), predictor_keys
end

function _validate_column_names!(names, selected)
  name_strings = String.(names)
  length(unique(name_strings)) == length(name_strings) || throw(ArgumentError("table schema contains duplicate or ambiguous column names"))
  available = Set(Symbol.(names))
  for name in selected
    name in available || throw(ArgumentError("unknown selected column: $name"))
  end
  return nothing
end

function _row_names(row)
  names = propertynames(row)
  isempty(names) && throw(ArgumentError("table rows must expose named columns"))
  return Symbol.(names)
end

function _numeric_value(value, column::Symbol)
  ismissing(value) && throw(ArgumentError("column $column contains missing values"))
  value isa Bool && throw(ArgumentError("column $column must be numeric, not Bool"))
  value isa Real || throw(ArgumentError("column $column must contain numeric values"))
  isfinite(value) || throw(ArgumentError("column $column contains non-finite values"))
  return value
end

function _ordered_values(values, label::AbstractString)
  ordered = collect(values)
  try
    sort!(ordered)
  catch err
    throw(ArgumentError("$label values must support deterministic ordering with isless"))
  end
  value_type = Base.promote_typeof(ordered...)
  return value_type[ordered...]
end

function _time_less(a, b)
  try
    return isless(a, b)
  catch err
    throw(ArgumentError("time values and treatment_time must be mutually orderable"))
  end
end

function _as_key(value)
  ismissing(value) && throw(ArgumentError("unit and time columns may not contain missing values"))
  return value
end

function _string_ids(values)
  return String[string(value) for value in values]
end

function _collect_panel(table, unit_col, time_col, outcome_col, predictor_cols)
  Tables.istable(table) || throw(ArgumentError("input must satisfy Tables.istable"))

  schema = Tables.schema(table)
  selected = [unit_col, time_col, outcome_col, predictor_cols...]
  if schema !== nothing
    _validate_column_names!(collect(schema.names), selected)
  end

  rows = Tables.rows(table)
  observations = Dict{Tuple{Any,Any},Tuple{Any,Vector{Any}}}()
  units_seen = Set{Any}()
  times_seen = Set{Any}()
  numeric_types = Type[]
  row_count = 0
  names_checked = schema !== nothing

  for row in rows
    row_count += 1
    if !names_checked
      _validate_column_names!(_row_names(row), selected)
      names_checked = true
    end

    unit_value = _as_key(getproperty(row, unit_col))
    time_value = _as_key(getproperty(row, time_col))
    outcome_value = _numeric_value(getproperty(row, outcome_col), outcome_col)
    predictor_values = Any[_numeric_value(getproperty(row, predictor_col), predictor_col) for predictor_col in predictor_cols]

    push!(numeric_types, typeof(outcome_value))
    append!(numeric_types, typeof.(predictor_values))

    key = (unit_value, time_value)
    haskey(observations, key) && throw(ArgumentError("duplicate observation for unit=$(unit_value), time=$(time_value)"))
    observations[key] = (outcome_value, predictor_values)
    push!(units_seen, unit_value)
    push!(times_seen, time_value)
  end

  row_count > 0 || throw(ArgumentError("input table contains no rows"))
  names_checked || throw(ArgumentError("table schema is unknown and no rows were available for inference"))
  return observations, units_seen, times_seen, float(promote_type(numeric_types...))
end

function _panel_arrays(table; unit, time, outcome, predictors, treated, treatment_time)
  unit_col, time_col, outcome_col, predictor_cols = _selected_columns(unit, time, outcome, predictors)
  observations, units_seen, times_seen, T = _collect_panel(table, unit_col, time_col, outcome_col, predictor_cols)

  treated in units_seen || throw(ArgumentError("treated unit $(treated) is absent from the table"))
  length(units_seen) > 1 || throw(ArgumentError("donor pool is empty"))

  ordered_times = _ordered_values(times_seen, "time")
  treatment_time in ordered_times || throw(ArgumentError("treatment_time must be one of the observed time values"))
  pre_times = [value for value in ordered_times if _time_less(value, treatment_time)]
  post_times = [value for value in ordered_times if !_time_less(value, treatment_time)]
  isempty(pre_times) && throw(ArgumentError("at least one pre-treatment observation is required"))
  isempty(post_times) && throw(ArgumentError("at least one post-treatment observation is required"))

  ordered_units = _ordered_values(units_seen, "unit")
  donors = [unit_value for unit_value in ordered_units if unit_value != treated]
  isempty(donors) && throw(ArgumentError("donor pool is empty"))

  expected_observations = length(ordered_units) * length(ordered_times)
  length(observations) == expected_observations || throw(ArgumentError("panel is unbalanced; every unit must have every time value exactly once"))

  K = length(predictor_cols)
  J = length(donors)
  T_pre = length(pre_times)
  T_all = length(ordered_times)
  X1 = Vector{T}(undef, K)
  X0 = Matrix{T}(undef, K, J)
  Y1 = Vector{T}(undef, T_pre)
  Y0 = Matrix{T}(undef, T_pre, J)
  actual = Vector{T}(undef, T_all)
  donor_outcomes = Matrix{T}(undef, T_all, J)
  all_outcomes = Matrix{T}(undef, T_all, length(ordered_units))
  all_predictors = Array{T,3}(undef, T_all, K, length(ordered_units))

  for (time_index, time_value) in pairs(ordered_times)
    treated_outcome, _ = observations[(treated, time_value)]
    actual[time_index] = T(treated_outcome)
    for (unit_index, unit_value) in pairs(ordered_units)
      unit_outcome, unit_predictors = observations[(unit_value, time_value)]
      all_outcomes[time_index, unit_index] = T(unit_outcome)
      for predictor_index in 1:K
        all_predictors[time_index, predictor_index, unit_index] = T(unit_predictors[predictor_index])
      end
    end
    for (donor_index, donor) in pairs(donors)
      donor_outcome, _ = observations[(donor, time_value)]
      donor_outcomes[time_index, donor_index] = T(donor_outcome)
    end
  end

  for (pre_index, time_value) in pairs(pre_times)
    Y1[pre_index] = actual[pre_index]
    for donor_index in 1:J
      Y0[pre_index, donor_index] = donor_outcomes[pre_index, donor_index]
    end
  end

  for predictor_index in 1:K
    total = zero(T)
    for time_value in pre_times
      _, values = observations[(treated, time_value)]
      total += T(values[predictor_index])
    end
    X1[predictor_index] = total / T(T_pre)

    for (donor_index, donor) in pairs(donors)
      total = zero(T)
      for time_value in pre_times
        _, values = observations[(donor, time_value)]
        total += T(values[predictor_index])
      end
      X0[predictor_index, donor_index] = total / T(T_pre)
    end
  end

  treatment_index = findfirst(==(treatment_time), ordered_times)::Int
  predictor_names = String.(predictor_cols)
  donor_ids = _string_ids(donors)
  panel = SyntheticControl.SyntheticControlPanelData(
    ordered_times,
    treatment_time,
    ordered_units,
    treated,
    all_outcomes,
    all_predictors,
    predictor_names
  )
  return X1, Y1, X0, Y0, predictor_names, donor_ids, string(treated),
         PanelMetadata(ordered_times, treatment_time, treatment_index, actual, donor_outcomes), panel
end

"""
    SyntheticControl.from_table(table; unit, time, outcome, predictors, treated, treatment_time, kwargs...)

Build a `SyntheticControlProblem` from a Tables.jl-compatible long-format
panel. `unit`, `time`, `outcome`, and `predictors` select columns by `Symbol`
or `String`. Predictor columns are averaged over rows whose time is before
`treatment_time`; post-treatment rows are retained only for `path_table`.

The source is materialized once with `Tables.rows`, so single-pass row
iterators are supported. Inputs must form a balanced panel with one row per
`(unit, time)` pair and finite numeric outcomes and predictors.

# Examples

```julia
using SyntheticControl, Tables

panel = (
  unit = repeat(["treated", "a", "b"], inner=4),
  year = repeat(2001:2004, 3),
  outcome = [10.0, 11.0, 14.0, 16.0, 9.0, 10.0, 11.0, 12.0, 11.0, 11.5, 12.0, 12.5],
  income = [20.0, 21.0, 23.0, 24.0, 18.0, 19.0, 20.0, 21.0, 22.0, 22.0, 23.0, 24.0],
)
problem = from_table(panel; unit=:unit, time=:year, outcome=:outcome,
                     predictors=[:income], treated="treated", treatment_time=2003)
size(problem.X0) == (1, 2)
```
"""
function SyntheticControl.from_table(
  table;
  unit,
  time,
  outcome,
  predictors,
  treated,
  treatment_time,
  kwargs...
)
  X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id, metadata, panel = _panel_arrays(
    table;
    unit=unit,
    time=time,
    outcome=outcome,
    predictors=predictors,
    treated=treated,
    treatment_time=treatment_time
  )
  problem = SyntheticControl.SyntheticControlProblem(
    X1,
    Y1,
    X0,
    Y0,
    predictor_names,
    donor_ids,
    treated_id;
    kwargs...
  )
  PANEL_METADATA[problem.data] = metadata
  SyntheticControl._register_panel_data!(problem.data, panel)
  return problem
end

"""
    SyntheticControl.weights_table(result)

Return a Tables.jl-compatible column table with schema `donor, weight`.

# Examples

```julia
using SyntheticControl, Tables

problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
table = weights_table(problem.result)
Tables.columnnames(table) == (:donor, :weight)
```
"""
function SyntheticControl.weights_table(result::ResultLike)
  return (
    donor=copy(result.data.donor_ids),
    weight=copy(result.W),
  )
end

"""
    SyntheticControl.balance_table(problem, result)

Return predictor balance as a Tables.jl-compatible column table with schema
`predictor, treated, synthetic, difference`.

# Examples

```julia
using SyntheticControl, Tables

problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
table = balance_table(problem, problem.result)
Tables.columnnames(table) == (:predictor, :treated, :synthetic, :difference)
```
"""
function SyntheticControl.balance_table(problem, result::ResultLike)
  data = getproperty(problem, :data)
  data === result.data || throw(ArgumentError("problem and result must reference the same SyntheticControlData"))
  synthetic = data.X0 * result.W
  treated = copy(data.X1)
  return (
    predictor=copy(data.predictor_names),
    treated=treated,
    synthetic=synthetic,
    difference=treated .- synthetic,
  )
end

"""
    SyntheticControl.path_table(problem, result)

Return a Tables.jl-compatible column table with schema `time, actual,
synthetic, gap, post_treatment`. For problems built by `from_table`, the table
contains all observed pre- and post-treatment periods. For matrix-built
problems, it contains the pre-treatment path only with integer time values.

# Examples

```julia
using SyntheticControl, Tables

problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
table = path_table(problem, problem.result)
Tables.columnnames(table) == (:time, :actual, :synthetic, :gap, :post_treatment)
```
"""
function SyntheticControl.path_table(problem, result::ResultLike)
  data = getproperty(problem, :data)
  data === result.data || throw(ArgumentError("problem and result must reference the same SyntheticControlData"))

  metadata = get(PANEL_METADATA, data, nothing)
  if metadata === nothing
    time = collect(1:length(data.Y1))
    actual = copy(data.Y1)
    synthetic = data.Y0 * result.W
    post_treatment = fill(false, length(time))
  else
    time = copy(metadata.time)
    actual = copy(metadata.actual)
    synthetic = metadata.donor_outcomes * result.W
    post_treatment = [index >= metadata.treatment_index for index in eachindex(time)]
  end

  return (
    time=time,
    actual=actual,
    synthetic=synthetic,
    gap=actual .- synthetic,
    post_treatment=post_treatment,
  )
end

"""
    SyntheticControl.diagnostics_table(fit; weight_sum_tolerance=nothing)

Return one row with the stable schema `pre_rmspe, pre_mae,
max_absolute_pre_gap, effective_donor_count, largest_donor_weight, weight_sum,
weight_sum_error, predictor_mae, predictor_max_absolute_imbalance,
predictor_mean_relative_imbalance, predictor_max_relative_imbalance, solver,
solver_success, termination_status, iteration_count, objective_value,
runtime_seconds, objective_evaluations`.

Solver metadata not retained by the fitted result is a typed `missing` value.
See [`SyntheticControl.fit_diagnostics`](@ref) for statistical definitions and
validation behavior.

# Examples

```julia
using SyntheticControl, Tables
data = SyntheticControlData([2.0], [1.0], reshape([1.0, 3.0], 1, 2), [0.0 2.0],
                            ["x"], ["a", "b"], "t")
fit = SyntheticControl.SyntheticControlResult(data, [0.5, 0.5], [1.0], 0.0)
Tables.columnnames(diagnostics_table(fit))[1] == :pre_rmspe
```
"""
function SyntheticControl.diagnostics_table(fit::ResultLike; weight_sum_tolerance=nothing)
  diagnostics = SyntheticControl.fit_diagnostics(fit; weight_sum_tolerance=weight_sum_tolerance)
  return _diagnostics_columns(diagnostics)
end

"""
    _diagnostics_columns(diagnostics)

Build the stable one-row diagnostics column table from already calculated
diagnostics. This helper performs no fitting or diagnostic calculation.

# Examples

```julia
isdefined(SyntheticControlTablesExt, :_diagnostics_columns)
```
"""
function _diagnostics_columns(diagnostics)
  T = typeof(diagnostics.pre_rmspe)
  return (
    pre_rmspe=T[diagnostics.pre_rmspe],
    pre_mae=T[diagnostics.pre_mae],
    max_absolute_pre_gap=T[diagnostics.max_absolute_pre_gap],
    effective_donor_count=T[diagnostics.effective_donor_count],
    largest_donor_weight=T[diagnostics.largest_donor_weight],
    weight_sum=T[diagnostics.weight_sum],
    weight_sum_error=T[diagnostics.weight_sum_error],
    predictor_mae=T[diagnostics.predictor_mae],
    predictor_max_absolute_imbalance=T[diagnostics.predictor_max_absolute_imbalance],
    predictor_mean_relative_imbalance=T[diagnostics.predictor_mean_relative_imbalance],
    predictor_max_relative_imbalance=T[diagnostics.predictor_max_relative_imbalance],
    solver=Symbol[diagnostics.solver],
    solver_success=Union{Missing,Bool}[diagnostics.solver_success],
    termination_status=Union{Missing,Symbol}[diagnostics.termination_status],
    iteration_count=Union{Missing,Int}[diagnostics.iteration_count],
    objective_value=T[diagnostics.objective_value],
    runtime_seconds=Union{Missing,T}[diagnostics.runtime_seconds],
    objective_evaluations=Union{Missing,Int}[diagnostics.objective_evaluations],
  )
end

"""
    SyntheticControl.diagnostics_table(suite::RobustnessSuiteResult)

Return the suite's stored baseline diagnostics with the same stable one-row
schema as `diagnostics_table(fit)`. The suite baseline is not refit and its
diagnostics are not recalculated.

# Examples

```julia
using SyntheticControl, Tables
isdefined(SyntheticControl, :RobustnessSuiteResult)
```
"""
function SyntheticControl.diagnostics_table(suite::SyntheticControl.RobustnessSuiteResult)
  return _diagnostics_columns(suite.diagnostics)
end

"""
    SyntheticControl.predictor_diagnostics_table(fit; weight_sum_tolerance=nothing)

Return one row per predictor with stable schema `predictor, treated,
synthetic, difference, absolute_difference, relative_difference`. Rows remain
in the fit's predictor order. See [`SyntheticControl.predictor_diagnostics`](@ref)
for definitions and validation behavior.

# Examples

```julia
using SyntheticControl, Tables
data = SyntheticControlData([2.0], [1.0], reshape([1.0, 3.0], 1, 2), [0.0 2.0],
                            ["x"], ["a", "b"], "t")
fit = SyntheticControl.SyntheticControlResult(data, [0.5, 0.5], [1.0], 0.0)
Tables.columnnames(predictor_diagnostics_table(fit)) ==
  (:predictor, :treated, :synthetic, :difference, :absolute_difference, :relative_difference)
```
"""
function SyntheticControl.predictor_diagnostics_table(fit::ResultLike; weight_sum_tolerance=nothing)
  diagnostics = SyntheticControl.predictor_diagnostics(fit; weight_sum_tolerance=weight_sum_tolerance)
  return (
    predictor=copy(diagnostics.predictor),
    treated=copy(diagnostics.treated),
    synthetic=copy(diagnostics.synthetic),
    difference=copy(diagnostics.difference),
    absolute_difference=copy(diagnostics.absolute_difference),
    relative_difference=copy(diagnostics.relative_difference),
  )
end

"""
    SyntheticControl.placebo_summary(result::InSpacePlaceboResult)

Return one row for the treated unit followed by one row per attempted
in-space assignment. The stable schema is `unit, is_treated, pre_rmspe,
post_rmspe, rmspe_ratio, included, exclusion_reason, solver_status`.

# Examples

```julia
using SyntheticControl, Tables
Tables.columnnames((unit=String[], is_treated=Bool[], pre_rmspe=Float64[],
                    post_rmspe=Float64[], rmspe_ratio=Float64[], included=Bool[],
                    exclusion_reason=Union{Missing,String}[], solver_status=Symbol[]))
```
"""
function SyntheticControl.placebo_summary(result::SyntheticControl.InSpacePlaceboResult{T}) where {T}
  rows = [result.treated; result.refits]
  return (
    unit=[row.assignment for row in rows],
    is_treated=[index == 1 for index in eachindex(rows)],
    pre_rmspe=T[row.pre_rmspe for row in rows],
    post_rmspe=T[row.post_rmspe for row in rows],
    rmspe_ratio=T[row.rmspe_ratio for row in rows],
    included=[index == 1 ? isfinite(row.rmspe_ratio) : row.included for (index, row) in pairs(rows)],
    exclusion_reason=Union{Missing,String}[
      row.exclusion_reason === nothing ? missing : String(row.exclusion_reason) for row in rows
    ],
    solver_status=Symbol[row.solver_status for row in rows],
  )
end

"""
    SyntheticControl.leave_one_out_summary(result::LeaveOneOutResult)

Return the stable leave-one-out schema `omitted_donor, original_weight,
pre_rmspe, post_rmspe, rmspe_ratio, mean_post_gap, cumulative_post_gap,
max_path_deviation, solver_status`.

# Examples

```julia
using SyntheticControl, Tables
isdefined(SyntheticControl, :leave_one_out_summary)
```
"""
function SyntheticControl.leave_one_out_summary(result::SyntheticControl.LeaveOneOutResult{T}) where {T}
  panel_donors = [unit for (index, unit) in pairs(result.panel.unit_ids) if index != result.panel.treated_index]
  weights = T[]
  for refit in result.refits
    position = findfirst(isequal(refit.assignment), panel_donors)
    push!(weights, result.original.W[position])
  end
  return (
    omitted_donor=[refit.assignment for refit in result.refits],
    original_weight=weights,
    pre_rmspe=T[refit.pre_rmspe for refit in result.refits],
    post_rmspe=T[refit.post_rmspe for refit in result.refits],
    rmspe_ratio=T[refit.rmspe_ratio for refit in result.refits],
    mean_post_gap=T[refit.mean_post_gap for refit in result.refits],
    cumulative_post_gap=T[refit.cumulative_post_gap for refit in result.refits],
    max_path_deviation=T[refit.max_path_deviation for refit in result.refits],
    solver_status=Symbol[refit.solver_status for refit in result.refits],
  )
end

"""
    SyntheticControl.in_time_summary(result::InTimePlaceboResult)

Return the stable in-time schema `placebo_time, pre_rmspe, post_rmspe,
rmspe_ratio, mean_post_gap, cumulative_post_gap, pre_periods, post_periods,
solver_status`. Period columns contain observation counts.

# Examples

```julia
using SyntheticControl, Tables
isdefined(SyntheticControl, :in_time_summary)
```
"""
function SyntheticControl.in_time_summary(result::SyntheticControl.InTimePlaceboResult{T}) where {T}
  return (
    placebo_time=[refit.assignment for refit in result.refits],
    pre_rmspe=T[refit.pre_rmspe for refit in result.refits],
    post_rmspe=T[refit.post_rmspe for refit in result.refits],
    rmspe_ratio=T[refit.rmspe_ratio for refit in result.refits],
    mean_post_gap=T[refit.mean_post_gap for refit in result.refits],
    cumulative_post_gap=T[refit.cumulative_post_gap for refit in result.refits],
    pre_periods=Int[max(refit.treatment_index - 1, 0) for refit in result.refits],
    post_periods=Int[isempty(refit.time) ? 0 : length(refit.time) - refit.treatment_index + 1 for refit in result.refits],
    solver_status=Symbol[refit.solver_status for refit in result.refits],
  )
end

"""
    _append_refit_path_rows!(columns, refit; included=refit.included)

Append common long-path columns from one stored robustness refit and return
the number of appended rows. Successful refits append every stored period;
failed refits append one sentinel row with missing time and path values. The
helper validates successful stored-path dimensions and never estimates.

# Examples

```julia
isdefined(SyntheticControlTablesExt, :_append_refit_path_rows!)
```
"""
function _append_refit_path_rows!(columns, refit; included::Bool=refit.included)
  refit.solver_status in (:success, :failed) ||
    throw(ArgumentError("unknown stored solver status $(refit.solver_status)"))
  if refit.solver_status === :success
    path_length = length(refit.time)
    length(refit.actual) == path_length || throw(DimensionMismatch("stored actual path must match stored time"))
    length(refit.synthetic) == path_length || throw(DimensionMismatch("stored synthetic path must match stored time"))
    length(refit.gap) == path_length || throw(DimensionMismatch("stored gap path must match stored time"))
    1 <= refit.treatment_index <= path_length ||
      throw(ArgumentError("stored treatment_index must identify the first post-treatment period"))
    for position in eachindex(refit.time)
      push!(columns.analysis_id, refit.assignment)
      push!(columns.time, refit.time[position])
      push!(columns.actual, refit.actual[position])
      push!(columns.synthetic, refit.synthetic[position])
      push!(columns.gap, refit.gap[position])
      push!(columns.is_post_treatment, position >= refit.treatment_index)
      push!(columns.included, included)
      push!(columns.solver_status, refit.solver_status)
      push!(columns.failure_reason, missing)
    end
    return path_length
  end

  push!(columns.analysis_id, refit.assignment)
  push!(columns.time, missing)
  push!(columns.actual, missing)
  push!(columns.synthetic, missing)
  push!(columns.gap, missing)
  push!(columns.is_post_treatment, missing)
  push!(columns.included, false)
  push!(columns.solver_status, refit.solver_status)
  push!(columns.failure_reason, refit.failure_reason === nothing ? missing : refit.failure_reason)
  return 1
end

"""
    _robustness_path_columns(T, AnalysisID, Time)

Allocate typed mutable columns shared by all robustness long-path tables.
Time and numeric path columns allow `missing` so failed assignments can be
represented without erasing their native analysis identifiers.

# Examples

```julia
columns = SyntheticControlTablesExt._robustness_path_columns(Float64, Symbol, Int)
eltype(columns.time) == Union{Missing,Int}
```
"""
function _robustness_path_columns(::Type{T}, ::Type{AI}, ::Type{TT}) where {T,AI,TT}
  return (
    analysis_id=AI[],
    time=Union{Missing,TT}[],
    actual=Union{Missing,T}[],
    synthetic=Union{Missing,T}[],
    gap=Union{Missing,T}[],
    is_post_treatment=Union{Missing,Bool}[],
    included=Bool[],
    solver_status=Symbol[],
    failure_reason=Union{Missing,String}[],
  )
end

"""
    SyntheticControl.placebo_paths(result::InSpacePlaceboResult)

Return the treated path followed by every attempted in-space assignment in
stored order. The stable common columns are `analysis_id, time, actual,
synthetic, gap, is_post_treatment, included, solver_status, failure_reason`;
additional columns are `is_treated, exclusion_reason`.

Successful filtered assignments retain complete paths with `included=false`.
Failed assignments retain one sentinel row with missing time and path values.
The treated assignment is inference-included only when its stored RMSPE ratio
is finite. No fitting or statistic calculation occurs.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=4), time=repeat(1:4, 3),
         y=[2.,3,5,7, 1,2,3,4, 3,4,5,6], x=[2.,3,4,5, 1,2,3,4, 3,4,5,6])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=3)
Tables.istable(placebo_paths(in_space_placebos(problem, solve(problem))))
```
"""
function SyntheticControl.placebo_paths(
  result::SyntheticControl.InSpacePlaceboResult{T,UI,TT}
) where {T,UI,TT}
  columns = _robustness_path_columns(T, UI, TT)
  is_treated = Bool[]
  exclusion_reason = Union{Missing,Symbol}[]
  rows = [result.treated; result.refits]
  for (index, refit) in pairs(rows)
    treated = index == 1
    included = treated ? isfinite(refit.rmspe_ratio) : refit.included
    appended = _append_refit_path_rows!(columns, refit; included=included)
    append!(is_treated, fill(treated, appended))
    reason = refit.exclusion_reason === nothing ? missing : refit.exclusion_reason
    append!(exclusion_reason, fill(reason, appended))
  end
  return merge(columns, (; is_treated=is_treated, exclusion_reason=exclusion_reason))
end

"""
    SyntheticControl.leave_one_out_paths(result::LeaveOneOutResult)

Return every stored leave-one-out refit path in assignment order. The stable
common columns are `analysis_id, time, actual, synthetic, gap,
is_post_treatment, included, solver_status, failure_reason`; the additional
`original_weight` column repeats the omitted donor's baseline weight.

Successful refits retain complete paths and failed refits retain one sentinel
row. The original baseline path is not emitted and remains available as
`result.original_path`. No fitting or statistic calculation occurs.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=4), time=repeat(1:4, 3),
         y=[2.,3,5,7, 1,2,3,4, 3,4,5,6], x=[2.,3,4,5, 1,2,3,4, 3,4,5,6])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=3)
Tables.istable(leave_one_out_paths(leave_one_out(problem, solve(problem))))
```
"""
function SyntheticControl.leave_one_out_paths(
  result::SyntheticControl.LeaveOneOutResult{T,UI,TT}
) where {T,UI,TT}
  columns = _robustness_path_columns(T, UI, TT)
  original_weight = T[]
  panel_donors = [
    unit for (index, unit) in pairs(result.panel.unit_ids) if index != result.panel.treated_index
  ]
  for refit in result.refits
    position = findfirst(isequal(refit.assignment), panel_donors)
    position === nothing && throw(ArgumentError("stored omitted donor is absent from the original donor pool"))
    appended = _append_refit_path_rows!(columns, refit; included=refit.solver_status === :success)
    append!(original_weight, fill(result.original.W[position], appended))
  end
  return merge(columns, (; original_weight=original_weight))
end

"""
    SyntheticControl.in_time_paths(result::InTimePlaceboResult)

Return every stored in-time placebo path in pseudo-date order with stable
schema `analysis_id, time, actual, synthetic, gap, is_post_treatment,
included, solver_status, failure_reason`. Here `analysis_id` is the native
pseudo-treatment time. Successful refits retain complete effective windows;
failed pseudo-dates retain one sentinel row. No fitting or statistic
calculation occurs.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=5), time=repeat([1,3,6,10,15], 3),
         y=[2.,3,4,6,8, 1,2,3,4,5, 3,4,5,6,7],
         x=[2.,3,4,5,6, 1,2,3,4,5, 3,4,5,6,7])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=10)
timing = in_time_placebos(problem, solve(problem); placebo_times=[6])
Tables.istable(in_time_paths(timing))
```
"""
function SyntheticControl.in_time_paths(
  result::SyntheticControl.InTimePlaceboResult{T,UI,TT}
) where {T,UI,TT}
  columns = _robustness_path_columns(T, TT, TT)
  for refit in result.refits
    _append_refit_path_rows!(columns, refit; included=refit.solver_status === :success)
  end
  return columns
end

"""
    SyntheticControl.pointwise_placebo_summary(result::PointwisePlaceboInferenceResult)

Return one row per post-treatment period with stable schema `time, statistic,
alternative, treated_statistic, extreme_count, assignment_count, p_value,
p_value_resolution`. Values are copied from the stored inference result and
no inference or fitting is performed.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=4), time=repeat(1:4, 3),
         y=[2.,3,5,7, 1,2,3,4, 3,4,5,6], x=[2.,3,4,5, 1,2,3,4, 3,4,5,6])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=3)
inference = pointwise_placebo_inference(in_space_placebos(problem, solve(problem)))
Tables.istable(pointwise_placebo_summary(inference))
```
"""
function SyntheticControl.pointwise_placebo_summary(
  result::SyntheticControl.PointwisePlaceboInferenceResult{T}
) where {T}
  row_count = length(result.time)
  return (
    time=copy(result.time),
    statistic=fill(result.statistic, row_count),
    alternative=fill(result.alternative, row_count),
    treated_statistic=copy(result.treated_statistic),
    extreme_count=copy(result.extreme_count),
    assignment_count=fill(result.assignment_count, row_count),
    p_value=copy(result.p_value),
    p_value_resolution=fill(one(T) / T(result.assignment_count), row_count),
  )
end

"""
    SyntheticControl.aggregate_placebo_summary(result::AggregatePlaceboInferenceResult)

Return one row with stable schema `statistic, alternative, periods,
period_count, treated_statistic, extreme_count, assignment_count, p_value,
p_value_resolution`. The `periods` cell is a typed vector retaining native
observed-time values. Values are copied from the stored result; no inference
or fitting is performed.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=4), time=repeat(1:4, 3),
         y=[2.,3,5,7, 1,2,3,4, 3,4,5,6], x=[2.,3,4,5, 1,2,3,4, 3,4,5,6])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=3)
inference = aggregate_placebo_inference(in_space_placebos(problem, solve(problem)))
Tables.istable(aggregate_placebo_summary(inference))
```
"""
function SyntheticControl.aggregate_placebo_summary(
  result::SyntheticControl.AggregatePlaceboInferenceResult{T,UI,TT}
) where {T,UI,TT}
  return (
    statistic=Symbol[result.statistic],
    alternative=Symbol[result.alternative],
    periods=Vector{TT}[copy(result.periods)],
    period_count=Int[length(result.periods)],
    treated_statistic=T[result.treated_statistic],
    extreme_count=Int[result.extreme_count],
    assignment_count=Int[result.assignment_count],
    p_value=T[result.p_value],
    p_value_resolution=T[one(T) / T(result.assignment_count)],
  )
end

"""
    SyntheticControl.specification_definitions(result::SpecificationSensitivityResult)

Return one row per attempted specification with stable schema
`specification_id, pre_periods, omitted_predictors, predictor_periods, donors,
solver_status, included, failure_reason`. Optional unchanged dimensions are
typed `missing`; requested time and donor collections retain panel types.

# Examples

```julia
using SyntheticControl, Tables
isdefined(SyntheticControl, :specification_definitions)
```
"""
function SyntheticControl.specification_definitions(
  result::SyntheticControl.SpecificationSensitivityResult{T,SID,UI,TT}
) where {T,SID,UI,TT}
  pre_periods = Union{Missing,Vector{TT}}[]
  omitted_predictors = Vector{String}[]
  predictor_periods = Union{Missing,Vector{TT}}[]
  donors = Union{Missing,Vector{UI}}[]
  for refit in result.refits
    specification = refit.specification
    push!(pre_periods, specification.pre_periods === nothing ? missing : TT.(specification.pre_periods))
    push!(omitted_predictors, copy(specification.omitted_predictors))
    push!(predictor_periods, specification.predictor_periods === nothing ? missing : TT.(specification.predictor_periods))
    push!(donors, specification.donors === nothing ? missing : UI.(specification.donors))
  end
  return (
    specification_id=SID[refit.specification_id for refit in result.refits],
    pre_periods=pre_periods,
    omitted_predictors=omitted_predictors,
    predictor_periods=predictor_periods,
    donors=donors,
    solver_status=Symbol[refit.solver_status for refit in result.refits],
    included=Bool[refit.included for refit in result.refits],
    failure_reason=Union{Missing,String}[
      refit.failure_reason === nothing ? missing : refit.failure_reason for refit in result.refits
    ],
  )
end

"""
    SyntheticControl.specification_diagnostics(result::SpecificationSensitivityResult)

Return one row per specification with stable scalar fit-diagnostic columns
followed by `solver_status, included, exclusion_reason, failure_reason`.
Failed specifications use typed `missing` diagnostic values. Values are read
from stored diagnostics and are never recalculated.

# Examples

```julia
using SyntheticControl, Tables
isdefined(SyntheticControl, :specification_diagnostics)
```
"""
function SyntheticControl.specification_diagnostics(
  result::SyntheticControl.SpecificationSensitivityResult{T,SID}
) where {T,SID}
  diagnostics = [refit.diagnostics for refit in result.refits]
  numeric(name) = Union{Missing,T}[
    item === nothing ? missing : getproperty(item, name) for item in diagnostics
  ]
  return (
    specification_id=SID[refit.specification_id for refit in result.refits],
    pre_rmspe=numeric(:pre_rmspe),
    pre_mae=numeric(:pre_mae),
    max_absolute_pre_gap=numeric(:max_absolute_pre_gap),
    effective_donor_count=numeric(:effective_donor_count),
    largest_donor_weight=numeric(:largest_donor_weight),
    weight_sum=numeric(:weight_sum),
    weight_sum_error=numeric(:weight_sum_error),
    predictor_mae=numeric(:predictor_mae),
    predictor_max_absolute_imbalance=numeric(:predictor_max_absolute_imbalance),
    predictor_mean_relative_imbalance=numeric(:predictor_mean_relative_imbalance),
    predictor_max_relative_imbalance=numeric(:predictor_max_relative_imbalance),
    solver=Union{Missing,Symbol}[item === nothing ? missing : item.solver for item in diagnostics],
    solver_success=Union{Missing,Bool}[item === nothing ? missing : item.solver_success for item in diagnostics],
    termination_status=Union{Missing,Symbol}[item === nothing ? missing : item.termination_status for item in diagnostics],
    iteration_count=Union{Missing,Int}[item === nothing ? missing : item.iteration_count for item in diagnostics],
    objective_value=numeric(:objective_value),
    runtime_seconds=numeric(:runtime_seconds),
    objective_evaluations=Union{Missing,Int}[item === nothing ? missing : item.objective_evaluations for item in diagnostics],
    solver_status=Symbol[refit.solver_status for refit in result.refits],
    included=Bool[refit.included for refit in result.refits],
    exclusion_reason=Union{Missing,Symbol}[
      refit.exclusion_reason === nothing ? missing : refit.exclusion_reason for refit in result.refits
    ],
    failure_reason=Union{Missing,String}[
      refit.failure_reason === nothing ? missing : refit.failure_reason for refit in result.refits
    ],
  )
end

"""
    SyntheticControl.specification_paths(result::SpecificationSensitivityResult)

Return successful full evaluation paths in specification and stored-time
order with schema `specification_id, time, actual, synthetic, gap,
is_post_treatment, included, solver_status, failure_reason`. Each failed
specification contributes one missing-valued sentinel row. No fitting or gap
calculation occurs.

# Examples

```julia
using SyntheticControl, Tables
isdefined(SyntheticControl, :specification_paths)
```
"""
function SyntheticControl.specification_paths(
  result::SyntheticControl.SpecificationSensitivityResult{T,SID,UI,TT}
) where {T,SID,UI,TT}
  specification_id = SID[]
  time = Union{Missing,TT}[]
  actual = Union{Missing,T}[]
  synthetic = Union{Missing,T}[]
  gap = Union{Missing,T}[]
  is_post_treatment = Union{Missing,Bool}[]
  included = Bool[]
  solver_status = Symbol[]
  failure_reason = Union{Missing,String}[]
  for refit in result.refits
    if refit.solver_status === :success
      path_length = length(refit.time)
      length(refit.actual) == path_length || throw(DimensionMismatch("stored specification actual path must match time"))
      length(refit.synthetic) == path_length || throw(DimensionMismatch("stored specification synthetic path must match time"))
      length(refit.gap) == path_length || throw(DimensionMismatch("stored specification gap path must match time"))
      for index in eachindex(refit.time)
        push!(specification_id, refit.specification_id)
        push!(time, refit.time[index])
        push!(actual, refit.actual[index])
        push!(synthetic, refit.synthetic[index])
        push!(gap, refit.gap[index])
        push!(is_post_treatment, index >= refit.treatment_index)
        push!(included, refit.included)
        push!(solver_status, refit.solver_status)
        push!(failure_reason, missing)
      end
    else
      push!(specification_id, refit.specification_id)
      push!(time, missing); push!(actual, missing); push!(synthetic, missing); push!(gap, missing)
      push!(is_post_treatment, missing); push!(included, false); push!(solver_status, refit.solver_status)
      push!(failure_reason, refit.failure_reason === nothing ? missing : refit.failure_reason)
    end
  end
  return (
    specification_id=specification_id, time=time, actual=actual,
    synthetic=synthetic, gap=gap, is_post_treatment=is_post_treatment,
    included=included, solver_status=solver_status, failure_reason=failure_reason,
  )
end

"""
    SyntheticControl.specification_weights(result::SpecificationSensitivityResult)

Return stored donor weights with schema `specification_id, donor, weight,
included, solver_status, failure_reason`. Each failed specification
contributes one sentinel row with missing donor and weight. No fitting occurs.

# Examples

```julia
using SyntheticControl, Tables
isdefined(SyntheticControl, :specification_weights)
```
"""
function SyntheticControl.specification_weights(
  result::SyntheticControl.SpecificationSensitivityResult{T,SID,UI}
) where {T,SID,UI}
  specification_id = SID[]
  donor = Union{Missing,UI}[]
  weight = Union{Missing,T}[]
  included = Bool[]
  solver_status = Symbol[]
  failure_reason = Union{Missing,String}[]
  for refit in result.refits
    if refit.solver_status === :success
      length(refit.donor_ids) == length(refit.donor_weights) ||
        throw(DimensionMismatch("stored specification donor weights must match donor identifiers"))
      for index in eachindex(refit.donor_ids)
        push!(specification_id, refit.specification_id)
        push!(donor, refit.donor_ids[index])
        push!(weight, refit.donor_weights[index])
        push!(included, refit.included)
        push!(solver_status, refit.solver_status)
        push!(failure_reason, missing)
      end
    else
      push!(specification_id, refit.specification_id); push!(donor, missing); push!(weight, missing)
      push!(included, false); push!(solver_status, refit.solver_status)
      push!(failure_reason, refit.failure_reason === nothing ? missing : refit.failure_reason)
    end
  end
  return (
    specification_id=specification_id, donor=donor, weight=weight,
    included=included, solver_status=solver_status, failure_reason=failure_reason,
  )
end

"""
    SyntheticControl.specification_balance(result::SpecificationSensitivityResult)

Return stored predictor diagnostics with schema `specification_id, predictor,
treated, synthetic, difference, absolute_difference, relative_difference,
predictor_weight, included, solver_status, failure_reason`. Classic predictor
weights are reported; penalized and failed fits use typed `missing`. Failed
specifications contribute one sentinel row and no calculations occur.

# Examples

```julia
using SyntheticControl, Tables
isdefined(SyntheticControl, :specification_balance)
```
"""
function SyntheticControl.specification_balance(
  result::SyntheticControl.SpecificationSensitivityResult{T,SID}
) where {T,SID}
  specification_id = SID[]
  predictor = Union{Missing,String}[]
  treated = Union{Missing,T}[]
  synthetic = Union{Missing,T}[]
  difference = Union{Missing,T}[]
  absolute_difference = Union{Missing,T}[]
  relative_difference = Union{Missing,T}[]
  predictor_weight = Union{Missing,T}[]
  included = Bool[]
  solver_status = Symbol[]
  failure_reason = Union{Missing,String}[]
  for refit in result.refits
    diagnostics = refit.predictor_diagnostics
    if refit.solver_status === :success && diagnostics !== nothing
      for index in eachindex(diagnostics.predictor)
        push!(specification_id, refit.specification_id)
        push!(predictor, diagnostics.predictor[index])
        push!(treated, diagnostics.treated[index])
        push!(synthetic, diagnostics.synthetic[index])
        push!(difference, diagnostics.difference[index])
        push!(absolute_difference, diagnostics.absolute_difference[index])
        push!(relative_difference, diagnostics.relative_difference[index])
        push!(predictor_weight, refit.predictor_weights === nothing ? missing : refit.predictor_weights[index])
        push!(included, refit.included); push!(solver_status, refit.solver_status); push!(failure_reason, missing)
      end
    else
      push!(specification_id, refit.specification_id); push!(predictor, missing)
      push!(treated, missing); push!(synthetic, missing); push!(difference, missing)
      push!(absolute_difference, missing); push!(relative_difference, missing); push!(predictor_weight, missing)
      push!(included, false); push!(solver_status, refit.solver_status)
      push!(failure_reason, refit.failure_reason === nothing ? missing : refit.failure_reason)
    end
  end
  return (
    specification_id=specification_id, predictor=predictor, treated=treated,
    synthetic=synthetic, difference=difference,
    absolute_difference=absolute_difference, relative_difference=relative_difference,
    predictor_weight=predictor_weight, included=included,
    solver_status=solver_status, failure_reason=failure_reason,
  )
end

"""
    _suite_component_counts(analysis, component)

Read component-level counts from a successful stored suite component.
Specification sensitivity reports no inference-eligible count because it is
a sensitivity analysis rather than a randomization-inference procedure.

# Examples

```julia
isdefined(SyntheticControlTablesExt, :_suite_component_counts)
```
"""
function _suite_component_counts(analysis::Symbol, component)
  if analysis === :specification_sensitivity
    successful = count(refit -> refit.solver_status === :success, component.refits)
    failed = length(component.refits) - successful
    included = count(refit -> refit.included, component.refits)
    return (successful, failed, successful - included, missing)
  end
  counts = SyntheticControl.robustness_counts(component)
  return (
    counts.successful, counts.failed, counts.filtered,
    counts.inference_eligible,
  )
end

"""
    SyntheticControl.robustness_summary(suite::RobustnessSuiteResult)

Return four deterministic rows, in the order `in_space, leave_one_out,
in_time, specification_sensitivity`, with stable schema `analysis,
requested, status, successful, failed, filtered, inference_eligible,
failure_reason`.

Counts describe stored component results only. Disabled, skipped, and
suite-level failed components use typed `missing` counts. Specification
sensitivity has `missing` inference eligibility because it does not perform
randomization inference. Creating the table never runs estimation.

# Examples

```julia
using SyntheticControl, Tables
isdefined(SyntheticControl, :robustness_summary)
```
"""
function SyntheticControl.robustness_summary(
  suite::SyntheticControl.RobustnessSuiteResult,
)
  analyses = (:in_space, :leave_one_out, :in_time, :specification_sensitivity)
  successful = Union{Missing,Int}[]
  failed = Union{Missing,Int}[]
  filtered = Union{Missing,Int}[]
  inference_eligible = Union{Missing,Int}[]
  requested = Bool[]
  status = Symbol[]
  failure_reason = Union{Missing,String}[]

  for analysis in analyses
    state = getproperty(suite.states, analysis)
    component = getproperty(suite, analysis)
    push!(requested, state.requested)
    push!(status, state.status)
    push!(failure_reason, state.failure_reason === nothing ? missing : state.failure_reason)
    if state.status === :success
      component === nothing && throw(ArgumentError("successful suite component $analysis has no stored result"))
      component_counts = _suite_component_counts(analysis, component)
      push!(successful, component_counts[1])
      push!(failed, component_counts[2])
      push!(filtered, component_counts[3])
      push!(inference_eligible, component_counts[4])
    else
      push!(successful, missing)
      push!(failed, missing)
      push!(filtered, missing)
      push!(inference_eligible, missing)
    end
  end

  return (
    analysis=Symbol[analyses...], requested=requested, status=status,
    successful=successful, failed=failed, filtered=filtered,
    inference_eligible=inference_eligible, failure_reason=failure_reason,
  )
end

end
