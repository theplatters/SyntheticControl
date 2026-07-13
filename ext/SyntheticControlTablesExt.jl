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

end
