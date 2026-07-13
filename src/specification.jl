"""
    SCMSpecification(id; pre_periods=nothing, omitted_predictors=String[],
                     predictor_periods=nothing, donors=nothing)
    SCMSpecification{SID,PW,OP,AW,DP}

Describe one synthetic-control sensitivity specification. `pre_periods`
selects outcome fitting periods, `omitted_predictors` removes named
predictors, `predictor_periods` selects periods used to average retained
predictors, and `donors` selects the donor pool. A `nothing` field preserves
that baseline dimension. Inputs are collected and copied; the constructor
does not validate them against a panel or mutate its arguments.

# Examples

```julia
spec = SCMSpecification(:short_pre; pre_periods=[2002, 2003], donors=[:a, :b])
spec.id == :short_pre
```
"""
struct SCMSpecification{SID,PW,OP,AW,DP}
  id::SID
  pre_periods::PW
  omitted_predictors::OP
  predictor_periods::AW
  donors::DP
end

function SCMSpecification(
  id;
  pre_periods=nothing,
  omitted_predictors=String[],
  predictor_periods=nothing,
  donors=nothing
)
  resolved_pre = pre_periods === nothing ? nothing : collect(pre_periods)
  resolved_omitted = String.(collect(omitted_predictors))
  resolved_predictor_periods = predictor_periods === nothing ? nothing : collect(predictor_periods)
  resolved_donors = donors === nothing ? nothing : collect(donors)
  return SCMSpecification(
    id, resolved_pre, resolved_omitted, resolved_predictor_periods, resolved_donors,
  )
end

"""
    SpecificationSensitivityRefit{T,SID,UI,TT,S}

One attempted specification refit. Successful records contain the full
evaluation path, donor and (when defined) predictor weights, scalar and
predictor diagnostics, and the complete solver result. Penalized SCM has no
estimated predictor-weight vector, so `predictor_weights === nothing`.
Failed records retain the specification, status, and exception message with
empty paths and weights.

# Examples

```julia
isdefined(SyntheticControl, :SpecificationSensitivityRefit)
```
"""
struct SpecificationSensitivityRefit{T<:AbstractFloat,SID,UI,TT,S}
  specification::S
  specification_id::SID
  donor_ids::Vector{UI}
  predictor_names::Vector{String}
  time::Vector{TT}
  treatment_index::Int
  result::Union{Nothing,SyntheticControlResult,PenalizedSyntheticControlResult}
  actual::Vector{T}
  synthetic::Vector{T}
  gap::Vector{T}
  donor_weights::Vector{T}
  predictor_weights::Union{Nothing,Vector{T}}
  diagnostics::Union{Nothing,FitDiagnostics{T}}
  predictor_diagnostics::Union{Nothing,PredictorDiagnostics{T}}
  solver_status::Symbol
  failure_reason::Union{Nothing,String}
  included::Bool
  exclusion_reason::Union{Nothing,Symbol}
end

"""
    SpecificationSensitivityResult{T,SID,UI,TT}

Typed batch result for specification sensitivity. `baseline` is the original
fit, `baseline_diagnostics` is calculated once, `panel` retains generic unit
and time types, and `refits` preserves specification input order including
failures.

# Examples

```julia
isdefined(SyntheticControl, :SpecificationSensitivityResult)
```
"""
struct SpecificationSensitivityResult{T<:AbstractFloat,SID,UI,TT}
  baseline::Union{SyntheticControlResult,PenalizedSyntheticControlResult}
  baseline_diagnostics::FitDiagnostics{T}
  panel::SyntheticControlPanelData{T,UI,TT}
  refits::Vector{SpecificationSensitivityRefit}
  minimum_pre_periods::Int
end

"""
    preperiod_specifications(windows; ids=nothing)

Create one [`SCMSpecification`](@ref) per alternative outcome fitting window.
Generated identifiers are `:preperiod_1`, `:preperiod_2`, and so on unless
`ids` is supplied. Windows and identifiers must be nonempty and aligned;
panel-specific validation occurs during [`specification_sensitivity`](@ref).

# Examples

```julia
specs = preperiod_specifications([[1, 2], [1, 2, 3]])
specs[2].id == :preperiod_2
```
"""
function preperiod_specifications(windows; ids=nothing)
  resolved = collect(windows)
  isempty(resolved) && throw(ArgumentError("pre-period windows must not be empty"))
  resolved_ids = ids === nothing ? [Symbol("preperiod_", index) for index in eachindex(resolved)] : collect(ids)
  length(resolved_ids) == length(resolved) || throw(DimensionMismatch("ids must match pre-period windows"))
  return [SCMSpecification(resolved_ids[index]; pre_periods=resolved[index]) for index in eachindex(resolved)]
end

"""
    leave_one_predictor_out(problem; ids=nothing)

Create one specification per baseline predictor, omitting exactly that
predictor. Generated identifiers use `without_<predictor>` unless `ids` is
provided. A one-predictor problem throws because removing its last predictor
is invalid.

# Examples

```julia
problem = SyntheticControlProblem([1.0, 2.0], [1.0], [0.0 2.0; 1.0 3.0],
                                  [0.0 2.0], ["x", "z"], ["a", "b"], "t")
length(leave_one_predictor_out(problem)) == 2
```
"""
function leave_one_predictor_out(problem; ids=nothing)
  names = problem.data.predictor_names
  length(names) > 1 || throw(ArgumentError("cannot remove the last predictor"))
  resolved_ids = ids === nothing ? [Symbol("without_", name) for name in names] : collect(ids)
  length(resolved_ids) == length(names) || throw(DimensionMismatch("ids must match predictors"))
  return [SCMSpecification(resolved_ids[index]; omitted_predictors=[names[index]]) for index in eachindex(names)]
end

"""
    aggregation_period_specifications(windows; ids=nothing)

Create one specification per predictor-aggregation window. Generated
identifiers are `:aggregation_1`, `:aggregation_2`, and so on unless `ids` is
provided. Panel-specific post-treatment leakage validation occurs when run.

# Examples

```julia
aggregation_period_specifications([[1], [1, 2]])[1].predictor_periods == [1]
```
"""
function aggregation_period_specifications(windows; ids=nothing)
  resolved = collect(windows)
  isempty(resolved) && throw(ArgumentError("aggregation windows must not be empty"))
  resolved_ids = ids === nothing ? [Symbol("aggregation_", index) for index in eachindex(resolved)] : collect(ids)
  length(resolved_ids) == length(resolved) || throw(DimensionMismatch("ids must match aggregation windows"))
  return [SCMSpecification(resolved_ids[index]; predictor_periods=resolved[index]) for index in eachindex(resolved)]
end

"""
    donor_pool_specifications(pools; ids=nothing)

Create one specification per alternative donor pool. Generated identifiers
are `:donor_pool_1`, `:donor_pool_2`, and so on unless `ids` is provided.
Unknown, duplicate, treated, and empty pools are validated when run.

# Examples

```julia
donor_pool_specifications([[:a, :b], [:a]])[2].donors == [:a]
```
"""
function donor_pool_specifications(pools; ids=nothing)
  resolved = collect(pools)
  isempty(resolved) && throw(ArgumentError("donor pools must not be empty"))
  resolved_ids = ids === nothing ? [Symbol("donor_pool_", index) for index in eachindex(resolved)] : collect(ids)
  length(resolved_ids) == length(resolved) || throw(DimensionMismatch("ids must match donor pools"))
  return [SCMSpecification(resolved_ids[index]; donors=resolved[index]) for index in eachindex(resolved)]
end

"""
    _specification_period_indices(panel, selected, label; minimum=1)

Resolve an optional collection of observed pre-treatment values to indices in
panel order. `nothing` selects every baseline pre-treatment period. Explicit
collections must be unique, known, strictly pre-treatment, and contain at
least `minimum` observations.

# Examples

```julia
isdefined(SyntheticControl, :_specification_period_indices)
```
"""
function _specification_period_indices(panel, selected, label::AbstractString; minimum::Int=1)
  baseline = collect(1:(panel.treatment_index - 1))
  if selected === nothing
    length(baseline) >= minimum || throw(ArgumentError("$label has fewer than $minimum observations"))
    return baseline
  end
  values = collect(selected)
  isempty(values) && throw(ArgumentError("$label must not be empty"))
  length(unique(values)) == length(values) || throw(ArgumentError("$label contains duplicates"))
  indices = Int[]
  for index in baseline
    panel.time[index] in values && push!(indices, index)
  end
  length(indices) == length(values) || throw(ArgumentError("$label must contain only observed pre-treatment values"))
  length(indices) >= minimum || throw(ArgumentError("$label has fewer than $minimum observations"))
  return indices
end

"""
    _specification_donor_indices(panel, donors)

Resolve an optional donor pool in baseline panel order. `nothing` selects all
non-treated units. Explicit pools must be nonempty, unique, known, and cannot
contain the treated unit.

# Examples

```julia
isdefined(SyntheticControl, :_specification_donor_indices)
```
"""
function _specification_donor_indices(panel, donors)
  baseline = [index for index in eachindex(panel.unit_ids) if index != panel.treated_index]
  donors === nothing && return baseline
  values = collect(donors)
  isempty(values) && throw(ArgumentError("donor pool must not be empty"))
  length(unique(values)) == length(values) || throw(ArgumentError("donor pool contains duplicates"))
  panel.unit_ids[panel.treated_index] in values && throw(ArgumentError("donor pool cannot contain the treated unit"))
  indices = Int[]
  for index in baseline
    panel.unit_ids[index] in values && push!(indices, index)
  end
  length(indices) == length(values) || throw(ArgumentError("donor pool contains an unknown donor"))
  return indices
end

"""
    _projected_predictor_starts(template, kept_predictors, T)

Project a classic problem's configured predictor starts onto retained
predictors, renormalize each column, replace all-zero projections with a
uniform start, and remove exact duplicates while preserving order.

# Examples

```julia
problem = SyntheticControlProblem([1.0, 2.0], [1.0], [0.0 2.0; 1.0 3.0],
                                  [0.0 2.0], ["x", "z"], ["a", "b"], "t")
size(SyntheticControl._projected_predictor_starts(problem, [1], Float64), 1) == 1
```
"""
function _projected_predictor_starts(template::SyntheticControlProblem, kept_predictors, ::Type{T}) where {T}
  starts = Vector{Vector{T}}()
  count = length(kept_predictors)
  for column in axes(template.starts, 2)
    candidate = T[template.starts[index, column] for index in kept_predictors]
    total = sum(candidate)
    if total > zero(T)
      candidate ./= total
    else
      fill!(candidate, one(T) / T(count))
    end
    any(existing -> existing == candidate, starts) || push!(starts, candidate)
  end
  isempty(starts) && push!(starts, fill(one(T) / T(count), count))
  return reduce(hcat, starts)
end

"""
    _specification_problem(template, data, kept_predictors)

Construct an independent problem with the baseline estimator configuration.
Classic SCM preserves stopping thresholds and projects configured predictor
starts when predictors are removed; penalized SCM preserves `lambda`.

# Examples

```julia
isdefined(SyntheticControl, :_specification_problem)
```
"""
function _specification_problem(
  template::SyntheticControlProblem{T}, data::SyntheticControlData{T}, kept_predictors
) where {T}
  problem = SyntheticControlProblem(
    data;
    target_mspe=template.target_mspe,
    min_relative_mspe_improvement=template.min_relative_mspe_improvement,
  )
  problem.starts = _projected_predictor_starts(template, kept_predictors, T)
  problem.nstarts = size(problem.starts, 2)
  problem.inner_caches = [InnerWeightCache(data) for _ in 1:problem.nstarts]
  problem.search_results = OuterSearchResults(
    Matrix{T}(undef, length(data.predictor_names), problem.nstarts),
    Matrix{T}(undef, length(data.donor_ids), problem.nstarts),
    Vector{T}(undef, problem.nstarts),
  )
  return problem
end

function _specification_problem(
  template::PenalizedSyntheticControlProblem{T}, data::SyntheticControlData{T}, kept_predictors
) where {T}
  return PenalizedSyntheticControlProblem(data; lambda=template.lambda)
end

"""
    _specification_data(panel, specification, minimum_pre_periods)

Validate and materialize one specification as aligned SCM data. Returns the
data plus resolved donor, predictor, fitting-period, and aggregation-period
indices. Predictor aggregation defaults to the baseline pre-period even when
the outcome fitting window changes.

# Examples

```julia
isdefined(SyntheticControl, :_specification_data)
```
"""
function _specification_data(panel, specification::SCMSpecification, minimum_pre_periods::Int)
  pre = _specification_period_indices(
    panel, specification.pre_periods, "pre-treatment fitting window";
    minimum=minimum_pre_periods,
  )
  aggregation = _specification_period_indices(
    panel, specification.predictor_periods, "predictor aggregation window";
    minimum=1,
  )
  donors = _specification_donor_indices(panel, specification.donors)
  omitted = specification.omitted_predictors
  length(unique(omitted)) == length(omitted) || throw(ArgumentError("omitted predictors contains duplicates"))
  all(name -> name in panel.predictor_names, omitted) || throw(ArgumentError("omitted predictors contains an unknown predictor"))
  kept = [index for index in eachindex(panel.predictor_names) if !(panel.predictor_names[index] in omitted)]
  isempty(kept) && throw(ArgumentError("a specification cannot remove the last predictor"))

  T = eltype(panel.outcomes)
  X1 = Vector{T}(undef, length(kept))
  X0 = Matrix{T}(undef, length(kept), length(donors))
  for (row, predictor) in pairs(kept)
    X1[row] = sum(panel.predictors[aggregation, predictor, panel.treated_index]) / T(length(aggregation))
    for (column, donor) in pairs(donors)
      X0[row, column] = sum(panel.predictors[aggregation, predictor, donor]) / T(length(aggregation))
    end
  end
  Y1 = collect(panel.outcomes[pre, panel.treated_index])
  Y0 = Matrix(panel.outcomes[pre, donors])
  names = collect(panel.predictor_names[kept])
  data = SyntheticControlData(
    X1, Y1, X0, Y0, names, string.(panel.unit_ids[donors]),
    string(panel.unit_ids[panel.treated_index]),
  )
  return data, donors, kept, pre, aggregation
end

"""
    _successful_specification_refit(problem, panel, specification, minimum_pre_periods)

Reconstruct, solve, and diagnose one validated specification using an
independent problem cache. Returns a successful
[`SpecificationSensitivityRefit`](@ref) evaluated over the full baseline
panel.

# Examples

```julia
isdefined(SyntheticControl, :_successful_specification_refit)
```
"""
function _successful_specification_refit(problem, panel, specification, minimum_pre_periods)
  data, donors, kept, _pre, _aggregation = _specification_data(
    panel, specification, minimum_pre_periods,
  )
  refit_problem = _specification_problem(problem, data, kept)
  fit = solve(refit_problem)
  all(isfinite, fit.W) || throw(ErrorException("solver returned non-finite donor weights"))
  isfinite(fit.mspe) || throw(ErrorException("solver returned non-finite MSPE"))
  actual = collect(panel.outcomes[:, panel.treated_index])
  synthetic = collect(panel.outcomes[:, donors] * fit.W)
  gap = outcome_gap(actual, synthetic)
  diagnostics = fit_diagnostics(fit)
  predictor = predictor_diagnostics(fit)
  predictor_weights = fit isa SyntheticControlResult ? copy(fit.V) : nothing
  T = eltype(panel.outcomes)
  SID = typeof(specification.id)
  UI = eltype(panel.unit_ids)
  TT = eltype(panel.time)
  return SpecificationSensitivityRefit{T,SID,UI,TT,typeof(specification)}(
    specification, specification.id, collect(panel.unit_ids[donors]),
    copy(data.predictor_names), copy(panel.time), panel.treatment_index, fit,
    actual, synthetic, gap, copy(fit.W), predictor_weights, diagnostics,
    predictor, :success, nothing, true, nothing,
  )
end

"""
    _failed_specification_refit(panel, specification, error)

Create a retained failed specification record with empty fitted values and
the rendered exception message. No error is swallowed unless the caller has
selected `on_failure=:record`.

# Examples

```julia
isdefined(SyntheticControl, :_failed_specification_refit)
```
"""
function _failed_specification_refit(panel, specification, error)
  T = eltype(panel.outcomes)
  SID = typeof(specification.id)
  UI = eltype(panel.unit_ids)
  TT = eltype(panel.time)
  return SpecificationSensitivityRefit{T,SID,UI,TT,typeof(specification)}(
    specification, specification.id, UI[], String[], TT[], panel.treatment_index,
    nothing, T[], T[], T[], T[], nothing, nothing, nothing, :failed,
    sprint(showerror, error), false, :failed,
  )
end

"""
    specification_sensitivity(problem, solution, specifications;
                              on_failure=:record, minimum_pre_periods=2)
    specification_sensitivity(problem, specifications; kwargs...)

Completely refit a batch of [`SCMSpecification`](@ref) objects against the
full panel retained by [`from_table`](@ref). The baseline problem and fit are
never mutated. Unspecified dimensions preserve baseline fitting periods,
predictor aggregation, predictors, donor pool, estimator, and solver
configuration. Every successful specification is evaluated on the same full
observed panel.

Input order is preserved. Specification identifiers and definitions must be
unique, and identifiers must share one concrete type. `minimum_pre_periods`
defaults to two. Invalid specifications and solver errors are retained with
`on_failure=:record` or rethrown with `:error`. Returns a typed
[`SpecificationSensitivityResult`](@ref).

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=5), time=repeat(1:5, 3),
         y=[2.,3,4,6,8, 1,2,3,4,5, 3,4,5,6,7],
         x=[2.,3,4,5,6, 1,2,3,4,5, 3,4,5,6,7])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=4)
result = specification_sensitivity(
  problem, solve(problem), [SCMSpecification(:short; pre_periods=[2, 3])],
)
only(result.refits).solver_status == :success
```
"""
function specification_sensitivity(
  problem,
  solution,
  specifications;
  on_failure::Symbol=:record,
  minimum_pre_periods::Int=2,
)
  _validate_original(problem, solution)
  _validate_failure_policy(on_failure)
  minimum_pre_periods >= 1 || throw(ArgumentError("minimum_pre_periods must be positive"))
  panel = _panel_data(problem)
  specs = collect(specifications)
  isempty(specs) && throw(ArgumentError("specifications must not be empty"))
  all(spec -> spec isa SCMSpecification, specs) || throw(ArgumentError("every specification must be an SCMSpecification"))
  id_type = typeof(first(specs).id)
  all(spec -> typeof(spec.id) === id_type, specs) || throw(ArgumentError("specification identifiers must share one concrete type"))
  ids = [spec.id for spec in specs]
  length(unique(ids)) == length(ids) || throw(ArgumentError("specification identifiers must be unique"))
  definitions = [
    (spec.pre_periods, spec.omitted_predictors, spec.predictor_periods, spec.donors)
    for spec in specs
  ]
  length(unique(definitions)) == length(definitions) || throw(ArgumentError("duplicate specification definitions are not allowed"))

  refits = Vector{SpecificationSensitivityRefit}(undef, length(specs))
  for (index, specification) in pairs(specs)
    try
      refits[index] = _successful_specification_refit(
        problem, panel, specification, minimum_pre_periods,
      )
    catch error
      on_failure === :error && rethrow()
      refits[index] = _failed_specification_refit(panel, specification, error)
    end
  end
  T = eltype(panel.outcomes)
  UI = eltype(panel.unit_ids)
  TT = eltype(panel.time)
  return SpecificationSensitivityResult{T,id_type,UI,TT}(
    solution, fit_diagnostics(solution), panel, refits, minimum_pre_periods,
  )
end

function specification_sensitivity(problem, specifications; kwargs...)
  return specification_sensitivity(problem, solve(problem), specifications; kwargs...)
end
