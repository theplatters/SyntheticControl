"""
    InSpaceOptions(; units=nothing, include_treated=false, rmspe_cutoff=nothing,
                   on_failure=:record, parallel=false)

Advanced options for the in-space component of [`robustness_suite`](@ref).
Fields map directly to [`in_space_placebos`](@ref).

# Examples

```julia
InSpaceOptions(rmspe_cutoff=5.0).rmspe_cutoff == 5.0
```
"""
struct InSpaceOptions{U,C}
  units::U
  include_treated::Bool
  rmspe_cutoff::C
  on_failure::Symbol
  parallel::Bool
end

function InSpaceOptions(;
  units=nothing,
  include_treated::Bool=false,
  rmspe_cutoff=nothing,
  on_failure::Symbol=:record,
  parallel::Bool=false,
)
  resolved_units = units === nothing ? nothing : collect(units)
  return InSpaceOptions(resolved_units, include_treated, rmspe_cutoff, on_failure, parallel)
end

"""
    LeaveOneOutOptions(; donors=nothing, active_only=false, weight_tol=nothing,
                       on_failure=:record, parallel=false)

Advanced options for the leave-one-out component of
[`robustness_suite`](@ref). A `nothing` weight tolerance uses the component
API's element-type default.

# Examples

```julia
LeaveOneOutOptions(active_only=true).active_only
```
"""
struct LeaveOneOutOptions{D,W}
  donors::D
  active_only::Bool
  weight_tol::W
  on_failure::Symbol
  parallel::Bool
end

function LeaveOneOutOptions(;
  donors=nothing,
  active_only::Bool=false,
  weight_tol=nothing,
  on_failure::Symbol=:record,
  parallel::Bool=false,
)
  resolved_donors = donors === nothing ? nothing : collect(donors)
  return LeaveOneOutOptions(resolved_donors, active_only, weight_tol, on_failure, parallel)
end

"""
    InTimeOptions(; placebo_times, post_window=nothing,
                  stop_before_treatment=true, minimum_pre_periods=2,
                  minimum_post_periods=1, on_failure=:record, parallel=false)

Advanced options for the in-time component of [`robustness_suite`](@ref).
Fields map directly to [`in_time_placebos`](@ref). `placebo_times` is copied
and must be supplied.

# Examples

```julia
InTimeOptions(placebo_times=[3, 4]).placebo_times == [3, 4]
```
"""
struct InTimeOptions{TT,PW}
  placebo_times::Vector{TT}
  post_window::PW
  stop_before_treatment::Bool
  minimum_pre_periods::Int
  minimum_post_periods::Int
  on_failure::Symbol
  parallel::Bool
end

function InTimeOptions(;
  placebo_times,
  post_window=nothing,
  stop_before_treatment::Bool=true,
  minimum_pre_periods::Int=2,
  minimum_post_periods::Int=1,
  on_failure::Symbol=:record,
  parallel::Bool=false,
)
  resolved_times = collect(placebo_times)
  return InTimeOptions(
    resolved_times, post_window, stop_before_treatment, minimum_pre_periods,
    minimum_post_periods, on_failure, parallel,
  )
end

"""
    SpecificationSensitivityOptions(specifications; on_failure=:record,
                                    minimum_pre_periods=2)

Advanced options for the specification-sensitivity component of
[`robustness_suite`](@ref). Specifications are copied and passed to
[`specification_sensitivity`](@ref).

# Examples

```julia
options = SpecificationSensitivityOptions([SCMSpecification(:baseline)])
length(options.specifications) == 1
```
"""
struct SpecificationSensitivityOptions{S}
  specifications::Vector{S}
  on_failure::Symbol
  minimum_pre_periods::Int
end


function SpecificationSensitivityOptions(
  specifications;
  on_failure::Symbol=:record,
  minimum_pre_periods::Int=2,
)
  resolved = collect(specifications)
  return SpecificationSensitivityOptions(resolved, on_failure, minimum_pre_periods)
end

"""
    RobustnessAnalysisState(requested, status; failure_reason=nothing)

Explicit suite-level state for one component. `status` is `:success`,
`:disabled`, `:skipped`, or `:failed`. Disabled components were not requested;
skipped components were requested without required configuration; failed
components threw under the suite's recording policy.

# Examples

```julia
RobustnessAnalysisState(false, :disabled).status == :disabled
```
"""
struct RobustnessAnalysisState
  requested::Bool
  status::Symbol
  failure_reason::Union{Nothing,String}

  function RobustnessAnalysisState(requested::Bool, status::Symbol; failure_reason=nothing)
    status in (:success, :disabled, :skipped, :failed) ||
      throw(ArgumentError("suite analysis status must be :success, :disabled, :skipped, or :failed"))
    reason = failure_reason === nothing ? nothing : String(failure_reason)
    return new(requested, status, reason)
  end
end

"""
    RobustnessSuiteResult

Typed orchestration result retaining the baseline fit and diagnostics plus
the unchanged component result types. Disabled, skipped, and suite-level
failed components use `nothing`; `states` distinguishes those cases and
stores failure reasons.

# Examples

```julia
isdefined(SyntheticControl, :RobustnessSuiteResult)
```
"""
struct RobustnessSuiteResult{B,D,IS,LO,IT,SS,ST}
  baseline::B
  diagnostics::D
  in_space::IS
  leave_one_out::LO
  in_time::IT
  specification_sensitivity::SS
  states::ST
end


"""
    _run_suite_component(operation; suite_failure=:record)

Run one requested component and return `(result, state)`. With
`suite_failure=:record`, thrown component errors become `:failed` states;
`:error` rethrows. Successful component results are returned unchanged.

# Examples

```julia
value, state = SyntheticControl._run_suite_component(() -> 1)
(value, state.status) == (1, :success)
```
"""
function _run_suite_component(operation; suite_failure::Symbol=:record)
  suite_failure in (:record, :error) || throw(ArgumentError("suite_failure must be :record or :error"))
  try
    return operation(), RobustnessAnalysisState(true, :success)
  catch error
    suite_failure === :error && rethrow()
    return nothing, RobustnessAnalysisState(
      true, :failed; failure_reason=sprint(showerror, error),
    )
  end
end


"""
    robustness_suite(problem, fit; in_space=false, leave_one_out=false,
                     in_time=false, specification_sensitivity=false,
                     placebo_times=nothing, specifications=nothing,
                     suite_failure=:record)
    robustness_suite(fit; kwargs...)

Run selected robustness components through their existing public APIs. The
suite is orchestration only: it reuses `fit`, never resolves the baseline
again, and preserves component result types. The fit-only method recovers the
problem registered by `solve`; manually constructed fits must use the
explicit `(problem, fit)` method.

`in_space=true` uses defaults or accepts [`InSpaceOptions`](@ref).
`leave_one_out=true` checks all donors, `:active` checks positive-weight
donors, and [`LeaveOneOutOptions`](@ref) supplies advanced settings.
`placebo_times=[...]` enables in-time defaults; alternatively pass
[`InTimeOptions`](@ref) through `in_time`. `specifications=[...]` enables
specification sensitivity, or pass [`SpecificationSensitivityOptions`](@ref).

Disabled analyses are `nothing` with state `:disabled`. Requested `in_time`
or specification sensitivity without required inputs is `:skipped`.
Component exceptions are isolated as `:failed` with `suite_failure=:record`
or rethrown with `:error`. Individual refit failure policies remain those of
their component options. Result and assignment ordering is deterministic.

# Examples

```julia
using SyntheticControl, CommonSolve, Tables
panel = (unit=repeat([:t, :a, :b], inner=5), time=repeat(1:5, 3),
         y=[2.,3,4,6,8, 1,2,3,4,5, 3,4,5,6,7],
         x=[2.,3,4,5,6, 1,2,3,4,5, 3,4,5,6,7])
problem = from_table(panel; unit=:unit, time=:time, outcome=:y,
                     predictors=[:x], treated=:t, treatment_time=4)
fit = solve(problem)
suite = robustness_suite(fit; in_space=true, leave_one_out=:active,
                         placebo_times=[3])
suite.states.in_space.status == :success
```
"""
function robustness_suite(
  problem,
  fit;
  in_space=false,
  leave_one_out=false,
  in_time=false,
  specification_sensitivity=false,
  placebo_times=nothing,
  specifications=nothing,
  suite_failure::Symbol=:record,
)
  _validate_original(problem, fit)
  _panel_data(problem)
  suite_failure in (:record, :error) || throw(ArgumentError("suite_failure must be :record or :error"))
  diagnostics = fit_diagnostics(fit)

  in_space_result, in_space_state = if in_space === false
    nothing, RobustnessAnalysisState(false, :disabled)
  else
    options = in_space === true ? InSpaceOptions() : in_space
    options isa InSpaceOptions || throw(ArgumentError("in_space must be true, false, or InSpaceOptions"))
    _run_suite_component(; suite_failure=suite_failure) do
      in_space_placebos(
        problem, fit; units=options.units, include_treated=options.include_treated,
        rmspe_cutoff=options.rmspe_cutoff, on_failure=options.on_failure,
        parallel=options.parallel,
      )
    end
  end

  leave_result, leave_state = if leave_one_out === false
    nothing, RobustnessAnalysisState(false, :disabled)
  else
    options = if leave_one_out === true
      LeaveOneOutOptions()
    elseif leave_one_out === :active
      LeaveOneOutOptions(active_only=true)
    else
      leave_one_out
    end
    options isa LeaveOneOutOptions || throw(ArgumentError(
      "leave_one_out must be true, false, :active, or LeaveOneOutOptions"
    ))
    _run_suite_component(; suite_failure=suite_failure) do
      if options.weight_tol === nothing
        SyntheticControl.leave_one_out(
          problem, fit; donors=options.donors, active_only=options.active_only,
          on_failure=options.on_failure, parallel=options.parallel,
        )
      else
        SyntheticControl.leave_one_out(
          problem, fit; donors=options.donors, active_only=options.active_only,
          weight_tol=options.weight_tol, on_failure=options.on_failure,
          parallel=options.parallel,
        )
      end
    end
  end

  in_time_result, in_time_state = if in_time === false && placebo_times === nothing
    nothing, RobustnessAnalysisState(false, :disabled)
  elseif in_time === true && placebo_times === nothing
    nothing, RobustnessAnalysisState(
      true, :skipped; failure_reason="placebo_times are required for in-time analysis",
    )
  else
    (in_time === false || in_time === true || in_time isa InTimeOptions) ||
      throw(ArgumentError("in_time must be true, false, or InTimeOptions"))
    in_time isa InTimeOptions && placebo_times !== nothing && throw(ArgumentError(
      "pass placebo times through either in_time=InTimeOptions(...) or placebo_times, not both"
    ))
    options = in_time isa InTimeOptions ? in_time : InTimeOptions(placebo_times=placebo_times)
    _run_suite_component(; suite_failure=suite_failure) do
      in_time_placebos(
        problem, fit; placebo_times=options.placebo_times,
        post_window=options.post_window,
        stop_before_treatment=options.stop_before_treatment,
        minimum_pre_periods=options.minimum_pre_periods,
        minimum_post_periods=options.minimum_post_periods,
        on_failure=options.on_failure, parallel=options.parallel,
      )
    end
  end

  specification_result, specification_state = if specification_sensitivity === false &&
    specifications === nothing
    nothing, RobustnessAnalysisState(false, :disabled)
  elseif specification_sensitivity === true && specifications === nothing
    nothing, RobustnessAnalysisState(
      true, :skipped; failure_reason="specifications are required for specification sensitivity",
    )
  else
    (specification_sensitivity === false || specification_sensitivity === true ||
      specification_sensitivity isa SpecificationSensitivityOptions) ||
      throw(ArgumentError(
        "specification_sensitivity must be true, false, or SpecificationSensitivityOptions",
      ))
    specification_sensitivity isa SpecificationSensitivityOptions && specifications !== nothing &&
      throw(ArgumentError("pass specifications through either SpecificationSensitivityOptions or specifications, not both"))
    options = specification_sensitivity isa SpecificationSensitivityOptions ?
      specification_sensitivity : SpecificationSensitivityOptions(specifications)
    _run_suite_component(; suite_failure=suite_failure) do
      SyntheticControl.specification_sensitivity(
        problem, fit, options.specifications; on_failure=options.on_failure,
        minimum_pre_periods=options.minimum_pre_periods,
      )
    end
  end

  states = (
    in_space=in_space_state,
    leave_one_out=leave_state,
    in_time=in_time_state,
    specification_sensitivity=specification_state,
  )
  return RobustnessSuiteResult(
    fit, diagnostics, in_space_result, leave_result, in_time_result,
    specification_result, states,
  )
end


function robustness_suite(fit; kwargs...)
  return robustness_suite(_problem_for_solution(fit), fit; kwargs...)
end
