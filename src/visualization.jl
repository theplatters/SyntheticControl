abstract type AbstractSyntheticControlResultLike end

const SyntheticControlResultLike = Union{SyntheticControlResult,PenalizedSyntheticControlResult}

"""
    SyntheticControlPathData(time, treatment_index, treatment_time, actual, synthetic, treated_id)
    SyntheticControlPathData(result; time=nothing, treatment_index=nothing, treatment_time=nothing,
                             Y1_post=nothing, Y0_post=nothing)

Validated outcome paths for plotting one treated unit and its synthetic
counterfactual. `time`, `actual`, and `synthetic` must have equal lengths.
`treatment_index` is the first post-treatment observation, so observations
`1:(treatment_index - 1)` are pre-treatment and `treatment_index:end` are
post-treatment. `treatment_time` is the coordinate where recipes draw the
vertical treatment marker.

The result constructor uses the package's existing result object and appends
optional post-treatment outcomes. `Y1_post` is a vector of treated outcomes
and `Y0_post` is a matrix with one row per post-treatment period and one
column per donor. If no post-treatment data are supplied, the stored
pre-treatment paths are used and the default treatment marker is placed after
the final pre-treatment observation. The constructor does not mutate the
result.

# Examples

```julia
using SyntheticControl, CommonSolve

problem = SyntheticControlProblem([1.0], [1.0, 2.0], reshape([0.5, 1.5], 1, 2),
                                  [0.8 1.2; 1.8 2.2],
                                  ["p"], ["d1", "d2"], "treated")
result = solve(problem)
paths = SyntheticControlPathData(result; Y1_post=[3.0], Y0_post=[2.8 3.2])
paths.treatment_index == 3
```
"""
struct SyntheticControlPathData{T<:AbstractFloat,TT}
  time::Vector{TT}
  treatment_index::Int
  treatment_time::TT
  actual::Vector{T}
  synthetic::Vector{T}
  treated_id::String

  function SyntheticControlPathData(
    time::AbstractVector{TT},
    treatment_index::Int,
    treatment_time::TT,
    actual::AbstractVector{T},
    synthetic::AbstractVector{T},
    treated_id::AbstractString
  ) where {T<:AbstractFloat,TT}
    length(time) == length(actual) || throw(DimensionMismatch("time length must match actual outcome length"))
    length(synthetic) == length(actual) || throw(DimensionMismatch("synthetic outcome length must match actual outcome length"))
    !isempty(actual) || throw(ArgumentError("outcome paths must contain at least one observation"))
    1 <= treatment_index <= length(actual) + 1 || throw(ArgumentError("treatment_index must be between 1 and length(actual) + 1"))
    treatment_time in time || treatment_index == length(actual) + 1 || throw(ArgumentError("treatment_time must appear in time unless treatment_index is length(actual) + 1"))
    _assert_finite_outcomes("actual", actual)
    _assert_finite_outcomes("synthetic", synthetic)
    return new{T,TT}(collect(time), treatment_index, treatment_time, collect(actual), collect(synthetic), String(treated_id))
  end
end

"""
    SyntheticControlPlaceboResult(time, treatment_index, treatment_time, treated_id,
                                  placebo_ids, treated_actual, treated_synthetic,
                                  placebo_actual, placebo_synthetic)

Validated placebo diagnostic data for plotting treated and placebo gaps and
RMSPE ratios. `placebo_actual` and `placebo_synthetic` must be `T × J`
matrices aligned to `time`, with one column per placebo unit.

This type stores prepared paths only; it does not estimate placebo synthetic
controls. Use it after computing placebo paths with the same estimation
workflow used for the treated unit.

# Examples

```julia
time = 1:4
placebo = SyntheticControlPlaceboResult(
  time, 3, 3, "treated", ["p1", "p2"],
  [1.0, 2.0, 4.0, 5.0], [1.0, 1.5, 2.0, 2.5],
  [1.0 1.0; 2.0 1.5; 3.0 2.0; 4.0 2.5],
  [1.0 0.8; 1.5 1.2; 2.0 1.6; 2.5 2.0],
)
length(placebo.placebo_ids) == 2
```
"""
struct SyntheticControlPlaceboResult{T<:AbstractFloat,TT}
  time::Vector{TT}
  treatment_index::Int
  treatment_time::TT
  treated_id::String
  placebo_ids::Vector{String}
  treated_actual::Vector{T}
  treated_synthetic::Vector{T}
  placebo_actual::Matrix{T}
  placebo_synthetic::Matrix{T}

  function SyntheticControlPlaceboResult(
    time::AbstractVector{TT},
    treatment_index::Int,
    treatment_time::TT,
    treated_id::AbstractString,
    placebo_ids::AbstractVector{<:AbstractString},
    treated_actual::AbstractVector{T},
    treated_synthetic::AbstractVector{T},
    placebo_actual::AbstractMatrix{T},
    placebo_synthetic::AbstractMatrix{T}
  ) where {T<:AbstractFloat,TT}
    nperiods = length(time)
    nperiods == length(treated_actual) || throw(DimensionMismatch("time length must match treated_actual length"))
    length(treated_synthetic) == nperiods || throw(DimensionMismatch("treated_synthetic length must match time length"))
    size(placebo_actual, 1) == nperiods || throw(DimensionMismatch("placebo_actual rows must match time length"))
    size(placebo_synthetic, 1) == nperiods || throw(DimensionMismatch("placebo_synthetic rows must match time length"))
    size(placebo_actual, 2) == length(placebo_ids) || throw(DimensionMismatch("placebo_actual columns must match placebo_ids length"))
    size(placebo_synthetic, 2) == length(placebo_ids) || throw(DimensionMismatch("placebo_synthetic columns must match placebo_ids length"))
    !isempty(placebo_ids) || throw(ArgumentError("placebo_ids must contain at least one placebo unit"))
    1 <= treatment_index <= nperiods || throw(ArgumentError("treatment_index must identify the first post-treatment observation"))
    treatment_time in time || throw(ArgumentError("treatment_time must appear in time"))
    _assert_finite_outcomes("treated_actual", treated_actual)
    _assert_finite_outcomes("treated_synthetic", treated_synthetic)
    _assert_finite_outcomes("placebo_actual", placebo_actual)
    _assert_finite_outcomes("placebo_synthetic", placebo_synthetic)
    return new{T,TT}(
      collect(time),
      treatment_index,
      treatment_time,
      String(treated_id),
      String.(placebo_ids),
      collect(treated_actual),
      collect(treated_synthetic),
      collect(placebo_actual),
      collect(placebo_synthetic)
    )
  end
end

function _assert_finite_outcomes(name::AbstractString, x)
  for value in x
    ismissing(value) && throw(ArgumentError("$name contains missing values"))
    isfinite(value) || throw(ArgumentError("$name contains non-finite values"))
  end
  return nothing
end

function _default_time(nperiods::Int)
  return collect(1:nperiods)
end

function _default_treatment_time(time, treatment_index::Int)
  if treatment_index <= length(time)
    return time[treatment_index]
  end
  last_time = time[end]
  return last_time isa Real ? last_time + one(last_time) : last_time
end

function _validate_result_weights(result::SyntheticControlResultLike)
  length(result.W) == size(result.data.Y0, 2) || throw(DimensionMismatch("result donor weights must match Y0 columns"))
  return nothing
end

"""
    actual_outcome(result; Y1_post=nothing)
    actual_outcome(paths::SyntheticControlPathData)

Return the treated unit's observed outcome path. For result objects, the
pre-treatment path `result.data.Y1` is returned with optional `Y1_post`
appended. The returned vector is newly allocated.

Throws `ArgumentError` if supplied outcomes contain missing or non-finite
values.

# Examples

```julia
problem = SyntheticControlProblem([1.0], [1.0, 2.0], reshape([0.5, 1.5], 1, 2),
                                  [0.8 1.2; 1.8 2.2],
                                  ["p"], ["d1", "d2"], "treated")
result = solve(problem)
SyntheticControl.actual_outcome(result; Y1_post=[3.0]) == [1.0, 2.0, 3.0]
```
"""
function actual_outcome(result::SyntheticControlResultLike; Y1_post=nothing)
  actual = Y1_post === nothing ? copy(result.data.Y1) : vcat(result.data.Y1, eltype(result.data.Y1).(collect(Y1_post)))
  _assert_finite_outcomes("actual outcome", actual)
  return actual
end

actual_outcome(paths::SyntheticControlPathData) = copy(paths.actual)

"""
    synthetic_outcome(result; Y0_post=nothing)
    synthetic_outcome(paths::SyntheticControlPathData)

Return the synthetic-control outcome path `Y0 * W`. For result objects, the
pre-treatment fitted path is returned with optional post-treatment donor
outcomes `Y0_post * result.W` appended. The returned vector is newly
allocated.

Throws `DimensionMismatch` when donor outcome columns do not match the result
weights and `ArgumentError` for missing or non-finite values.

# Examples

```julia
problem = SyntheticControlProblem([1.0], [1.0, 2.0], reshape([0.5, 1.5], 1, 2),
                                  [0.8 1.2; 1.8 2.2],
                                  ["p"], ["d1", "d2"], "treated")
result = solve(problem)
length(SyntheticControl.synthetic_outcome(result; Y0_post=[2.8 3.2])) == 3
```
"""
function synthetic_outcome(result::SyntheticControlResultLike; Y0_post=nothing)
  _validate_result_weights(result)
  fitted_pre = result.data.Y0 * result.W
  if Y0_post === nothing
    synthetic = fitted_pre
  else
    size(Y0_post, 2) == length(result.W) || throw(DimensionMismatch("Y0_post columns must match result donor weights"))
    synthetic = vcat(fitted_pre, Y0_post * result.W)
  end
  _assert_finite_outcomes("synthetic outcome", synthetic)
  return synthetic
end

synthetic_outcome(paths::SyntheticControlPathData) = copy(paths.synthetic)

"""
    SyntheticControlPathData(result; time=nothing, treatment_index=nothing,
                             treatment_time=nothing, Y1_post=nothing, Y0_post=nothing)

Build validated plotting paths from an existing synthetic-control result. See
[`SyntheticControlPathData`](@ref) for field meanings.
"""
function SyntheticControlPathData(
  result::SyntheticControlResultLike;
  time=nothing,
  treatment_index::Union{Nothing,Int}=nothing,
  treatment_time=nothing,
  Y1_post=nothing,
  Y0_post=nothing
)
  actual = actual_outcome(result; Y1_post=Y1_post)
  synthetic = synthetic_outcome(result; Y0_post=Y0_post)
  length(actual) == length(synthetic) || throw(DimensionMismatch("actual and synthetic outcomes must have the same length"))
  resolved_time = time === nothing ? _default_time(length(actual)) : collect(time)
  resolved_treatment_index = treatment_index === nothing ? length(result.data.Y1) + 1 : treatment_index
  resolved_treatment_time = treatment_time === nothing ? _default_treatment_time(resolved_time, resolved_treatment_index) : treatment_time
  return SyntheticControlPathData(
    resolved_time,
    resolved_treatment_index,
    resolved_treatment_time,
    actual,
    synthetic,
    result.data.treated_id
  )
end

"""
    outcome_gap(actual, synthetic)
    outcome_gap(result; Y1_post=nothing, Y0_post=nothing)
    outcome_gap(paths::SyntheticControlPathData)

Return the estimated treatment-effect path `actual .- synthetic`.

Arguments must have compatible lengths and finite values. The returned vector
is newly allocated and inputs are not mutated.

# Examples

```julia
SyntheticControl.outcome_gap([2.0, 4.0], [1.5, 3.0]) == [0.5, 1.0]
```
"""
function outcome_gap(actual::AbstractVector{T}, synthetic::AbstractVector{T}) where {T<:AbstractFloat}
  length(actual) == length(synthetic) || throw(DimensionMismatch("actual and synthetic outcomes must have the same length"))
  _assert_finite_outcomes("actual outcome", actual)
  _assert_finite_outcomes("synthetic outcome", synthetic)
  return actual .- synthetic
end

outcome_gap(result::SyntheticControlResultLike; Y1_post=nothing, Y0_post=nothing) = outcome_gap(
  actual_outcome(result; Y1_post=Y1_post),
  synthetic_outcome(result; Y0_post=Y0_post)
)

outcome_gap(paths::SyntheticControlPathData) = outcome_gap(paths.actual, paths.synthetic)

"""
    placebo_gaps(placebo::SyntheticControlPlaceboResult)

Return a matrix of placebo gaps, one column per placebo unit, computed as
`placebo_actual .- placebo_synthetic`.

# Examples

```julia
placebo = SyntheticControlPlaceboResult(1:2, 2, 2, "t", ["p"],
                                        [1.0, 2.0], [0.5, 1.5],
                                        reshape([1.0, 2.0], 2, 1),
                                        reshape([0.5, 1.0], 2, 1))
SyntheticControl.placebo_gaps(placebo) == reshape([0.5, 1.0], 2, 1)
```
"""
function placebo_gaps(placebo::SyntheticControlPlaceboResult)
  return placebo.placebo_actual .- placebo.placebo_synthetic
end

function _segment(values::AbstractVector, treatment_index::Int, segment::Symbol)
  1 <= treatment_index <= length(values) || throw(ArgumentError("treatment_index must identify the first post-treatment observation"))
  if segment === :pre
    treatment_index > 1 || throw(ArgumentError("at least one pre-treatment observation is required"))
    return @view values[1:(treatment_index-1)]
  elseif segment === :post
    treatment_index <= length(values) || throw(ArgumentError("at least one post-treatment observation is required"))
    return @view values[treatment_index:end]
  end
  throw(ArgumentError("unknown RMSPE segment $segment"))
end

function _rmspe(values)
  _assert_finite_outcomes("gap", values)
  total = zero(eltype(values))
  for value in values
    total += value * value
  end
  return sqrt(total / length(values))
end

"""
    pre_treatment_rmspe(gaps, treatment_index)
    pre_treatment_rmspe(paths::SyntheticControlPathData)

Return the root mean squared prediction error over observations before
`treatment_index`.

Throws `ArgumentError` if no pre-treatment observations exist or gaps are
missing or non-finite.

# Examples

```julia
SyntheticControl.pre_treatment_rmspe([0.0, 2.0, 10.0], 3) ≈ sqrt(2)
```
"""
pre_treatment_rmspe(gaps::AbstractVector{T}, treatment_index::Int) where {T<:AbstractFloat} = _rmspe(_segment(gaps, treatment_index, :pre))
pre_treatment_rmspe(paths::SyntheticControlPathData) = pre_treatment_rmspe(outcome_gap(paths), paths.treatment_index)

"""
    post_treatment_rmspe(gaps, treatment_index)
    post_treatment_rmspe(paths::SyntheticControlPathData)

Return the root mean squared prediction error over observations from
`treatment_index` through the end of the gap path.

Throws `ArgumentError` if no post-treatment observations exist or gaps are
missing or non-finite.

# Examples

```julia
SyntheticControl.post_treatment_rmspe([0.0, 2.0, 3.0], 3) ≈ 3.0
```
"""
post_treatment_rmspe(gaps::AbstractVector{T}, treatment_index::Int) where {T<:AbstractFloat} = _rmspe(_segment(gaps, treatment_index, :post))
post_treatment_rmspe(paths::SyntheticControlPathData) = post_treatment_rmspe(outcome_gap(paths), paths.treatment_index)

"""
    rmspe_ratio(pre_rmspe, post_rmspe)
    rmspe_ratio(gaps, treatment_index)
    rmspe_ratio(paths::SyntheticControlPathData)

Return `post_rmspe / pre_rmspe`. If both RMSPE values are zero the ratio is
defined as `1`. If pre-treatment RMSPE is zero and post-treatment RMSPE is
positive, the ratio is `Inf`; such non-finite ratios are excluded from
placebo p-values by [`filter_placebos`](@ref).

# Examples

```julia
SyntheticControl.rmspe_ratio(2.0, 6.0) == 3.0
```
"""
function rmspe_ratio(pre_rmspe::T, post_rmspe::T) where {T<:AbstractFloat}
  pre_rmspe >= zero(T) || throw(ArgumentError("pre-treatment RMSPE must be non-negative"))
  post_rmspe >= zero(T) || throw(ArgumentError("post-treatment RMSPE must be non-negative"))
  isfinite(pre_rmspe) || throw(ArgumentError("pre-treatment RMSPE must be finite"))
  isfinite(post_rmspe) || throw(ArgumentError("post-treatment RMSPE must be finite"))
  if pre_rmspe == zero(T)
    return post_rmspe == zero(T) ? one(T) : T(Inf)
  end
  return post_rmspe / pre_rmspe
end

function rmspe_ratio(gaps::AbstractVector{T}, treatment_index::Int) where {T<:AbstractFloat}
  return rmspe_ratio(pre_treatment_rmspe(gaps, treatment_index), post_treatment_rmspe(gaps, treatment_index))
end

rmspe_ratio(paths::SyntheticControlPathData) = rmspe_ratio(outcome_gap(paths), paths.treatment_index)

"""
    placebo_rmspe_ratios(placebo)

Return `(treated_ratio, placebo_ratios)`, where each ratio is
`post_treatment_rmspe / pre_treatment_rmspe` using the placebo result's
treatment index.

# Examples

```julia
placebo = SyntheticControlPlaceboResult(1:3, 3, 3, "t", ["p"],
                                        [1.0, 2.0, 5.0], [1.0, 1.0, 2.0],
                                        reshape([1.0, 2.0, 3.0], 3, 1),
                                        reshape([1.0, 1.0, 1.0], 3, 1))
treated_ratio, placebo_ratios = SyntheticControl.placebo_rmspe_ratios(placebo)
length(placebo_ratios) == 1
```
"""
function placebo_rmspe_ratios(placebo::SyntheticControlPlaceboResult{T}) where {T}
  treated_ratio = rmspe_ratio(placebo.treated_actual .- placebo.treated_synthetic, placebo.treatment_index)
  gaps = placebo_gaps(placebo)
  ratios = Vector{T}(undef, size(gaps, 2))
  for donor in axes(gaps, 2)
    ratios[donor] = rmspe_ratio(gaps[:, donor], placebo.treatment_index)
  end
  return treated_ratio, ratios
end

"""
    filter_placebos(placebo; pre_rmspe_threshold=nothing)

Return a Boolean vector identifying placebo units included in placebo plots
and p-value calculations. Units are excluded when their RMSPE ratio is
non-finite or, if `pre_rmspe_threshold` is supplied, when pre-treatment RMSPE
is greater than that threshold.

`pre_rmspe_threshold` is an absolute RMSPE cutoff and must be non-negative.

# Examples

```julia
placebo = SyntheticControlPlaceboResult(1:3, 3, 3, "t", ["good", "bad"],
  [1.0, 2.0, 5.0], [1.0, 1.0, 2.0],
  [1.0 10.0; 2.0 20.0; 3.0 30.0],
  [1.0 0.0; 1.0 0.0; 1.0 0.0])
SyntheticControl.filter_placebos(placebo; pre_rmspe_threshold=5.0) == [true, false]
```
"""
function filter_placebos(placebo::SyntheticControlPlaceboResult; pre_rmspe_threshold=nothing)
  if pre_rmspe_threshold !== nothing
    pre_rmspe_threshold >= 0 || throw(ArgumentError("pre_rmspe_threshold must be non-negative"))
    isfinite(pre_rmspe_threshold) || throw(ArgumentError("pre_rmspe_threshold must be finite"))
  end
  gaps = placebo_gaps(placebo)
  _treated_ratio, ratios = placebo_rmspe_ratios(placebo)
  included = trues(length(placebo.placebo_ids))
  for donor in axes(gaps, 2)
    pre = pre_treatment_rmspe(gaps[:, donor], placebo.treatment_index)
    included[donor] = isfinite(ratios[donor]) && (pre_rmspe_threshold === nothing || pre <= pre_rmspe_threshold)
  end
  return included
end

"""
    randomization_p_value(placebo; pre_rmspe_threshold=nothing)

Return the finite-sample randomization p-value comparing the treated unit's
RMSPE ratio with included placebo ratios.

The denominator includes the treated unit and all included placebo units. Ties
are counted as extreme using `>=`. Placebo units with non-finite ratios are
excluded, and units with pre-treatment RMSPE above `pre_rmspe_threshold` are
excluded when a threshold is supplied. The treated ratio must be finite.

# Examples

```julia
placebo = SyntheticControlPlaceboResult(1:3, 3, 3, "t", ["p1", "p2"],
  [1.0, 2.0, 5.0], [1.0, 1.0, 2.0],
  [1.0 1.0; 2.0 2.0; 4.0 3.0],
  [1.0 1.0; 1.0 1.5; 1.0 2.0])
0 < SyntheticControl.randomization_p_value(placebo) <= 1
```
"""
function randomization_p_value(placebo::SyntheticControlPlaceboResult; pre_rmspe_threshold=nothing)
  treated_ratio, ratios = placebo_rmspe_ratios(placebo)
  isfinite(treated_ratio) || throw(ArgumentError("treated RMSPE ratio must be finite"))
  included = filter_placebos(placebo; pre_rmspe_threshold=pre_rmspe_threshold)
  any(included) || throw(ArgumentError("no placebo units remain after filtering"))
  numerator = one(Int)
  denominator = one(Int)
  for idx in eachindex(ratios, included)
    if included[idx]
      denominator += 1
      ratios[idx] >= treated_ratio && (numerator += 1)
    end
  end
  return numerator / denominator
end

function _makie_extension_error(fname::Symbol)
  throw(ArgumentError("Makie plotting extension is not loaded; load GLMakie, CairoMakie, or Makie before calling $fname"))
end

"""
    pathplot(result_or_paths; kwargs...)
    pathplot!(axis_or_scene, result_or_paths; kwargs...)

Makie recipe for observed and synthetic outcome paths. Load `GLMakie`,
`CairoMakie`, or `Makie` to activate this plotting method.

Recipe attributes style plotted elements: `actual_color`,
`actual_linewidth`, `actual_linestyle`, `actual_label`,
`synthetic_color`, `synthetic_linewidth`, `synthetic_linestyle`,
`synthetic_label`, `treatment_color`, `treatment_linewidth`,
`treatment_linestyle`, `treatment_label`, and `show_treatment`. Use
`nothing` for a label to exclude that element from legends. Configure axis
titles, labels, ticks, limits, and scales with `Axis(...)` or the
non-mutating `axis=(; ...)` keyword. Configure legends with `axislegend` or
`Legend`.

# Examples

```julia
using SyntheticControl, CairoMakie

paths = SyntheticControlPathData(1:3, 3, 3, [1.0, 2.0, 4.0], [1.0, 1.5, 2.0], "treated")
fig = Figure()
ax = Axis(fig[1, 1]; title="Paths")
pathplot!(ax, paths; actual_color=:black, synthetic_linestyle=:dash)
axislegend(ax)
fig
```
"""
pathplot(args...; kwargs...) = _makie_extension_error(:pathplot)

"""
    pathplot!(axis_or_scene, result_or_paths; kwargs...)

Mutating Makie recipe for observed and synthetic outcome paths. Load
`GLMakie`, `CairoMakie`, or `Makie` to activate this plotting method. See
[`pathplot`](@ref) for recipe attributes. This method adds plot primitives to
the supplied axis or scene and does not modify axis titles, ticks, limits, or
scales.
"""
pathplot!(args...; kwargs...) = _makie_extension_error(:pathplot!)

"""
    gapplot(result_or_paths; kwargs...)
    gapplot!(axis_or_scene, result_or_paths; kwargs...)

Makie recipe for actual-minus-synthetic treatment-effect gaps. Load
`GLMakie`, `CairoMakie`, or `Makie` to activate this plotting method.

Recipe attributes are `gap_color`, `gap_linewidth`, `gap_linestyle`,
`gap_label`, `zero_color`, `zero_linewidth`, `zero_linestyle`,
`zero_label`, `show_zero`, `treatment_color`, `treatment_linewidth`,
`treatment_linestyle`, `treatment_label`, and `show_treatment`. Configure
axes and legends through Makie's standard `Axis`, `axis=(; ...)`,
`axislegend`, and `Legend` APIs.

# Examples

```julia
using SyntheticControl, CairoMakie

paths = SyntheticControlPathData(1:3, 3, 3, [1.0, 2.0, 4.0], [1.0, 1.5, 2.0], "treated")
gapplot(paths; gap_color=:firebrick, axis=(; title="Treatment effect"))
```
"""
gapplot(args...; kwargs...) = _makie_extension_error(:gapplot)

"""
    gapplot!(axis_or_scene, result_or_paths; kwargs...)

Mutating Makie recipe for actual-minus-synthetic treatment-effect gaps. Load
`GLMakie`, `CairoMakie`, or `Makie` to activate this plotting method. See
[`gapplot`](@ref) for recipe attributes. This method preserves user-provided
axis settings.
"""
gapplot!(args...; kwargs...) = _makie_extension_error(:gapplot!)

"""
    placeboplot(placebo; kwargs...)
    placeboplot!(axis_or_scene, placebo; kwargs...)

Makie recipe for treated and placebo gap paths. Load `GLMakie`, `CairoMakie`,
or `Makie` to activate this plotting method.

Recipe attributes are `treated_color`, `treated_linewidth`,
`treated_linestyle`, `treated_label`, `placebo_color`, `placebo_alpha`,
`placebo_linewidth`, `placebo_linestyle`, `placebo_label`, `show_placebos`,
`show_excluded`, `excluded_color`, `excluded_alpha`, `excluded_linewidth`,
`excluded_linestyle`, `excluded_label`, `pre_rmspe_threshold`, `zero_color`,
`zero_linewidth`, `zero_linestyle`, `zero_label`, `show_zero`,
`treatment_color`, `treatment_linewidth`, `treatment_linestyle`,
`treatment_label`, and `show_treatment`. Opacity attributes must be between
0 and 1. Configure axes and legends with Makie.

# Examples

```julia
using SyntheticControl, CairoMakie

placebo = SyntheticControlPlaceboResult(
  1:3, 3, 3, "treated", ["p1"],
  [1.0, 2.0, 5.0], [1.0, 1.0, 2.0],
  reshape([1.0, 2.0, 4.0], 3, 1),
  reshape([1.0, 1.0, 1.0], 3, 1),
)
placeboplot(placebo; treated_color=:firebrick)
```
"""
placeboplot(args...; kwargs...) = _makie_extension_error(:placeboplot)

"""
    placeboplot!(axis_or_scene, placebo; kwargs...)

Mutating Makie recipe for treated and placebo gap paths. Load `GLMakie`,
`CairoMakie`, or `Makie` to activate this plotting method. See
[`placeboplot`](@ref) for recipe attributes. This method preserves
user-provided axis settings.
"""
placeboplot!(args...; kwargs...) = _makie_extension_error(:placeboplot!)

"""
    placebodistribution(placebo; kwargs...)
    placebodistribution!(axis_or_scene, placebo; kwargs...)

Makie recipe for placebo RMSPE-ratio distributions and randomization
p-values. Load `GLMakie`, `CairoMakie`, or `Makie` to activate this plotting
method.

Recipe attributes are `pre_rmspe_threshold`, `placebo_color`,
`placebo_alpha`, `placebo_markersize`, `placebo_marker`, `placebo_label`,
`treated_color`, `treated_alpha`, `treated_markersize`, `treated_marker`,
`treated_label`, `treated_line_color`, `treated_linewidth`,
`treated_linestyle`, `treated_line_label`, `show_treated_line`,
`show_p_value`, `p_value_label`, `p_value_color`, `p_value_fontsize`, and
`p_value_align`. Opacity attributes must be between 0 and 1. Configure axes
and legends with Makie.

# Examples

```julia
using SyntheticControl, CairoMakie

placebo = SyntheticControlPlaceboResult(
  1:3, 3, 3, "treated", ["p1"],
  [1.0, 2.0, 5.0], [1.0, 1.0, 2.0],
  reshape([1.0, 2.0, 4.0], 3, 1),
  reshape([1.0, 1.0, 1.0], 3, 1),
)
placebodistribution(placebo; treated_color=:firebrick)
```
"""
placebodistribution(args...; kwargs...) = _makie_extension_error(:placebodistribution)

"""
    placebodistribution!(axis_or_scene, placebo; kwargs...)

Mutating Makie recipe for placebo RMSPE-ratio distributions and randomization
p-values. Load `GLMakie`, `CairoMakie`, or `Makie` to activate this plotting
method. See [`placebodistribution`](@ref) for recipe attributes. This method
preserves user-provided axis settings.
"""
placebodistribution!(args...; kwargs...) = _makie_extension_error(:placebodistribution!)
