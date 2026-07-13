module SyntheticControlMakieExt

using Makie
using SyntheticControl

const PathInput = Union{SyntheticControl.SyntheticControlResultLike,SyntheticControl.SyntheticControlPathData}

function _paths(input::SyntheticControl.SyntheticControlPathData)
  return input
end

function _paths(input::SyntheticControl.SyntheticControlResultLike)
  return SyntheticControl.SyntheticControlPathData(input)
end

function _nonnegative(value, name::Symbol)
  value < 0 && throw(ArgumentError("$name must be non-negative"))
  return value
end

function _opacity(value, name::Symbol)
  0 <= value <= 1 || throw(ArgumentError("$name must be between 0 and 1"))
  return value
end

function _validated_nonnegative(attribute, name::Symbol)
  return Makie.lift(value -> _nonnegative(value, name), attribute)
end

function _color_with_alpha(color_attribute, alpha_attribute, alpha_name::Symbol)
  return Makie.lift(color_attribute, alpha_attribute) do color, alpha
    return (color, _opacity(alpha, alpha_name))
  end
end

"""
    pathplot(result_or_paths; kwargs...)
    pathplot!(axis_or_scene, result_or_paths; kwargs...)

Plot observed treated-unit outcomes and synthetic-control counterfactual
paths. `result_or_paths` may be a `SyntheticControlResult`,
`PenalizedSyntheticControlResult`, or prepared `SyntheticControlPathData`.

Recipe attributes style the plotted elements only. Configure axes with
`Axis(...)` or the non-mutating `axis=(; ...)` keyword, and create legends
with `axislegend` or `Legend`.

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
Makie.@recipe PathPlot (result,) begin
  "Color of the observed treated-unit path."
  actual_color = @inherit color :black
  "Line width of the observed treated-unit path. Must be non-negative."
  actual_linewidth = @inherit linewidth 2
  "Line style of the observed treated-unit path."
  actual_linestyle = nothing
  "Legend label for the observed treated-unit path. Use `nothing` to exclude it."
  actual_label = "Observed"
  "Color of the synthetic-control path."
  synthetic_color = :dodgerblue3
  "Line width of the synthetic-control path. Must be non-negative."
  synthetic_linewidth = @inherit linewidth 2
  "Line style of the synthetic-control path."
  synthetic_linestyle = :dash
  "Legend label for the synthetic-control path. Use `nothing` to exclude it."
  synthetic_label = "Synthetic"
  "Color of the treatment-time marker."
  treatment_color = :gray35
  "Line width of the treatment-time marker. Must be non-negative."
  treatment_linewidth = @inherit linewidth 2
  "Line style of the treatment-time marker."
  treatment_linestyle = :dash
  "Legend label for the treatment-time marker. Use `nothing` to exclude it."
  treatment_label = "Treatment"
  "Whether to draw the treatment-time marker."
  show_treatment = true
end

Makie.preferred_axis_type(::PathPlot) = Makie.Axis
Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, ::PathPlot) = (;
  xlabel="Time",
  ylabel="Outcome",
  title="Observed and synthetic outcomes",
)

function Makie.plot!(plot::PathPlot)
  paths = _paths(plot.result[])
  Makie.lines!(
    plot,
    paths.time,
    paths.actual;
    color=plot.actual_color,
    linestyle=plot.actual_linestyle,
    linewidth=_validated_nonnegative(plot.actual_linewidth, :actual_linewidth),
    label=plot.actual_label,
  )
  Makie.lines!(
    plot,
    paths.time,
    paths.synthetic;
    color=plot.synthetic_color,
    linestyle=plot.synthetic_linestyle,
    linewidth=_validated_nonnegative(plot.synthetic_linewidth, :synthetic_linewidth),
    label=plot.synthetic_label,
  )
  if plot.show_treatment[]
    Makie.vlines!(
      plot,
      [paths.treatment_time];
      color=plot.treatment_color,
      linestyle=plot.treatment_linestyle,
      linewidth=_validated_nonnegative(plot.treatment_linewidth, :treatment_linewidth),
      label=plot.treatment_label,
    )
  end
  return plot
end

"""
    gapplot(result_or_paths; kwargs...)
    gapplot!(axis_or_scene, result_or_paths; kwargs...)

Plot the treatment-effect gap `actual_outcome .- synthetic_outcome` with
optional zero and treatment-time reference lines.

Recipe attributes style the gap, zero line, and treatment marker. Configure
axes and legends through Makie's normal `Axis`, `axis=(; ...)`, `axislegend`,
and `Legend` APIs.
"""
Makie.@recipe GapPlot (result,) begin
  "Color of the gap path."
  gap_color = @inherit color :black
  "Line width of the gap path. Must be non-negative."
  gap_linewidth = @inherit linewidth 2
  "Line style of the gap path."
  gap_linestyle = nothing
  "Legend label for the gap path. Use `nothing` to exclude it."
  gap_label = "Gap"
  "Color of the horizontal zero line."
  zero_color = :gray50
  "Line width of the horizontal zero line. Must be non-negative."
  zero_linewidth = 1
  "Line style of the horizontal zero line."
  zero_linestyle = :dot
  "Legend label for the zero line. Use `nothing` to exclude it."
  zero_label = "Zero"
  "Whether to draw the horizontal zero line."
  show_zero = true
  "Color of the treatment-time marker."
  treatment_color = :gray35
  "Line width of the treatment-time marker. Must be non-negative."
  treatment_linewidth = 1
  "Line style of the treatment-time marker."
  treatment_linestyle = :dash
  "Legend label for the treatment-time marker. Use `nothing` to exclude it."
  treatment_label = "Treatment"
  "Whether to draw the treatment-time marker."
  show_treatment = true
end

Makie.preferred_axis_type(::GapPlot) = Makie.Axis
Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, ::GapPlot) = (;
  xlabel="Time",
  ylabel="Actual - synthetic",
  title="Synthetic-control gap",
)

function Makie.plot!(plot::GapPlot)
  paths = _paths(plot.result[])
  gap = SyntheticControl.outcome_gap(paths)
  if plot.show_zero[]
    Makie.hlines!(
      plot,
      [zero(eltype(gap))];
      color=plot.zero_color,
      linestyle=plot.zero_linestyle,
      linewidth=_validated_nonnegative(plot.zero_linewidth, :zero_linewidth),
      label=plot.zero_label,
    )
  end
  Makie.lines!(
    plot,
    paths.time,
    gap;
    color=plot.gap_color,
    linestyle=plot.gap_linestyle,
    linewidth=_validated_nonnegative(plot.gap_linewidth, :gap_linewidth),
    label=plot.gap_label,
  )
  if plot.show_treatment[]
    Makie.vlines!(
      plot,
      [paths.treatment_time];
      color=plot.treatment_color,
      linestyle=plot.treatment_linestyle,
      linewidth=_validated_nonnegative(plot.treatment_linewidth, :treatment_linewidth),
      label=plot.treatment_label,
    )
  end
  return plot
end

"""
    placeboplot(placebo; kwargs...)
    placeboplot!(axis_or_scene, placebo; kwargs...)

Plot treated and placebo gap paths. Placebo filtering is computed by
`filter_placebos`; rendering consumes the resulting inclusion mask.

Use recipe attributes for line appearance, opacity, labels, optional excluded
placebos, zero lines, and treatment markers. Use Makie's standard axis and
legend APIs for layout, ticks, titles, and legend placement.
"""
Makie.@recipe PlaceboPlot (placebo,) begin
  "Color of the treated-unit gap."
  treated_color = @inherit color :black
  "Line width of the treated-unit gap. Must be non-negative."
  treated_linewidth = 3
  "Line style of the treated-unit gap."
  treated_linestyle = nothing
  "Legend label for the treated-unit gap. Use `nothing` to exclude it."
  treated_label = "Treated"
  "Color of included placebo gaps."
  placebo_color = :gray45
  "Opacity of included placebo gaps. Must be between 0 and 1."
  placebo_alpha = 0.35
  "Line width of included placebo gaps. Must be non-negative."
  placebo_linewidth = 1
  "Line style of included placebo gaps."
  placebo_linestyle = nothing
  "Legend label for the first included placebo gap. Use `nothing` to exclude it."
  placebo_label = "Included placebos"
  "Whether to draw included placebo gaps."
  show_placebos = true
  "Whether to show excluded placebo gaps."
  show_excluded = false
  "Color of excluded placebo gaps when shown."
  excluded_color = :gray80
  "Opacity of excluded placebo gaps when shown. Must be between 0 and 1."
  excluded_alpha = 0.18
  "Line width of excluded placebo gaps. Must be non-negative."
  excluded_linewidth = 1
  "Line style of excluded placebo gaps."
  excluded_linestyle = nothing
  "Legend label for the first excluded placebo gap. Use `nothing` to exclude it."
  excluded_label = "Excluded placebos"
  "Absolute pre-treatment RMSPE cutoff for placebo inclusion, or `nothing`."
  pre_rmspe_threshold = nothing
  "Color of the horizontal zero line."
  zero_color = :gray50
  "Line width of the horizontal zero line. Must be non-negative."
  zero_linewidth = 1
  "Line style of the horizontal zero line."
  zero_linestyle = :dot
  "Legend label for the zero line. Use `nothing` to exclude it."
  zero_label = "Zero"
  "Whether to draw the horizontal zero line."
  show_zero = true
  "Color of the treatment-time marker."
  treatment_color = :gray35
  "Line width of the treatment-time marker. Must be non-negative."
  treatment_linewidth = 1
  "Line style of the treatment-time marker."
  treatment_linestyle = :dash
  "Legend label for the treatment-time marker. Use `nothing` to exclude it."
  treatment_label = "Treatment"
  "Whether to draw the treatment-time marker."
  show_treatment = true
end

Makie.preferred_axis_type(::PlaceboPlot) = Makie.Axis
Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, ::PlaceboPlot) = (;
  xlabel="Time",
  ylabel="Actual - synthetic",
  title="Treated and placebo gaps",
)

function Makie.plot!(plot::PlaceboPlot)
  placebo = plot.placebo[]
  included = SyntheticControl.filter_placebos(placebo; pre_rmspe_threshold=plot.pre_rmspe_threshold[])
  gaps = SyntheticControl.placebo_gaps(placebo)
  treated_gap = placebo.treated_actual .- placebo.treated_synthetic

  if plot.show_zero[]
    Makie.hlines!(
      plot,
      [zero(eltype(treated_gap))];
      color=plot.zero_color,
      linestyle=plot.zero_linestyle,
      linewidth=_validated_nonnegative(plot.zero_linewidth, :zero_linewidth),
      label=plot.zero_label,
    )
  end

  included_label_used = false
  excluded_label_used = false
  for donor in axes(gaps, 2)
    if included[donor] && plot.show_placebos[]
      Makie.lines!(
        plot,
        placebo.time,
        gaps[:, donor];
        color=_color_with_alpha(plot.placebo_color, plot.placebo_alpha, :placebo_alpha),
        linewidth=_validated_nonnegative(plot.placebo_linewidth, :placebo_linewidth),
        linestyle=plot.placebo_linestyle,
        label=included_label_used ? nothing : plot.placebo_label,
      )
      included_label_used = true
    elseif !included[donor] && plot.show_excluded[]
      Makie.lines!(
        plot,
        placebo.time,
        gaps[:, donor];
        color=_color_with_alpha(plot.excluded_color, plot.excluded_alpha, :excluded_alpha),
        linewidth=_validated_nonnegative(plot.excluded_linewidth, :excluded_linewidth),
        linestyle=plot.excluded_linestyle,
        label=excluded_label_used ? nothing : plot.excluded_label,
      )
      excluded_label_used = true
    end
  end

  Makie.lines!(
    plot,
    placebo.time,
    treated_gap;
    color=plot.treated_color,
    linewidth=_validated_nonnegative(plot.treated_linewidth, :treated_linewidth),
    linestyle=plot.treated_linestyle,
    label=plot.treated_label,
  )
  if plot.show_treatment[]
    Makie.vlines!(
      plot,
      [placebo.treatment_time];
      color=plot.treatment_color,
      linestyle=plot.treatment_linestyle,
      linewidth=_validated_nonnegative(plot.treatment_linewidth, :treatment_linewidth),
      label=plot.treatment_label,
    )
  end
  return plot
end

"""
    placebodistribution(placebo; kwargs...)
    placebodistribution!(axis_or_scene, placebo; kwargs...)

Plot post/pre-treatment RMSPE ratios as a ranked dot plot. The treated unit's
rank and ratio are highlighted with a marker and optional horizontal reference
line; the randomization p-value can be annotated.

Customize marker, reference-line, text, and label attributes on the recipe.
Configure axis ticks, labels, titles, and legends through Makie's standard
APIs.
"""
Makie.@recipe PlaceboDistribution (placebo,) begin
  "Absolute pre-treatment RMSPE cutoff for placebo inclusion, or `nothing`."
  pre_rmspe_threshold = nothing
  "Color of placebo ratio markers."
  placebo_color = :gray50
  "Opacity of placebo ratio markers. Must be between 0 and 1."
  placebo_alpha = 1.0
  "Marker size for placebo ratio points. Must be non-negative."
  placebo_markersize = @inherit markersize 10
  "Marker shape for placebo ratio points."
  placebo_marker = @inherit marker Circle
  "Legend label for placebo ratio markers. Use `nothing` to exclude them."
  placebo_label = "Included placebo ratios"
  "Color of the treated-unit ratio marker."
  treated_color = :firebrick
  "Opacity of the treated-unit ratio marker. Must be between 0 and 1."
  treated_alpha = 1.0
  "Marker size for the treated-unit ratio point. Must be non-negative."
  treated_markersize = @inherit markersize 12.5
  "Marker shape for the treated-unit ratio point."
  treated_marker = @inherit marker Circle
  "Legend label for the treated-unit ratio marker. Use `nothing` to exclude it."
  treated_label = "Treated ratio"
  "Color of the treated-ratio reference line."
  treated_line_color = :firebrick
  "Line width of the treated-ratio reference line. Must be non-negative."
  treated_linewidth = 2
  "Line style of the treated-ratio reference line."
  treated_linestyle = :dash
  "Legend label for the treated-ratio reference line. Use `nothing` to exclude it."
  treated_line_label = "Treated ratio line"
  "Whether to draw the treated-ratio reference line."
  show_treated_line = true
  "Whether to annotate the randomization p-value."
  show_p_value = true
  "Text prefix for p-value annotation."
  p_value_label = "p = "
  "Color of the p-value annotation."
  p_value_color = :firebrick
  "Font size of the p-value annotation."
  p_value_fontsize = @inherit fontsize 16
  "Text alignment of the p-value annotation."
  p_value_align = (:left, :bottom)
end

Makie.preferred_axis_type(::PlaceboDistribution) = Makie.Axis
Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, ::PlaceboDistribution) = (;
  xlabel="Rank",
  ylabel="Post/pre RMSPE ratio",
  title="Placebo RMSPE ratios",
)
Makie.get_plots(plot::Union{PathPlot,GapPlot,PlaceboPlot,PlaceboDistribution}) = plot.plots

function Makie.plot!(plot::PlaceboDistribution)
  placebo = plot.placebo[]
  treated_ratio, ratios = SyntheticControl.placebo_rmspe_ratios(placebo)
  included = SyntheticControl.filter_placebos(placebo; pre_rmspe_threshold=plot.pre_rmspe_threshold[])
  included_ratios = sort(ratios[included])
  isempty(included_ratios) && throw(ArgumentError("no placebo units remain after filtering"))
  p_value = SyntheticControl.randomization_p_value(placebo; pre_rmspe_threshold=plot.pre_rmspe_threshold[])

  Makie.scatter!(
    plot,
    collect(eachindex(included_ratios)),
    included_ratios;
    color=_color_with_alpha(plot.placebo_color, plot.placebo_alpha, :placebo_alpha),
    markersize=_validated_nonnegative(plot.placebo_markersize, :placebo_markersize),
    marker=plot.placebo_marker,
    label=plot.placebo_label,
  )
  treated_rank = 1 + count(<(treated_ratio), included_ratios)
  Makie.scatter!(
    plot,
    [treated_rank],
    [treated_ratio];
    color=_color_with_alpha(plot.treated_color, plot.treated_alpha, :treated_alpha),
    markersize=_validated_nonnegative(plot.treated_markersize, :treated_markersize),
    marker=plot.treated_marker,
    label=plot.treated_label,
  )
  if plot.show_treated_line[]
    Makie.hlines!(
      plot,
      [treated_ratio];
      color=plot.treated_line_color,
      linewidth=_validated_nonnegative(plot.treated_linewidth, :treated_linewidth),
      linestyle=plot.treated_linestyle,
      label=plot.treated_line_label,
    )
  end
  if plot.show_p_value[]
    Makie.text!(
      plot,
      [1],
      [treated_ratio];
      text=Makie.lift(plot.p_value_label) do label
        return ["$(label)$(round(p_value; digits=3))"]
      end,
      color=plot.p_value_color,
      fontsize=_validated_nonnegative(plot.p_value_fontsize, :p_value_fontsize),
      align=plot.p_value_align,
    )
  end
  return plot
end

function SyntheticControl.pathplot(input::PathInput; kwargs...)
  return pathplot(input; kwargs...)
end

function SyntheticControl.pathplot!(axis_or_scene, input::PathInput; kwargs...)
  return pathplot!(axis_or_scene, input; kwargs...)
end

function SyntheticControl.gapplot(input::PathInput; kwargs...)
  return gapplot(input; kwargs...)
end

function SyntheticControl.gapplot!(axis_or_scene, input::PathInput; kwargs...)
  return gapplot!(axis_or_scene, input; kwargs...)
end

function SyntheticControl.placeboplot(input::SyntheticControl.SyntheticControlPlaceboResult; kwargs...)
  return placeboplot(input; kwargs...)
end

function SyntheticControl.placeboplot!(axis_or_scene, input::SyntheticControl.SyntheticControlPlaceboResult; kwargs...)
  return placeboplot!(axis_or_scene, input; kwargs...)
end

function SyntheticControl.placebodistribution(input::SyntheticControl.SyntheticControlPlaceboResult; kwargs...)
  return placebodistribution(input; kwargs...)
end

function SyntheticControl.placebodistribution!(axis_or_scene, input::SyntheticControl.SyntheticControlPlaceboResult; kwargs...)
  return placebodistribution!(axis_or_scene, input; kwargs...)
end

end
