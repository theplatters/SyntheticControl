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

function _legend_position(axis, position)
  position === nothing && return nothing
  return axis isa Makie.Axis ? Makie.axislegend(axis; position=position) : nothing
end

function _deduplicated_legend!(axis, labels, plots; position=:best)
  axis isa Makie.Axis || return nothing
  position === nothing && return nothing
  seen = Set{String}()
  unique_labels = String[]
  unique_plots = Any[]
  for (label, plot) in zip(labels, plots)
    label in seen && continue
    push!(seen, label)
    push!(unique_labels, label)
    push!(unique_plots, plot)
  end
  return Makie.axislegend(axis, unique_plots, unique_labels; position=position)
end

function _axis(plot)
  axis = Makie.current_axis()
  return axis isa Makie.Axis ? axis : Makie.parent(Makie.parent_scene(plot))
end

function _set_axis_labels!(plot, xlabel, ylabel, title)
  axis = _axis(plot)
  if axis isa Makie.Axis
    axis.xlabel = xlabel
    axis.ylabel = ylabel
    axis.title = title
  end
  return nothing
end

"""
    pathplot(result_or_paths; kwargs...)
    pathplot!(axis_or_scene, result_or_paths; kwargs...)

Makie full recipe for observed treated-unit outcomes and synthetic-control
counterfactual paths. `result_or_paths` may be an existing
`SyntheticControlResult`, `PenalizedSyntheticControlResult`, or prepared
`SyntheticControlPathData`.

Attributes include `actual_color`, `synthetic_color`, line styles,
`linewidth`, treatment-line styling, `show_treatment`, `xlabel`, `ylabel`,
`title`, and `legend_position`.
"""
Makie.@recipe PathPlot (result,) begin
  "Color of the observed treated-unit path."
  actual_color = @inherit color :black
  "Color of the synthetic-control path."
  synthetic_color = :dodgerblue3
  "Line style of the observed treated-unit path."
  actual_linestyle = nothing
  "Line style of the synthetic-control path."
  synthetic_linestyle = :dash
  "Line width for actual and synthetic paths."
  linewidth = @inherit linewidth 2
  "Color of the treatment-time marker."
  treatment_color = :gray35
  "Line style of the treatment-time marker."
  treatment_linestyle = :dash
  "Whether to draw the treatment-time marker."
  show_treatment = true
  "X-axis label."
  xlabel = "Time"
  "Y-axis label."
  ylabel = "Outcome"
  "Axis title."
  title = "Observed and synthetic outcomes"
  "Legend position, or `nothing` to skip creating a legend."
  legend_position = :rt
end

function Makie.plot!(plot::PathPlot)
  paths = _paths(plot.result[])
  _set_axis_labels!(plot, plot.xlabel[], plot.ylabel[], plot.title[])
  actual_plot = Makie.lines!(
    plot,
    paths.time,
    paths.actual;
    color=plot.actual_color,
    linestyle=plot.actual_linestyle,
    linewidth=plot.linewidth,
    label="Observed $(paths.treated_id)",
  )
  synthetic_plot = Makie.lines!(
    plot,
    paths.time,
    paths.synthetic;
    color=plot.synthetic_color,
    linestyle=plot.synthetic_linestyle,
    linewidth=plot.linewidth,
    label="Synthetic $(paths.treated_id)",
  )
  if plot.show_treatment[]
    treatment_plot = Makie.vlines!(
      plot,
      [paths.treatment_time];
      color=plot.treatment_color,
      linestyle=plot.treatment_linestyle,
      linewidth=plot.linewidth,
      label="Treatment",
    )
    _deduplicated_legend!(_axis(plot), ["Observed", "Synthetic", "Treatment"], [actual_plot, synthetic_plot, treatment_plot]; position=plot.legend_position[])
  else
    _deduplicated_legend!(_axis(plot), ["Observed", "Synthetic"], [actual_plot, synthetic_plot]; position=plot.legend_position[])
  end
  return plot
end

"""
    gapplot(result_or_paths; kwargs...)
    gapplot!(axis_or_scene, result_or_paths; kwargs...)

Makie full recipe for the treatment-effect gap
`actual_outcome .- synthetic_outcome`. Draws a horizontal zero line and a
vertical treatment-time marker.
"""
Makie.@recipe GapPlot (result,) begin
  "Color of the gap path."
  gap_color = @inherit color :black
  "Line style of the gap path."
  gap_linestyle = nothing
  "Line width of the gap path."
  gap_linewidth = @inherit linewidth 2
  "Color of the horizontal zero line."
  zero_color = :gray50
  "Line style of the horizontal zero line."
  zero_linestyle = :dot
  "Line width of the horizontal zero line."
  zero_linewidth = 1
  "Color of the treatment-time marker."
  treatment_color = :gray35
  "Line style of the treatment-time marker."
  treatment_linestyle = :dash
  "Line width of the treatment-time marker."
  treatment_linewidth = 1
  "Whether to draw the treatment-time marker."
  show_treatment = true
  "X-axis label."
  xlabel = "Time"
  "Y-axis label."
  ylabel = "Actual - synthetic"
  "Axis title."
  title = "Synthetic-control gap"
  "Legend position, or `nothing` to skip creating a legend."
  legend_position = :rt
end

function Makie.plot!(plot::GapPlot)
  paths = _paths(plot.result[])
  gap = SyntheticControl.outcome_gap(paths)
  _set_axis_labels!(plot, plot.xlabel[], plot.ylabel[], plot.title[])
  zero_plot = Makie.hlines!(
    plot,
    [zero(eltype(gap))];
    color=plot.zero_color,
    linestyle=plot.zero_linestyle,
    linewidth=plot.zero_linewidth,
  )
  gap_plot = Makie.lines!(
    plot,
    paths.time,
    gap;
    color=plot.gap_color,
    linestyle=plot.gap_linestyle,
    linewidth=plot.gap_linewidth,
  )
  if plot.show_treatment[]
    treatment_plot = Makie.vlines!(
      plot,
      [paths.treatment_time];
      color=plot.treatment_color,
      linestyle=plot.treatment_linestyle,
      linewidth=plot.treatment_linewidth,
    )
    _deduplicated_legend!(_axis(plot), ["Gap", "Zero", "Treatment"], [gap_plot, zero_plot, treatment_plot]; position=plot.legend_position[])
  else
    _deduplicated_legend!(_axis(plot), ["Gap", "Zero"], [gap_plot, zero_plot]; position=plot.legend_position[])
  end
  return plot
end

"""
    placeboplot(placebo; kwargs...)
    placeboplot!(axis_or_scene, placebo; kwargs...)

Makie full recipe for treated and placebo gap paths. Placebo filtering is
computed by `filter_placebos`; rendering only consumes the resulting inclusion
mask.
"""
Makie.@recipe PlaceboPlot (placebo,) begin
  "Color of the treated-unit gap."
  treated_color = @inherit color :black
  "Line width of the treated-unit gap."
  treated_linewidth = 3
  "Color of included placebo gaps."
  placebo_color = :gray45
  "Transparency applied to included placebo gaps."
  placebo_alpha = 0.35
  "Line width of placebo gaps."
  placebo_linewidth = 1
  "Whether to show excluded placebo gaps."
  show_excluded = false
  "Color of excluded placebo gaps when shown."
  excluded_color = :gray80
  "Transparency applied to excluded placebo gaps."
  excluded_alpha = 0.18
  "Absolute pre-treatment RMSPE cutoff for placebo inclusion, or `nothing`."
  pre_rmspe_threshold = nothing
  "Color of the horizontal zero line."
  zero_color = :gray50
  "Line style of the horizontal zero line."
  zero_linestyle = :dot
  "Color of the treatment-time marker."
  treatment_color = :gray35
  "Line style of the treatment-time marker."
  treatment_linestyle = :dash
  "Whether to draw the treatment-time marker."
  show_treatment = true
  "X-axis label."
  xlabel = "Time"
  "Y-axis label."
  ylabel = "Actual - synthetic"
  "Axis title."
  title = "Treated and placebo gaps"
  "Legend position, or `nothing` to skip creating a legend."
  legend_position = :rt
end

function Makie.plot!(plot::PlaceboPlot)
  placebo = plot.placebo[]
  included = SyntheticControl.filter_placebos(placebo; pre_rmspe_threshold=plot.pre_rmspe_threshold[])
  gaps = SyntheticControl.placebo_gaps(placebo)
  treated_gap = placebo.treated_actual .- placebo.treated_synthetic
  _set_axis_labels!(plot, plot.xlabel[], plot.ylabel[], plot.title[])
  zero_plot = Makie.hlines!(plot, [zero(eltype(treated_gap))]; color=plot.zero_color, linestyle=plot.zero_linestyle)
  included_plot = nothing
  excluded_plot = nothing
  for donor in axes(gaps, 2)
    if included[donor] || plot.show_excluded[]
      color = included[donor] ? (plot.placebo_color[], plot.placebo_alpha[]) : (plot.excluded_color[], plot.excluded_alpha[])
      placebo_plot = Makie.lines!(
        plot,
        placebo.time,
        gaps[:, donor];
        color=color,
        linewidth=plot.placebo_linewidth,
      )
      if included[donor] && included_plot === nothing
        included_plot = placebo_plot
      elseif !included[donor] && excluded_plot === nothing
        excluded_plot = placebo_plot
      end
    end
  end
  treated_plot = Makie.lines!(
    plot,
    placebo.time,
    treated_gap;
    color=plot.treated_color,
    linewidth=plot.treated_linewidth,
  )
  if plot.show_treatment[]
    treatment_plot = Makie.vlines!(plot, [placebo.treatment_time]; color=plot.treatment_color, linestyle=plot.treatment_linestyle)
    labels = ["Treated", "Zero", "Treatment"]
    plots = Any[treated_plot, zero_plot, treatment_plot]
  else
    labels = ["Treated", "Zero"]
    plots = Any[treated_plot, zero_plot]
  end
  if included_plot !== nothing
    push!(labels, "Included placebos")
    push!(plots, included_plot)
  end
  if excluded_plot !== nothing
    push!(labels, "Excluded placebos")
    push!(plots, excluded_plot)
  end
  _deduplicated_legend!(_axis(plot), labels, plots; position=plot.legend_position[])
  return plot
end

"""
    placebodistribution(placebo; kwargs...)
    placebodistribution!(axis_or_scene, placebo; kwargs...)

Makie full recipe for post/pre-treatment RMSPE ratios. The default is a
ranked dot plot because it shows the finite set of randomization units and
the treated unit's rank without binning.
"""
Makie.@recipe PlaceboDistribution (placebo,) begin
  "Absolute pre-treatment RMSPE cutoff for placebo inclusion, or `nothing`."
  pre_rmspe_threshold = nothing
  "Color of placebo ratio markers."
  placebo_color = :gray50
  "Color of the treated-unit ratio marker and line."
  treated_color = :firebrick
  "Marker size for placebo and treated ratio points."
  markersize = @inherit markersize 10
  "Line width of the treated-ratio reference line."
  treated_linewidth = 2
  "Whether to annotate the randomization p-value."
  show_p_value = true
  "Text prefix for p-value annotation."
  p_value_label = "p = "
  "X-axis label."
  xlabel = "Rank"
  "Y-axis label."
  ylabel = "Post/pre RMSPE ratio"
  "Axis title."
  title = "Placebo RMSPE ratios"
  "Legend position, or `nothing` to skip creating a legend."
  legend_position = :rt
end

function Makie.plot!(plot::PlaceboDistribution)
  placebo = plot.placebo[]
  treated_ratio, ratios = SyntheticControl.placebo_rmspe_ratios(placebo)
  included = SyntheticControl.filter_placebos(placebo; pre_rmspe_threshold=plot.pre_rmspe_threshold[])
  included_ratios = sort(ratios[included])
  isempty(included_ratios) && throw(ArgumentError("no placebo units remain after filtering"))
  p_value = SyntheticControl.randomization_p_value(placebo; pre_rmspe_threshold=plot.pre_rmspe_threshold[])
  _set_axis_labels!(plot, plot.xlabel[], plot.ylabel[], plot.title[])
  placebo_scatter = Makie.scatter!(
    plot,
    collect(eachindex(included_ratios)),
    included_ratios;
    color=plot.placebo_color,
    markersize=plot.markersize,
  )
  treated_rank = 1 + count(<(treated_ratio), included_ratios)
  treated_scatter = Makie.scatter!(
    plot,
    [treated_rank],
    [treated_ratio];
    color=plot.treated_color,
    markersize=1.25 * plot.markersize[],
  )
  treated_line = Makie.hlines!(
    plot,
    [treated_ratio];
    color=plot.treated_color,
    linewidth=plot.treated_linewidth,
    linestyle=:dash,
  )
  if plot.show_p_value[]
    Makie.text!(
      plot,
      [1],
      [treated_ratio];
      text=["$(plot.p_value_label[])$(round(p_value; digits=3))"],
      color=plot.treated_color,
      align=(:left, :bottom),
    )
  end
  _deduplicated_legend!(
    _axis(plot),
    ["Included placebo ratios", "Treated ratio", "Treated ratio line"],
    [placebo_scatter, treated_scatter, treated_line];
    position=plot.legend_position[],
  )
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
