#!/usr/bin/env julia

# Run from the repository root with:
#   julia --project=docs examples/visualization_suite.jl
#
# The docs environment includes CairoMakie for headless rendering. For an
# interactive GLMakie version, replace `using CairoMakie` with `using GLMakie`
# and call `GLMakie.activate!()`.

push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

using CairoMakie
using CommonSolve
using Statistics
using SyntheticControl

CairoMakie.activate!()

if isnothing(Base.get_extension(SyntheticControl, :SyntheticControlMakieExt))
  include(normpath(joinpath(@__DIR__, "..", "ext", "SyntheticControlMakieExt.jl")))
end

output_dir = normpath(joinpath(@__DIR__, "plots"))
mkpath(output_dir)

years = collect(2010:2021)
treatment_index = 9
treatment_year = years[treatment_index]
pre_years = years[1:(treatment_index - 1)]
post_years = years[treatment_index:end]
n_pre = length(pre_years)
n_post = length(post_years)
n_donors = 6

Y0_all = Matrix{Float64}(undef, length(years), n_donors)
for (time_index, year) in enumerate(years)
  centered_time = time_index - 1
  common_cycle = 1.8 * sin(0.8 * centered_time) + 0.6 * cos(0.35 * centered_time)
  for donor in 1:n_donors
    level = 36.0 + 2.4 * donor
    trend = 1.05 + 0.14 * donor
    curvature = 0.025 * (donor - 3.5) * centered_time^2
    donor_cycle = (0.45 + 0.08 * donor) * sin(0.55 * centered_time + 0.35 * donor)
    Y0_all[time_index, donor] = level + trend * centered_time + curvature + common_cycle + donor_cycle
  end
end

true_weights = [0.30, 0.24, 0.18, 0.14, 0.09, 0.05]
baseline_treated = Y0_all * true_weights
pre_deviation = [0.35, -0.22, 0.18, -0.12, 0.28, -0.18, 0.12, -0.08]
treatment_effect = [1.2, 2.5, 4.2, 6.0]
Y1_pre = baseline_treated[1:n_pre] .+ pre_deviation
Y1_post = baseline_treated[treatment_index:end] .+ treatment_effect

Y0_pre = Y0_all[1:n_pre, :]
Y0_post = Y0_all[treatment_index:end, :]
X0 = [
  vec(mean(Y0_pre[1:3, :], dims=1))';
  vec(mean(Y0_pre[4:6, :], dims=1))';
  vec(mean(Y0_pre[7:8, :], dims=1))';
]
X1 = [
  mean(Y1_pre[1:3]),
  mean(Y1_pre[4:6]),
  mean(Y1_pre[7:8]),
]

problem = SyntheticControlProblem(
  X1,
  Y1_pre,
  X0,
  Y0_pre,
  ["early pre mean", "middle pre mean", "late pre mean"],
  ["D$i" for i in 1:n_donors],
  "treated",
)

result = solve(problem)

paths = SyntheticControlPathData(
  result;
  time=years,
  Y1_post=Y1_post,
  Y0_post=Y0_post,
)

treated_gap = paths.actual .- paths.synthetic
placebo_gaps = hcat(
  [0.18, -0.12, 0.15, -0.10, 0.20, -0.16, 0.11, -0.08, 0.7, 1.0, 1.3, 1.5],
  [-0.30, 0.22, -0.18, 0.12, -0.26, 0.20, -0.12, 0.10, -0.3, -0.1, 0.2, 0.4],
  [0.42, -0.35, 0.25, -0.20, 0.36, -0.30, 0.24, -0.18, 2.6, 3.0, 3.4, 3.8],
  [-0.15, 0.10, -0.08, 0.12, -0.10, 0.08, -0.06, 0.05, -1.1, -1.4, -1.8, -2.0],
  [1.2, -1.0, 1.4, -1.2, 1.6, -1.4, 1.8, -1.6, 2.0, 2.3, 2.6, 2.9],
  [0.25, -0.18, 0.16, -0.14, 0.20, -0.16, 0.12, -0.10, 5.2, 5.9, 6.5, 7.0],
)
placebo_synthetic = Matrix{Float64}(undef, length(years), n_donors)
for donor in 1:n_donors
  placebo_synthetic[:, donor] = Y0_all[:, donor] .- 0.15 .* treated_gap
end
placebo_actual = placebo_synthetic .+ placebo_gaps

placebo = SyntheticControlPlaceboResult(
  years,
  treatment_index,
  treatment_year,
  "treated",
  ["small positive", "null", "large positive", "negative", "poor fit", "extreme"],
  paths.actual,
  paths.synthetic,
  placebo_actual,
  placebo_synthetic,
)

path_figure = pathplot(
  paths;
  actual_color=:black,
  synthetic_color=:dodgerblue3,
  title="Observed vs synthetic path",
)
save(joinpath(output_dir, "pathplot.png"), path_figure)

gap_figure = gapplot(
  paths;
  gap_color=:firebrick,
  title="Treatment-effect gap",
)
save(joinpath(output_dir, "gapplot.png"), gap_figure)

placebo_figure = placeboplot(
  placebo;
  pre_rmspe_threshold=1.0,
  treated_color=:firebrick,
  placebo_color=:gray45,
  title="Treated and placebo gaps",
)
save(joinpath(output_dir, "placeboplot.png"), placebo_figure)

distribution_figure = placebodistribution(
  placebo;
  pre_rmspe_threshold=1.0,
  treated_color=:firebrick,
  title="Placebo RMSPE ratios",
)
save(joinpath(output_dir, "placebodistribution.png"), distribution_figure)

panel = Figure(size=(1100, 800))
pathplot!(Axis(panel[1, 1]), paths; title="Paths")
gapplot!(Axis(panel[2, 1]), paths; gap_color=:firebrick, title="Gap")
placeboplot!(
  Axis(panel[1, 2]),
  placebo;
  pre_rmspe_threshold=1.0,
  title="Placebo gaps",
)
placebodistribution!(
  Axis(panel[2, 2]),
  placebo;
  pre_rmspe_threshold=1.0,
  treated_color=:firebrick,
  title="Placebo ratios",
)
save(joinpath(output_dir, "all_visualizations.png"), panel)

println("Saved visualization examples to $output_dir")
for filename in (
  "pathplot.png",
  "gapplot.png",
  "placeboplot.png",
  "placebodistribution.png",
  "all_visualizations.png",
)
  println("  ", joinpath(output_dir, filename))
end
