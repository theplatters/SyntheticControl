# Visualization

`SyntheticControl.jl` provides Makie recipes for outcome paths, treatment
effect gaps, placebo gaps, and placebo RMSPE-ratio distributions. The recipes
live in a Makie extension: load `GLMakie`, `CairoMakie`, or `Makie` before
calling them.

For interactive local use:

```julia
using GLMakie
using SyntheticControl

GLMakie.activate!()
```

The documentation build uses `CairoMakie` so examples can render in headless
environments. The recipes themselves are backend-independent Makie recipes.

## Complete Example

```jldoctest visualization
julia> using SyntheticControl, CommonSolve, CairoMakie

julia> problem = SyntheticControlProblem(
           [2.0],
           [20.0, 21.0, 22.0],
           reshape([1.0, 4.0, 5.0], 1, 3),
           [10.0 40.0 50.0;
            11.0 41.0 51.0;
            12.0 42.0 52.0],
           ["baseline"],
           ["D1", "D2", "D3"],
           "treated",
       );

julia> result = solve(problem);

julia> paths = SyntheticControlPathData(
           result;
           time=2001:2005,
           Y1_post=[24.0, 26.0],
           Y0_post=[13.0 43.0 53.0;
                    14.0 44.0 54.0],
       );

julia> length(outcome_gap(paths))
5
```

Create a standalone path plot:

```julia
fig_axis_plot = pathplot(paths; actual_color=:black, synthetic_color=:dodgerblue3)
```

Compose plots into a user-defined layout:

```julia
fig = Figure(size=(900, 700))
ax_path = Axis(fig[1, 1])
ax_gap = Axis(fig[2, 1])
pathplot!(ax_path, paths)
gapplot!(ax_gap, paths; gap_color=:firebrick)
fig
```

## Placebo Diagnostics

Placebo plots use prepared placebo paths. The package does not re-estimate
placebos inside the plotting code.

```jldoctest visualization
julia> placebo = SyntheticControlPlaceboResult(
           2001:2005,
           4,
           2004,
           "treated",
           ["placebo 1", "placebo 2", "poor fit"],
           [20.0, 21.0, 22.0, 24.0, 26.0],
           [19.0, 20.0, 21.0, 22.0, 23.0],
           [20.0 30.0 50.0;
            21.0 31.0 49.0;
            22.0 32.0 51.0;
            25.0 38.0 58.0;
            27.0 40.0 60.0],
           [19.0 29.0 10.0;
            20.0 30.0 10.0;
            21.0 31.0 10.0;
            22.0 33.0 10.0;
            23.0 35.0 10.0],
       );

julia> filter_placebos(placebo; pre_rmspe_threshold=5.0)
3-element BitVector:
 1
 1
 0

julia> 0 < randomization_p_value(placebo; pre_rmspe_threshold=5.0) <= 1
true
```

Use `placeboplot` to compare gap paths and `placebodistribution` to compare
RMSPE ratios:

```julia
fig = Figure(size=(1000, 800))
placeboplot!(Axis(fig[1, 1]), placebo; pre_rmspe_threshold=5.0)
placebodistribution!(Axis(fig[1, 2]), placebo; pre_rmspe_threshold=5.0)
fig
```

The distribution recipe defaults to a ranked dot plot instead of a histogram.
The donor pool is finite and often small, so the ranked plot shows every
included randomization unit and the treated unit's rank without binning.

## Interpretation

Path plots compare observed treated outcomes to the synthetic counterfactual.
Large post-treatment divergence is evidence of a treatment effect only when
pre-treatment fit is credible.

Gap plots show `actual - synthetic` directly. Pre-treatment gaps diagnose fit;
post-treatment gaps are the estimated treatment-effect path.

Placebo gap plots show whether the treated unit's gap is unusual relative to
donor units treated as if they had received the intervention. Placebos with
poor pre-treatment fit can be excluded using `pre_rmspe_threshold`, an
absolute pre-treatment RMSPE cutoff.

The placebo distribution uses

```math
R_j = \frac{\operatorname{RMSPE}_{j,\mathrm{post}}}
           {\operatorname{RMSPE}_{j,\mathrm{pre}}}
```

and reports

```math
p = \frac{1 + \#\{j: R_j \ge R_{\mathrm{treated}}\}}
         {1 + \#\{\mathrm{included\ placebo\ units}\}}.
```

The treated unit is included in the denominator and numerator. Ties are
counted as at least as extreme using `>=`. Placebo units with non-finite
ratios are excluded, and units above `pre_rmspe_threshold` are excluded when a
threshold is supplied. If pre-treatment RMSPE is zero, the ratio is `1` when
post-treatment RMSPE is also zero and `Inf` otherwise; non-finite placebo
ratios are excluded. Placebo-based p-values depend on the available donor
pool and are not conventional parametric p-values.

