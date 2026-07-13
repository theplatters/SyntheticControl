# Robustness and Placebo Inference

Robustness checks reuse the estimator and configuration of an original fit.
Classic-SCM refits preserve exact predictor-weight starts and stopping
configuration; penalized refits preserve `lambda`. Refitting, stored
statistics, Tables.jl summaries, and Makie rendering remain separate.

The APIs require full balanced-panel paths. Problems returned by
[`from_table`](@ref) retain those paths with original unit and time types. The
actual treatment time is the first post-treatment observation.

## Shared Example Panel

```jldoctest robustness
julia> using SyntheticControl, CommonSolve, Tables, CairoMakie

julia> panel = (
           unit=repeat([:treated, :a, :b, :c], inner=5),
           time=repeat([2001, 2003, 2006, 2010, 2015], 4),
           outcome=[10.,13,15,20,24, 8,9,11,12,13, 13,14,16,17,18, 17,19,21,22,23],
           x=[2.,2.2,2.4,2.6,2.8, 1,1.1,1.2,1.3,1.4,
              3,3.1,3.2,3.3,3.4, 5,5.1,5.2,5.3,5.4],
       );

julia> problem = from_table(panel; unit=:unit, time=:time, outcome=:outcome,
                            predictors=[:x], treated=:treated, treatment_time=2010);

julia> original = solve(problem);
```

## In-Space Placebos and Exact Inference

[`in_space_placebos`](@ref) makes each selected donor pseudo-treated and
completely refits the model. A placebo never appears in its own donor pool.
The originally treated unit is excluded by default to avoid post-treatment
contamination and enters only with `include_treated=true`.

```jldoctest robustness
julia> space = in_space_placebos(problem, original; rmspe_cutoff=5.0);

julia> robustness_counts(space).successful
3

julia> Tables.columnnames(placebo_summary(space))
(:unit, :is_treated, :pre_rmspe, :post_rmspe, :rmspe_ratio, :included, :exclusion_reason, :solver_status)

julia> space_plot = placeboplot(space; show_treatment=true);

julia> ratio_plot = placebodistribution(space);
```

For gaps ``g_{jt}=Y_{jt}-\widehat Y^N_{jt}``, pre-RMSPE uses times before
treatment and post-RMSPE starts at treatment:

```math
\operatorname{RMSPE}_{\mathrm{pre},j}
= \sqrt{\frac{1}{T_0}\sum_{t < T}g_{jt}^2},\qquad
\operatorname{RMSPE}_{\mathrm{post},j}
= \sqrt{\frac{1}{T_1}\sum_{t \ge T}g_{jt}^2}.
```

The ratio is post-RMSPE divided by pre-RMSPE. A relative cutoff `c` includes
a successful placebo when pre-RMSPE is at most `c` times treated pre-RMSPE.
Filtering changes inference eligibility but never deletes stored refits.

The exact p-value is the one-sided upper-tail rank

```math
p = \frac{1 + \#\{j:R_j\ge R_{\mathrm{treated}}\}}
         {1 + \#\{\text{eligible placebo assignments}\}}.
```

The added one is the treated assignment itself, not a continuity correction.
Ties count as extreme. Failed, filtered, `NaN`, and `Inf` placebo ratios are
excluded. Missing or non-finite panel inputs are rejected. Zero pre-RMSPE
gives ratio `1` when post-RMSPE is also zero and `Inf` otherwise. A non-finite
treated ratio or empty placebo set leaves `p_value === nothing`, and
[`randomization_p_value`](@ref) then throws. These assignment-set p-values
are not conventional parametric p-values.

## Leave-One-Out Donor Sensitivity

[`leave_one_out`](@ref) removes exactly one selected donor and re-estimates
all weights. By default every donor is attempted; `active_only=true` retains
original weights greater than `weight_tol`.

```jldoctest robustness
julia> loo = leave_one_out(problem, original);

julia> length(loo.refits)
3

julia> Tables.columnnames(leave_one_out_summary(loo))
(:omitted_donor, :original_weight, :pre_rmspe, :post_rmspe, :rmspe_ratio, :mean_post_gap, :cumulative_post_gap, :max_path_deviation, :solver_status)

julia> loo_plot = leaveoneoutplot(loo);
```

Cumulative post gap is a sum over observed periods and assumes no equal
spacing. Maximum path deviation compares the refitted and original synthetic
paths. Leave-one-out analysis measures donor-composition sensitivity and is
not a formal significance test.

## Generic In-Time Placebos

[`in_time_placebos`](@ref) accepts observed ordered pseudo-dates without
assuming integer steps, equal spacing, or calendar arithmetic. Times before a
pseudo-date fit the model; the pseudo-date begins evaluation. By default,
evaluation ends before the real intervention.

```jldoctest robustness
julia> timing = in_time_placebos(problem, original; placebo_times=[2006]);

julia> only(timing.refits).time
3-element Vector{Int64}:
 2001
 2003
 2006

julia> Tables.columnnames(in_time_summary(timing))
(:placebo_time, :pre_rmspe, :post_rmspe, :rmspe_ratio, :mean_post_gap, :cumulative_post_gap, :pre_periods, :post_periods, :solver_status)

julia> timing_plot = intimeplaceboplot(timing);
```

`post_window=n` takes the first `n` available observed periods. An explicit
collection selects observed times, and a callable may return times or a
Boolean mask. `stop_before_treatment=false` explicitly allows evaluation
through real treatment. Minimum pre/post counts validate each window.
In-time placebos can reveal instability or spurious effects but do not by
themselves establish identification.

## Failures, Parallelism, and Reproducibility

Batch methods accept `on_failure=:record` (default) or `:error`. Recorded
failures retain assignment, effective periods, donor pool, exception message,
and `solver_status=:failed`; numeric summaries are `NaN`.
[`robustness_counts`](@ref) reports successful, failed, filtered, and eligible
counts.

`parallel=true` distributes independent refits over threads and restores
selection order. Each refit owns its caches. Current estimators are
deterministic and expose no stochastic solver or RNG setting, so serial and
parallel results are scheduling-independent.
