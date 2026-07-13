# Fit Diagnostics

[`fit_diagnostics`](@ref) calculates typed, backend-independent diagnostics
from either a [`SyntheticControlResult`](@ref) or a
[`PenalizedSyntheticControlResult`](@ref). After loading Tables.jl,
[`diagnostics_table`](@ref) presents the scalar values as one row and
[`predictor_diagnostics_table`](@ref) presents one row per predictor.

```jldoctest diagnostics
julia> using SyntheticControl, Tables

julia> data = SyntheticControlData(
           [2.0, 0.0], [2.0, 4.0, 8.0],
           [1.0 3.0; -1.0 1.0],
           [1.0 3.0; 3.0 5.0; 7.0 9.0],
           ["level", "zero baseline"], ["a", "b"], "treated",
       );

julia> fit = SyntheticControl.SyntheticControlResult(
           data, [0.25, 0.75], [0.5, 0.5], 0.25,
       );

julia> diagnostics = fit_diagnostics(fit);

julia> (diagnostics.pre_rmspe, diagnostics.pre_mae,
        diagnostics.effective_donor_count)
(0.5, 0.5, 1.6)

julia> Tables.columnnames(predictor_diagnostics_table(fit))
(:predictor, :treated, :synthetic, :difference, :absolute_difference, :relative_difference)
```

## Definitions and validation

All rows of `fit.data.Y1` and `fit.data.Y0` belong to the pre-treatment
period. These are the same aligned inputs used by the estimator. Package data
constructors reject missing and non-finite inputs; diagnostics additionally
reject non-finite derived paths and predictor values.

The scalar outcome diagnostics are:

- `pre_rmspe`: square root of the mean squared pre-treatment gap.
- `pre_mae`: mean absolute pre-treatment gap.
- `max_absolute_pre_gap`: largest absolute pre-treatment gap.

Donor weights are never normalized by the diagnostics API. They must be
nonempty, finite, nonnegative, and sum to one within `sqrt(eps(T))`, where `T`
is their element type. Set `weight_sum_tolerance` explicitly to choose a
different finite, nonnegative tolerance. `effective_donor_count` uses the raw
accepted weights as `1 / sum(abs2, W)`; `largest_donor_weight`, `weight_sum`,
and `weight_sum_error` also report those raw weights.

Predictor differences use original, unstandardized predictor units:

```math
d_k = X_{1k} - \sum_j X_{0kj}w_j.
```

Absolute imbalance is `abs(d_k)`. Relative imbalance is
`abs(d_k) / abs(X1[k])`, with `0 / 0` defined as zero and a nonzero difference
relative to a zero treated value defined as `Inf`. The scalar table reports
the mean and maximum absolute and relative predictor imbalance; the
predictor table retains every individual value in predictor order.

## Stable scalar schema

`diagnostics_table(fit)` has exactly one row with columns:

| Group | Columns |
| --- | --- |
| Outcome fit | `pre_rmspe`, `pre_mae`, `max_absolute_pre_gap` |
| Donor concentration | `effective_donor_count`, `largest_donor_weight`, `weight_sum`, `weight_sum_error` |
| Predictor fit | `predictor_mae`, `predictor_max_absolute_imbalance`, `predictor_mean_relative_imbalance`, `predictor_max_relative_imbalance` |
| Solver | `solver`, `solver_success`, `termination_status`, `iteration_count`, `objective_value`, `runtime_seconds`, `objective_evaluations` |

The current built-in solvers retain an objective but do not retain a
convergence flag, termination code, iteration count, runtime, or evaluation
count in the fit object. Those columns therefore contain typed `missing`
values. They are not inferred from finite weights. For classic SCM,
`objective_value` is the stored pre-treatment MSPE; for penalized SCM, it is
the stored penalized objective.
