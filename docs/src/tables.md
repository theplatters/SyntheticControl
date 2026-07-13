# Tables Integration

`SyntheticControl.jl` provides an optional Tables.jl extension for long-format
panel data. Loading the core package alone does not load Tables.jl.

```jldoctest tables_activation
julia> using SyntheticControl

julia> using Tables
```

After both packages are loaded, [`from_table`](@ref) constructs a
[`SyntheticControlProblem`](@ref) from any Tables.jl-compatible source.

## Input Schema

The input must contain one row per unit and time period with columns for:

- unit identifier,
- time identifier,
- outcome,
- one or more numeric predictors.

Column selectors may be `Symbol`s or `String`s. The panel is materialized in
memory once with `Tables.rows`, so single-pass row iterators are supported.
Rows are never dropped silently.

```jldoctest tables_end_to_end
julia> using SyntheticControl, CommonSolve, Tables

julia> panel = (
           unit = repeat(["treated", "donor_a", "donor_b"], inner=5),
           year = repeat(2001:2005, 3),
           outcome = [
               10.0, 11.0, 12.0, 15.0, 17.0,
                9.0, 10.0, 11.0, 12.0, 13.0,
               11.0, 11.5, 12.0, 12.5, 13.0,
           ],
           income = [
               20.0, 21.0, 22.0, 23.0, 24.0,
               18.0, 19.0, 20.0, 21.0, 22.0,
               22.0, 22.0, 23.0, 23.0, 24.0,
           ],
       );

julia> problem = from_table(panel; unit=:unit, time=:year, outcome=:outcome,
                            predictors=[:income], treated="treated",
                            treatment_time=2004);

julia> problem.X1
1-element Vector{Float64}:
 21.0

julia> problem.X0
1×2 Matrix{Float64}:
 19.0  22.3333

julia> result = solve(problem);

julia> sum(result.W) ≈ 1.0
true
```

Time values and donor unit identifiers are sorted explicitly, so results do
not depend on input row order. Times before `treatment_time` are
pre-treatment; the treatment time itself is the first post-treatment period.
Predictor columns are averaged over pre-treatment observations only. The
matrix orientation is:

- `X1`: treated predictor vector with length `K`.
- `X0`: `K × J` donor predictor matrix.
- `Y1`: treated pre-treatment outcome vector with length `T_pre`.
- `Y0`: `T_pre × J` donor pre-treatment outcome matrix.

## Output Tables

The extension returns lightweight Tables.jl-compatible named tuples of
vectors. It does not depend on DataFrames.jl or CSV.jl.

```jldoctest tables_end_to_end
julia> Tables.columnnames(weights_table(result))
(:donor, :weight)

julia> Tables.columnnames(balance_table(problem, result))
(:predictor, :treated, :synthetic, :difference)

julia> Tuple(first(Tables.columnnames(diagnostics_table(result)), 3))
(:pre_rmspe, :pre_mae, :max_absolute_pre_gap)

julia> path = path_table(problem, result);

julia> Tables.columnnames(path)
(:time, :actual, :synthetic, :gap, :post_treatment)

julia> path.post_treatment
5-element Vector{Bool}:
 0
 0
 0
 1
 1
```

Stable schemas are:

| Function | Columns |
| --- | --- |
| `weights_table` | `donor`, `weight` |
| `balance_table` | `predictor`, `treated`, `synthetic`, `difference` |
| `path_table` | `time`, `actual`, `synthetic`, `gap`, `post_treatment` |
| `diagnostics_table` | `pre_rmspe`, `pre_mae`, `max_absolute_pre_gap`, `effective_donor_count`, `largest_donor_weight`, `weight_sum`, `weight_sum_error`, `predictor_mae`, `predictor_max_absolute_imbalance`, `predictor_mean_relative_imbalance`, `predictor_max_relative_imbalance`, `solver`, `solver_success`, `termination_status`, `iteration_count`, `objective_value`, `runtime_seconds`, `objective_evaluations` |
| `predictor_diagnostics_table` | `predictor`, `treated`, `synthetic`, `difference`, `absolute_difference`, `relative_difference` |
| `placebo_summary` | `unit`, `is_treated`, `pre_rmspe`, `post_rmspe`, `rmspe_ratio`, `included`, `exclusion_reason`, `solver_status` |
| `leave_one_out_summary` | `omitted_donor`, `original_weight`, `pre_rmspe`, `post_rmspe`, `rmspe_ratio`, `mean_post_gap`, `cumulative_post_gap`, `max_path_deviation`, `solver_status` |
| `in_time_summary` | `placebo_time`, `pre_rmspe`, `post_rmspe`, `rmspe_ratio`, `mean_post_gap`, `cumulative_post_gap`, `pre_periods`, `post_periods`, `solver_status` |
| `placebo_paths` | `analysis_id`, `time`, `actual`, `synthetic`, `gap`, `is_post_treatment`, `included`, `solver_status`, `failure_reason`, `is_treated`, `exclusion_reason` |
| `leave_one_out_paths` | `analysis_id`, `time`, `actual`, `synthetic`, `gap`, `is_post_treatment`, `included`, `solver_status`, `failure_reason`, `original_weight` |
| `in_time_paths` | `analysis_id`, `time`, `actual`, `synthetic`, `gap`, `is_post_treatment`, `included`, `solver_status`, `failure_reason` |
| `pointwise_placebo_summary` | `time`, `statistic`, `alternative`, `treated_statistic`, `extreme_count`, `assignment_count`, `p_value`, `p_value_resolution` |
| `aggregate_placebo_summary` | `statistic`, `alternative`, `periods`, `period_count`, `treated_statistic`, `extreme_count`, `assignment_count`, `p_value`, `p_value_resolution` |
| `specification_definitions` | `specification_id`, `pre_periods`, `omitted_predictors`, `predictor_periods`, `donors`, `solver_status`, `included`, `failure_reason` |
| `specification_diagnostics` | `specification_id`, scalar fit-diagnostic columns, `solver_status`, `included`, `exclusion_reason`, `failure_reason` |
| `specification_paths` | `specification_id`, `time`, `actual`, `synthetic`, `gap`, `is_post_treatment`, `included`, `solver_status`, `failure_reason` |
| `specification_weights` | `specification_id`, `donor`, `weight`, `included`, `solver_status`, `failure_reason` |
| `specification_balance` | `specification_id`, `predictor`, `treated`, `synthetic`, `difference`, `absolute_difference`, `relative_difference`, `predictor_weight`, `included`, `solver_status`, `failure_reason` |
| `diagnostics_table(suite)` | Same one-row schema as `diagnostics_table(fit)`, read from stored suite diagnostics |
| `robustness_summary` | `analysis`, `requested`, `status`, `successful`, `failed`, `filtered`, `inference_eligible`, `failure_reason` |

Use any Tables.jl sink for downstream work:

```julia
using DataFrames, CSV

DataFrame(weights_table(result))
CSV.write("weights.csv", weights_table(result))
```

## Validation

`from_table` rejects missing required columns, unknown predictor columns,
duplicate `(unit, time)` rows, absent treated units, empty donor pools,
unbalanced panels, missing or non-finite numeric values, non-numeric outcomes
or predictors, and treatment times outside the observed time values. Unit and
time identifiers may be strings, symbols, integers, or date-like values when
they support equality and deterministic ordering.

Tables built from `from_table` retain post-treatment paths for
[`path_table`](@ref). The Makie visualization helpers can use the same result
with explicit post-treatment arrays, or you can inspect the tabular path
directly before plotting.

The same retained panel powers robustness refits. Robustness summary tables
contain one row per attempted assignment, including failed or filtered
assignments; numeric statistics are `NaN` for failed fits and
`solver_status == :failed` identifies them. `exclusion_reason` distinguishes
poor-fit, non-finite-ratio, and failed in-space assignments.

Robustness path tables use complete stored paths for successful assignments,
including filtered in-space paths. Each failed assignment contributes one
sentinel row with its native analysis identifier and failure reason; time,
path, and post-treatment fields are `missing`. Rows are deterministic and
these reporting functions never trigger estimation.

Time-specific inference summaries consume typed, already-calculated
inference results. The pointwise table has one row per post-treatment period;
the aggregate table has one row and stores its native-time evaluation window
in the typed `periods` cell. `p_value_resolution` is the reciprocal of the
assignment count. Summary construction never reruns inference.

Specification-sensitivity tables retain input order and native unit/time
types. Successful tables use stored paths, weights, and diagnostics; each
failed specification contributes one sentinel row, while scalar diagnostics
use typed `missing`. Reporting never reconstructs or solves an SCM problem.

Suite summaries always contain four rows in component order: in-space,
leave-one-out, in-time, and specification sensitivity. Counts are `missing`
for disabled, skipped, or suite-level failed components. Successful
components report counts from stored results, including recorded refit
failures. Specification sensitivity uses `missing` for
`inference_eligible` because it is not a randomization-inference analysis.
