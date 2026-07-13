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
| `placebo_summary` | `unit`, `is_treated`, `pre_rmspe`, `post_rmspe`, `rmspe_ratio`, `included`, `exclusion_reason`, `solver_status` |
| `leave_one_out_summary` | `omitted_donor`, `original_weight`, `pre_rmspe`, `post_rmspe`, `rmspe_ratio`, `mean_post_gap`, `cumulative_post_gap`, `max_path_deviation`, `solver_status` |
| `in_time_summary` | `placebo_time`, `pre_rmspe`, `post_rmspe`, `rmspe_ratio`, `mean_post_gap`, `cumulative_post_gap`, `pre_periods`, `post_periods`, `solver_status` |

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
