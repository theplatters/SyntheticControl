# Interpreting Results

[`SyntheticControlResult`](@ref) and [`PenalizedSyntheticControlResult`](@ref)
store fitted weights and diagnostics. Donor weights should be read together
with `data.donor_ids`; predictor weights in the classic result should be read
with `data.predictor_names`.

```jldoctest interpret
julia> using SyntheticControl, CommonSolve

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

julia> donor_weights = collect(zip(result.data.donor_ids, result.W));

julia> length(donor_weights)
3

julia> result.mspe >= 0
true
```

For the penalized solver, `predictor_loss`, `penalty`, and `objective` explain
the optimization tradeoff. `mspe` remains an outcome-fit diagnostic computed
from pre-treatment outcomes.

