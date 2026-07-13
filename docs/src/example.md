# End-to-End Example

This example constructs a small one-predictor dataset where donor `D1` is near
the treated unit and donors `D2` and `D3` provide alternative matches.

```jldoctest endtoend
julia> using SyntheticControl, CommonSolve

julia> X1 = [2.0];

julia> Y1 = [20.0, 21.0, 22.0];

julia> X0 = reshape([1.0, 4.0, 5.0], 1, 3);

julia> Y0 = [10.0 40.0 50.0;
             11.0 41.0 51.0;
             12.0 42.0 52.0];

julia> predictor_names = ["baseline"];

julia> donor_ids = ["D1", "D2", "D3"];

julia> classic = SyntheticControlProblem(
           X1, Y1, X0, Y0, predictor_names, donor_ids, "treated",
       );

julia> classic_result = solve(classic);

julia> sum(classic_result.W) ≈ 1.0
true

julia> all(>=(0), classic_result.W)
true
```

The fitted pre-treatment path is the donor outcome matrix multiplied by donor
weights:

```jldoctest endtoend
julia> fitted = Y0 * classic_result.W;

julia> length(fitted) == length(Y1)
true

julia> classic_result.mspe ≈ SyntheticControl.calculate_mspe(Y1, Y0, classic_result.W)
true
```

The penalized estimator uses the same data with a penalty parameter:

```jldoctest endtoend
julia> penalized = PenalizedSyntheticControlProblem(
           X1, Y1, X0, Y0, predictor_names, donor_ids, "treated";
           lambda=1.0,
       );

julia> penalized_result = solve(penalized);

julia> penalized_result.objective ≈
       penalized_result.predictor_loss + penalized.lambda * penalized_result.penalty
true
```

