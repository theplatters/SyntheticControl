# SyntheticControl.jl

`SyntheticControl.jl` implements classic and penalized synthetic-control
estimators for aligned pre-treatment predictor and outcome matrices.

The main workflow is:

1. Build [`SyntheticControlData`](@ref), or construct a problem directly from
   matrices.
2. Choose [`SyntheticControlProblem`](@ref) for classic SCM or
   [`PenalizedSyntheticControlProblem`](@ref) for the penalized variant.
3. Call [`CommonSolve.solve`](@ref) and inspect the returned result.

```jldoctest quickstart
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

julia> sum(result.W) ≈ 1.0
true
```

## Pages

- [Installation](@ref)
- [Concepts](@ref)
- [End-to-End Example](@ref)
- [Solver Configuration](@ref)
- [Interpreting Results](@ref)
- [Visualization](@ref)
- [API Reference](@ref)
- [Implementation Details](@ref)
