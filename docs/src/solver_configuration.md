# Solver Configuration

## Classic Solver

[`SyntheticControlProblem`](@ref) accepts keyword arguments that control the
outer predictor-weight search:

- `max_pair_starts`: add two-predictor starting points beyond the default
  uniform, regression-derived, and corner starts.
- `target_mspe`: stop when the best MSPE is no larger than this positive
  target. The default `0` disables this target.
- `min_relative_mspe_improvement`: stop later search-depth passes when the
  relative improvement is small.

All three values must be non-negative.

```jldoctest config
julia> using SyntheticControl

julia> problem = SyntheticControlProblem(
           [1.0, 2.0],
           [1.0, 1.5, 2.0],
           [0.5 1.5 2.5; 1.5 2.5 3.5],
           [0.8 1.2 1.8;
            1.3 1.7 2.3;
            1.8 2.2 2.8],
           ["level", "trend"],
           ["A", "B", "C"],
           "treated";
           max_pair_starts=1,
           target_mspe=0.0,
           min_relative_mspe_improvement=0.005,
       );

julia> problem.nstarts >= 3
true
```

## Penalized Solver

[`PenalizedSyntheticControlProblem`](@ref) accepts `lambda`, a non-negative
penalty strength. Larger values put more weight on individually close donors.

```jldoctest config
julia> penalized = PenalizedSyntheticControlProblem(
           [2.0], [20.0], reshape([1.0, 4.0], 1, 2), [10.0 40.0],
           ["baseline"], ["near", "far"], "treated";
           lambda=3.0,
       );

julia> penalized.lambda == 3.0
true
```

