# API Reference

## Public API

```@docs
SyntheticControlData
SyntheticControlProblem
SyntheticControlResult
SyntheticControlResult(::SyntheticControlProblem{T}, ::Vector{T}, ::Vector{T}, ::T) where {T}
PenalizedSyntheticControlProblem
PenalizedSyntheticControlResult
PenalizedSyntheticControlResult(::PenalizedSyntheticControlProblem{T}, ::Vector{T}, ::T, ::T, ::T, ::T) where {T}
CommonSolve.solve(::SyntheticControlProblem)
CommonSolve.solve(::PenalizedSyntheticControlProblem)
```

## Public Mathematical Helpers

These helpers are not exported, but they are part of the module-level API used
by tests and examples.

```@docs
SyntheticControl.weight_squared_distance
SyntheticControl.calculate_mspe
```
