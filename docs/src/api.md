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
SyntheticControlPathData
SyntheticControlPlaceboResult
from_table
weights_table
balance_table
path_table
```

## Public Mathematical Helpers

These helpers are not exported, but they are part of the module-level API used
by tests and examples.

```@docs
SyntheticControl.weight_squared_distance
SyntheticControl.calculate_mspe
actual_outcome
synthetic_outcome
outcome_gap
placebo_gaps
pre_treatment_rmspe
post_treatment_rmspe
rmspe_ratio
placebo_rmspe_ratios
filter_placebos
randomization_p_value
pathplot
pathplot!
gapplot
gapplot!
placeboplot
placeboplot!
placebodistribution
placebodistribution!
```
