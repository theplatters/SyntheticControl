# Implementation Details

The following symbols are implementation details. They are documented so
contributors can maintain the solver and so `checkdocs=:all` can enforce
coverage, but they should not be treated as stable public API.

## Caches and Search Configuration

```@docs
SyntheticControl.QuadraticModelCache
SyntheticControl.ActiveSetCache
SyntheticControl.PredictorSearchWorkspace
SyntheticControl.InnerWeightCache
SyntheticControl.OuterSearchResults
SyntheticControl.SearchDepth
```

## Classic Solver Internals

```@docs
SyntheticControl.Base.getproperty(::SyntheticControl.SyntheticControlProblem, ::Symbol)
SyntheticControl.InnerWeightCache(::SyntheticControl.SyntheticControlData)
SyntheticControl.InnerWeightCache(::SyntheticControl.SyntheticControlProblem)
SyntheticControl.normalize_weights!
SyntheticControl.normalize_weight_column!
SyntheticControl.write_column!
SyntheticControl.read_column!
SyntheticControl.update_donor_weight_objective!
SyntheticControl.solve_active_set_kkt!
SyntheticControl.collect_active_indices!
SyntheticControl.project_to_simplex!
SyntheticControl.optimize_donor_weights!
SyntheticControl.evaluate_predictor_weights!
SyntheticControl.regression_predictor_weight_start
SyntheticControl.push_unique_predictor_start!
SyntheticControl.predictor_pair_from_ordinal
SyntheticControl.push_pair_predictor_starts!
SyntheticControl.build_predictor_starts
SyntheticControl.select_largest_weights!
SyntheticControl.select_spread_indices!
SyntheticControl.search_depths
SyntheticControl.optimize_predictor_weights_from_start!
SyntheticControl.run_search_depth!
SyntheticControl.best_search_result_index
SyntheticControl.keep_best_search_result!
SyntheticControl.target_mspe_reached
SyntheticControl.mspe_improvement_is_small
SyntheticControl.run_outer_search!
```

## Penalized Solver Internals

```@docs
SyntheticControl.Base.getproperty(::SyntheticControl.PenalizedSyntheticControlProblem, ::Symbol)
SyntheticControl.update_pairwise_predictor_distances!
SyntheticControl.update_penalized_donor_weight_objective!
SyntheticControl.penalized_predictor_loss
SyntheticControl.weighted_pairwise_penalty
SyntheticControl.update_penalized_result!
```

