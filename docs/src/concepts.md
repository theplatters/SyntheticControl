# Concepts

## Data Layout

[`SyntheticControlData`](@ref) stores the aligned treated unit, donor pool,
and metadata used by both estimators.

- `X1`: treated-unit predictors, length `K`.
- `X0`: donor predictors, size `K × J`.
- `Y1`: treated-unit pre-treatment outcomes, length `T_pre`.
- `Y0`: donor pre-treatment outcomes, size `T_pre × J`.
- `predictor_names`, `donor_ids`, `treated_id`: metadata used to interpret
  weights.

The constructor checks dimensions, finite numeric values, and predictor
variation. It also stores normalized predictors in `X1_normalized` and
`X0_normalized`; solvers use the normalized values internally.

```jldoctest concepts
julia> using SyntheticControl

julia> data = SyntheticControlData(
           [1.0, 2.0],
           [10.0, 11.0, 12.0],
           [0.8 1.4 2.0; 1.6 2.2 2.8],
           [9.0 10.5 12.0;
            10.0 11.5 13.0;
            11.0 12.5 14.0],
           ["level", "trend"],
           ["A", "B", "C"],
           "treated",
       );

julia> (length(data.X1), size(data.X0), length(data.Y1), size(data.Y0))
(2, (2, 3), 3, (3, 3))
```

## Classic SCM

[`SyntheticControlProblem`](@ref) searches over predictor weights `V`. For each
candidate `V`, it solves an inner simplex-constrained donor-weight problem for
`W`, then evaluates outcome fit with [`SyntheticControl.calculate_mspe`](@ref).

The result is [`SyntheticControlResult`](@ref), with:

- `W`: donor weights, one per donor.
- `V`: predictor weights, one per predictor.
- `mspe`: pre-treatment mean squared prediction error.

## Penalized SCM

[`PenalizedSyntheticControlProblem`](@ref) uses a penalty strength `lambda` to
favor donors that are individually close to the treated unit in normalized
predictor space. It returns [`PenalizedSyntheticControlResult`](@ref), which
reports the predictor loss, penalty, total objective, and MSPE.
