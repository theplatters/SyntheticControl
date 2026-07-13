"""
    PenalizedSyntheticControlResult(data, W, lambda, predictor_loss, penalty, objective, mspe)
    PenalizedSyntheticControlResult(problem, W, predictor_loss, penalty, objective, mspe)
    PenalizedSyntheticControlResult{T}

Result from the penalized synthetic-control solver. `W` contains donor
weights, `lambda` is the non-negative penalty strength, `predictor_loss` is
the treated-to-synthetic predictor discrepancy, `penalty` is the
donor-weighted pairwise treated-to-donor predictor discrepancy, `objective` is
`predictor_loss + lambda * penalty`, and `mspe` is the pre-treatment outcome
MSPE.

The data constructor returns a `PenalizedSyntheticControlResult{T}` and
throws `DimensionMismatch` when `W` does not match the donor count or
`ArgumentError` when `lambda` is negative. It stores `W` without copying.

# Examples

```julia
data = SyntheticControlData([2.0], [20.0, 21.0], reshape([1.0, 4.0], 1, 2),
                            [10.0 40.0; 11.0 41.0],
                            ["x"], ["near", "far"], "treated")
result = SyntheticControl.PenalizedSyntheticControlResult(
  data, [1.0, 0.0], 1.0, 0.0, 0.0, 0.0, 0.0,
)
result.lambda == 1.0
```
"""
mutable struct PenalizedSyntheticControlResult{T<:AbstractFloat}
  data::SyntheticControlData{T}
  W::Vector{T}
  lambda::T
  predictor_loss::T
  penalty::T
  objective::T
  mspe::T

  function PenalizedSyntheticControlResult(
    data::SyntheticControlData{T},
    W::Vector{T},
    lambda::T,
    predictor_loss::T,
    penalty::T,
    objective::T,
    mspe::T
  ) where {T}
    length(W) == length(data.donor_ids) || throw(DimensionMismatch("Weight vector W must match donor pool size"))
    lambda >= zero(T) || throw(ArgumentError("lambda must be non-negative"))
    return new{T}(data, W, lambda, predictor_loss, penalty, objective, mspe)
  end
end

"""
    PenalizedSyntheticControlProblem(data; lambda=1)
    PenalizedSyntheticControlProblem(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id; lambda=1)
    PenalizedSyntheticControlProblem{T}

Cached problem definition for the penalized synthetic-control estimator. The
problem stores shared `SyntheticControlData`, non-negative penalty strength
`lambda`, an inner donor-weight cache, precomputed treated-to-donor predictor
distances, and a mutable result object.

The matrix constructor first creates `SyntheticControlData`. `lambda` must be
non-negative. Construction allocates caches and pairwise distances but does
not solve the problem.

# Examples

```julia
problem = PenalizedSyntheticControlProblem([2.0], [20.0, 21.0],
                                           reshape([1.0, 4.0], 1, 2),
                                           [10.0 40.0; 11.0 41.0],
                                           ["x"], ["near", "far"], "treated";
                                           lambda=2.0)
problem.lambda == 2.0
```
"""
mutable struct PenalizedSyntheticControlProblem{T<:AbstractFloat}
  data::SyntheticControlData{T}
  lambda::T
  cache::InnerWeightCache{T}
  pairwise_distances::Vector{T}
  result::PenalizedSyntheticControlResult{T}

  function PenalizedSyntheticControlProblem(
    data::SyntheticControlData{T};
    lambda::Real=one(T)
  ) where {T}
    lambda >= 0 || throw(ArgumentError("lambda must be non-negative"))
    cache = InnerWeightCache(data)
    pairwise_distances = Vector{T}(undef, length(data.donor_ids))
    update_pairwise_predictor_distances!(pairwise_distances, data)
    W = fill(one(T) / T(length(data.donor_ids)), length(data.donor_ids))
    result = PenalizedSyntheticControlResult(
      data,
      W,
      T(lambda),
      typemax(T),
      typemax(T),
      typemax(T),
      typemax(T)
    )
    return new{T}(data, T(lambda), cache, pairwise_distances, result)
  end
end

"""
    Base.getproperty(problem::PenalizedSyntheticControlProblem, name::Symbol)

Return a field from `problem` or, when `name` is a field of
`SyntheticControlData`, return the corresponding field from `problem.data`.

Throws the usual `type has no field` error for unknown names. It has no side
effects.

# Examples

```julia
problem = PenalizedSyntheticControlProblem([2.0], [20.0], reshape([1.0, 4.0], 1, 2),
                                           [10.0 40.0],
                                           ["x"], ["near", "far"], "treated")
problem.X1 == [2.0]
```
"""
function Base.getproperty(problem::PenalizedSyntheticControlProblem, name::Symbol)
  if name in fieldnames(SyntheticControlData)
    return getproperty(getfield(problem, :data), name)
  end
  return getfield(problem, name)
end

"""
    PenalizedSyntheticControlProblem(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id; lambda=1)

Construct a penalized SCM problem directly from aligned treated and donor
matrices. Arguments follow `SyntheticControlData`; `lambda` must be
non-negative.

Returns `PenalizedSyntheticControlProblem{T}`. It may throw
`DimensionMismatch` or `ArgumentError` during data or keyword validation.

# Examples

```julia
problem = PenalizedSyntheticControlProblem([2.0], [20.0], reshape([1.0, 4.0], 1, 2),
                                           [10.0 40.0],
                                           ["x"], ["near", "far"], "treated")
length(problem.pairwise_distances) == 2
```
"""
function PenalizedSyntheticControlProblem(
  X1::Vector{T}, Y1::Vector{T},
  X0::Matrix{T}, Y0::Matrix{T},
  predictor_names::Vector{String}, donor_ids::Vector{String}, treated_id::String;
  lambda::Real=one(T)
) where {T<:AbstractFloat}
  data = SyntheticControlData(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)
  return PenalizedSyntheticControlProblem(data; lambda=lambda)
end

"""
    PenalizedSyntheticControlResult(problem, W, predictor_loss, penalty, objective, mspe)

Create a penalized SCM result attached to `problem.data` and using
`problem.lambda`.

Returns `PenalizedSyntheticControlResult(problem.data, W, problem.lambda, ...)`.
It may throw `DimensionMismatch` for incompatible donor weights and stores
`W` without copying.

# Examples

```julia
problem = PenalizedSyntheticControlProblem([2.0], [20.0], reshape([1.0, 4.0], 1, 2),
                                           [10.0 40.0],
                                           ["x"], ["near", "far"], "treated")
result = SyntheticControl.PenalizedSyntheticControlResult(problem, [1.0, 0.0], 0.0, 0.0, 0.0, 0.0)
result.data === problem.data
```
"""
function PenalizedSyntheticControlResult(
  problem::PenalizedSyntheticControlProblem{T},
  W::Vector{T},
  predictor_loss::T,
  penalty::T,
  objective::T,
  mspe::T
) where {T}
  return PenalizedSyntheticControlResult(problem.data, W, problem.lambda, predictor_loss, penalty, objective, mspe)
end

"""
    update_pairwise_predictor_distances!(distances, data)

Compute squared Euclidean distances between the treated normalized predictor
vector and each donor normalized predictor vector.

Returns `distances`. Mutates `distances` in place; it must have one entry per
donor in `data`.

# Examples

```julia
data = SyntheticControlData([2.0], [20.0], reshape([1.0, 4.0], 1, 2), [10.0 40.0],
                            ["x"], ["near", "far"], "treated")
distances = zeros(2)
SyntheticControl.update_pairwise_predictor_distances!(distances, data)
all(>=(0), distances)
```
"""
function update_pairwise_predictor_distances!(distances::Vector{T}, data::SyntheticControlData{T}) where {T}
  X0 = data.X0_normalized
  X1 = data.X1_normalized
  n_predictors, n_donors = size(X0)

  @inbounds for donor in 1:n_donors
    distance = zero(T)
    for predictor in 1:n_predictors
      residual = X1[predictor] - X0[predictor, donor]
      distance += residual * residual
    end
    distances[donor] = distance
  end

  return distances
end

"""
    update_penalized_donor_weight_objective!(problem)

Rebuild the quadratic and linear terms for the penalized donor-weight
objective using `problem.data`, `problem.lambda`, and precomputed pairwise
distances.

Returns `nothing`. Mutates `problem.cache.quadratic.gram` and
`problem.cache.quadratic.linear`; reads all other fields.

# Examples

```julia
problem = PenalizedSyntheticControlProblem([2.0], [20.0], reshape([1.0, 4.0], 1, 2),
                                           [10.0 40.0],
                                           ["x"], ["near", "far"], "treated")
SyntheticControl.update_penalized_donor_weight_objective!(problem)
isfinite(problem.cache.quadratic.gram[1, 1])
```
"""
function update_penalized_donor_weight_objective!(problem::PenalizedSyntheticControlProblem{T}) where {T}
  cache = problem.cache
  X0 = problem.data.X0_normalized
  X1 = problem.data.X1_normalized
  gram = cache.quadratic.gram
  linear = cache.quadratic.linear
  distances = problem.pairwise_distances
  n_predictors, n_donors = size(X0)
  penalty_scale = problem.lambda / T(2)

  fill!(gram, zero(T))
  fill!(linear, zero(T))

  @inbounds for donor in 1:n_donors
    linear_donor = -penalty_scale * distances[donor]
    for predictor in 1:n_predictors
      x0_donor = X0[predictor, donor]
      linear_donor += x0_donor * X1[predictor]

      for other_donor in donor:n_donors
        gram[donor, other_donor] += x0_donor * X0[predictor, other_donor]
      end
    end
    linear[donor] = linear_donor
  end

  @inbounds for donor in 2:n_donors
    for other_donor in 1:(donor-1)
      gram[donor, other_donor] = gram[other_donor, donor]
    end
  end

  return nothing
end

"""
    penalized_predictor_loss(data, W)

Compute the unweighted normalized-predictor discrepancy between the treated
unit and the synthetic donor combination defined by `W`.

Returns a scalar predictor loss. `W` must have one entry per donor in `data`.
The function allocates a temporary all-ones predictor-weight vector and does
not mutate `data` or `W`.

# Examples

```julia
data = SyntheticControlData([2.0], [20.0], reshape([1.0, 4.0], 1, 2), [10.0 40.0],
                            ["x"], ["near", "far"], "treated")
SyntheticControl.penalized_predictor_loss(data, [1.0, 0.0]) >= 0
```
"""
function penalized_predictor_loss(data::SyntheticControlData{T}, W::Vector{T}) where {T}
  uniform_predictor_weights = fill(one(T), length(data.X1_normalized))
  return weight_squared_distance(data.X0_normalized, data.X1_normalized, uniform_predictor_weights, W)
end

"""
    weighted_pairwise_penalty(distances, W)

Compute the donor-weighted average of treated-to-donor predictor distances.

Returns `sum(W .* distances)` using an allocation-free loop. `distances` and
`W` must share compatible indices. It has no side effects.

# Examples

```julia
SyntheticControl.weighted_pairwise_penalty([1.0, 4.0], [0.25, 0.75]) ≈ 3.25
```
"""
function weighted_pairwise_penalty(distances::Vector{T}, W::Vector{T}) where {T}
  penalty = zero(T)
  @inbounds for donor in eachindex(W, distances)
    penalty += W[donor] * distances[donor]
  end
  return penalty
end

"""
    update_penalized_result!(problem)

Copy the current penalized donor weights from `problem.cache` into
`problem.result` and refresh predictor loss, penalty, objective, and MSPE.

Returns `problem.result`. Mutates result fields and reads problem data,
pairwise distances, and cache donor weights.

# Examples

```julia
problem = PenalizedSyntheticControlProblem([2.0], [20.0], reshape([1.0, 4.0], 1, 2),
                                           [10.0 40.0],
                                           ["x"], ["near", "far"], "treated")
SyntheticControl.update_penalized_donor_weight_objective!(problem)
SyntheticControl.optimize_donor_weights!(problem.cache)
result = SyntheticControl.update_penalized_result!(problem)
isfinite(result.objective)
```
"""
function update_penalized_result!(problem::PenalizedSyntheticControlProblem{T}) where {T}
  result = problem.result
  W = problem.cache.donor_weights
  copyto!(result.W, W)
  result.predictor_loss = penalized_predictor_loss(problem.data, result.W)
  result.penalty = weighted_pairwise_penalty(problem.pairwise_distances, result.W)
  result.objective = result.predictor_loss + problem.lambda * result.penalty
  result.mspe = calculate_mspe(problem.data.Y1, problem.data.Y0, result.W)
  return result
end

"""
    solve(prob::PenalizedSyntheticControlProblem)

Solve a penalized synthetic-control problem using the CommonSolve interface.

Returns a `PenalizedSyntheticControlResult`. Mutates the problem's inner cache
and result object; repeated calls reuse the same allocations.

# Examples

```julia
using SyntheticControl, CommonSolve

problem = PenalizedSyntheticControlProblem([2.0], [20.0], reshape([1.0, 4.0], 1, 2),
                                           [10.0 40.0],
                                           ["x"], ["near", "far"], "treated")
result = solve(problem)
sum(result.W) ≈ 1.0
```
"""
function solve(prob::PenalizedSyntheticControlProblem{T}) where {T}
  update_penalized_donor_weight_objective!(prob)
  optimize_donor_weights!(prob.cache)
  return update_penalized_result!(prob)
end
