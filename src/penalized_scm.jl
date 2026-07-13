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

function Base.getproperty(problem::PenalizedSyntheticControlProblem, name::Symbol)
  if name in fieldnames(SyntheticControlData)
    return getproperty(getfield(problem, :data), name)
  end
  return getfield(problem, name)
end

function PenalizedSyntheticControlProblem(
  X1::Vector{T}, Y1::Vector{T},
  X0::Matrix{T}, Y0::Matrix{T},
  predictor_names::Vector{String}, donor_ids::Vector{String}, treated_id::String;
  lambda::Real=one(T)
) where {T<:AbstractFloat}
  data = SyntheticControlData(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)
  return PenalizedSyntheticControlProblem(data; lambda=lambda)
end

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

function penalized_predictor_loss(data::SyntheticControlData{T}, W::Vector{T}) where {T}
  uniform_predictor_weights = fill(one(T), length(data.X1_normalized))
  return weight_squared_distance(data.X0_normalized, data.X1_normalized, uniform_predictor_weights, W)
end

function weighted_pairwise_penalty(distances::Vector{T}, W::Vector{T}) where {T}
  penalty = zero(T)
  @inbounds for donor in eachindex(W, distances)
    penalty += W[donor] * distances[donor]
  end
  return penalty
end

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

function solve(prob::PenalizedSyntheticControlProblem{T}) where {T}
  update_penalized_donor_weight_objective!(prob)
  optimize_donor_weights!(prob.cache)
  return update_penalized_result!(prob)
end
