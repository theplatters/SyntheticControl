module SyntheticControl

using CommonSolve, LinearAlgebra
import CommonSolve: solve
using Base.Threads: @threads, nthreads
using Statistics
export SyntheticControlData, SyntheticControlProblem, SyntheticControlResult
export solve

const DEFAULT_MAX_PAIR_STARTS = 0
const DEFAULT_MAX_OUTER_ITERS = 50
const DEFAULT_MAX_OUTER_EVALUATIONS = 200
const DEFAULT_MAX_TRANSFER_DONORS = 4
const DEFAULT_MAX_TRANSFER_RECEIVERS = 8
const DEFAULT_MIN_RELATIVE_MSPE_IMPROVEMENT = 0.01

"""
    SyntheticControlData{T<:AbstractFloat}

Holds the cleanly aligned matrices and vectors required to solve a 
Synthetic Control Method optimization problem.
"""
struct SyntheticControlData{T<:AbstractFloat}
  # 1. Target Data (Treated Unit)
  X1::Vector{T}      # K x 1 vector of pre-treatment predictors
  X1_normalized::Vector{T}      # K x 1 vector of pre-treatment predictors
  Y1::Vector{T}      # T_pre x 1 vector of pre-treatment outcomes

  # 2. Donor Pool Data (Control Units)
  X0::Matrix{T}      # K x J matrix of predictors for J control units
  X0_normalized::Matrix{T}      # K x J matrix of predictors for J control units
  Y0::Matrix{T}      # T_pre x J matrix of outcomes for J control units

  # 3. Metadata for Tracking & Validation
  predictor_names::Vector{String}
  donor_ids::Vector{String}
  treated_id::String

  # Inner constructor for shape and dimension verification
  function SyntheticControlData(
    X1::Vector{T}, Y1::Vector{T},
    X0::Matrix{T}, Y0::Matrix{T},
    predictor_names::Vector{String}, donor_ids::Vector{String}, treated_id::String
  ) where {T<:AbstractFloat}

    K, J = size(X0)
    T_pre = length(Y1)

    length(X1) == K || throw(DimensionMismatch("X1 length must match rows of X0 (K)"))
    size(Y0, 1) == T_pre || throw(DimensionMismatch("Y0 rows must match length of Y1 (T_pre)"))
    size(Y0, 2) == J || throw(DimensionMismatch("Y0 columns must match columns of X0 (J)"))
    length(predictor_names) == K || throw(DimensionMismatch("Must provide K predictor names"))
    length(donor_ids) == J || throw(DimensionMismatch("Must provide J donor IDs"))
    all(isfinite, X1) || throw(ArgumentError("X1 contains Infs or NaNs"))
    all(isfinite, Y1) || throw(ArgumentError("Y1 contains Infs or NaNs"))
    all(isfinite, X0) || throw(ArgumentError("X0 contains Infs or NaNs"))
    all(isfinite, Y0) || throw(ArgumentError("Y0 contains Infs or NaNs"))

    std_divisor = vec(std(hcat(X0, X1), dims=2))
    all(isfinite, std_divisor) || throw(ArgumentError("Predictor standard deviations contain Infs or NaNs"))
    any(<(eps(T)), std_divisor) && throw(ArgumentError("Predictors have almost zero variance"))
    return new{T}(X1, X1 ./ std_divisor, Y1, X0, X0 ./ std_divisor, Y0, predictor_names, donor_ids, treated_id)
  end
end

mutable struct QuadraticModelCache{T<:AbstractFloat}
  gram::Matrix{T}
  linear::Vector{T}
end

mutable struct ActiveSetCache{T<:AbstractFloat}
  is_active::Vector{Bool}
  indices::Vector{Int}
  kkt::Matrix{T}
  rhs::Vector{T}
  solution::Vector{T}
  pivots::Vector{Vector{LinearAlgebra.BlasInt}}
end

mutable struct PredictorSearchWorkspace{T<:AbstractFloat}
  current::Vector{T}
  candidate::Vector{T}
  best_candidate::Vector{T}
  transfer_donors::Vector{Int}
  transfer_receivers::Vector{Int}
end

mutable struct InnerWeightCache{T<:AbstractFloat}
  data::SyntheticControlData{T}
  predictor_weights::Vector{T}
  donor_weights::Vector{T}
  quadratic::QuadraticModelCache{T}
  active_set::ActiveSetCache{T}
  predictor_search::PredictorSearchWorkspace{T}
end

mutable struct OuterSearchResults{T<:AbstractFloat}
  predictor_weights::Matrix{T}
  donor_weights::Matrix{T}
  mspe::Vector{T}
end

struct SearchDepth{T<:AbstractFloat}
  max_iters::Int
  max_evaluations::Int
  max_transfer_donors::Int
  max_transfer_receivers::Int
  min_transfer_weight::T
end

mutable struct SyntheticControlResult{T<:AbstractFloat}
  data::SyntheticControlData{T}
  W::Vector{T}                         # Optimal unit weights (J x 1)
  V::Vector{T}                         # Optimal predictor weights (diagonal elements, K x 1)
  mspe::T                              # Final pre-treatment Mean Squared Prediction Error

  function SyntheticControlResult(data::SyntheticControlData{T}, W::Vector{T}, V::Vector{T}, mspe::T) where {T}
    length(W) == length(data.donor_ids) || throw(DimensionMismatch("Weight vector W must match donor pool size"))
    length(V) == length(data.predictor_names) || throw(DimensionMismatch("Weight vector V must match predictor count"))
    return new{T}(data, W, V, mspe)
  end
end

mutable struct SyntheticControlProblem{T<:AbstractFloat}
  data::SyntheticControlData{T}
  starts::Matrix{T}
  nstarts::Int
  inner_caches::Vector{InnerWeightCache{T}}
  search_results::OuterSearchResults{T}
  best_predictor_weights::Vector{T}
  best_donor_weights::Vector{T}
  target_mspe::T
  min_relative_mspe_improvement::T
  result::SyntheticControlResult{T}

  function SyntheticControlProblem(
    data::SyntheticControlData{T};
    max_pair_starts::Int=DEFAULT_MAX_PAIR_STARTS,
    target_mspe::Real=zero(T),
    min_relative_mspe_improvement::Real=DEFAULT_MIN_RELATIVE_MSPE_IMPROVEMENT
  ) where {T}
    max_pair_starts >= 0 || throw(ArgumentError("max_pair_starts must be non-negative"))
    target_mspe >= 0 || throw(ArgumentError("target_mspe must be non-negative"))
    min_relative_mspe_improvement >= 0 || throw(ArgumentError("min_relative_mspe_improvement must be non-negative"))
    n_predictors, n_donors = size(data.X0)
    starts = build_predictor_starts(data; max_pair_starts=max_pair_starts)
    nstarts = size(starts, 2)
    inner_caches = [InnerWeightCache(data) for _ in 1:nstarts]
    search_results = OuterSearchResults(
      Matrix{T}(undef, n_predictors, nstarts),
      Matrix{T}(undef, n_donors, nstarts),
      Vector{T}(undef, nstarts)
    )
    best_predictor_weights = fill(one(T) / T(n_predictors), n_predictors)
    best_donor_weights = fill(one(T) / T(n_donors), n_donors)
    result = SyntheticControlResult(data, best_donor_weights, best_predictor_weights, typemax(T))
    return new{T}(
      data,
      starts,
      nstarts,
      inner_caches,
      search_results,
      best_predictor_weights,
      best_donor_weights,
      T(target_mspe),
      T(min_relative_mspe_improvement),
      result
    )
  end
end

function Base.getproperty(problem::SyntheticControlProblem, name::Symbol)
  if name in fieldnames(SyntheticControlData)
    return getproperty(getfield(problem, :data), name)
  end
  return getfield(problem, name)
end

function SyntheticControlProblem(
  X1::Vector{T}, Y1::Vector{T},
  X0::Matrix{T}, Y0::Matrix{T},
  predictor_names::Vector{String}, donor_ids::Vector{String}, treated_id::String;
  max_pair_starts::Int=DEFAULT_MAX_PAIR_STARTS,
  target_mspe::Real=zero(T),
  min_relative_mspe_improvement::Real=DEFAULT_MIN_RELATIVE_MSPE_IMPROVEMENT
) where {T<:AbstractFloat}
  data = SyntheticControlData(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)
  return SyntheticControlProblem(
    data;
    max_pair_starts=max_pair_starts,
    target_mspe=target_mspe,
    min_relative_mspe_improvement=min_relative_mspe_improvement
  )
end

InnerWeightCache(problem::SyntheticControlProblem{T}) where {T} = InnerWeightCache(problem.data)

function SyntheticControlResult(problem::SyntheticControlProblem{T}, W::Vector{T}, V::Vector{T}, mspe::T) where {T}
  return SyntheticControlResult(problem.data, W, V, mspe)
end

function weight_squared_distance(X0, X1, V, W)
  dist = zero(eltype(W))
  for k in eachindex(X1)
    res_k = X1[k]
    for j in eachindex(W)
      @inbounds res_k -= X0[k, j] * W[j]
    end
    dist += res_k * res_k * V[k]
  end
  return dist
end


function calculate_mspe(Y1, Y0, W)
  T0, J = size(Y0)

  s = zero(eltype(Y1))

  @inbounds for t in 1:T0
    r = Y1[t]

    for j in 1:J
      r -= Y0[t, j] * W[j]
    end

    s += r * r
  end

  return s / T0
end


function InnerWeightCache(data::SyntheticControlData{T}) where {T}
  n_predictors, n_donors = size(data.X0)
  predictor_weights = fill(one(T) / n_predictors, n_predictors)
  donor_weights = fill(one(T) / n_donors, n_donors)
  quadratic = QuadraticModelCache(
    zeros(T, n_donors, n_donors),
    zeros(T, n_donors)
  )
  active_set = ActiveSetCache(
    fill(true, n_donors),
    collect(1:n_donors),
    zeros(T, n_donors + 1, n_donors + 1),
    zeros(T, n_donors + 1),
    zeros(T, n_donors + 1),
    [Vector{LinearAlgebra.BlasInt}(undef, n) for n in 1:(n_donors + 1)]
  )
  predictor_search = PredictorSearchWorkspace(
    Vector{T}(undef, n_predictors),
    Vector{T}(undef, n_predictors),
    Vector{T}(undef, n_predictors),
    Vector{Int}(undef, n_predictors),
    Vector{Int}(undef, n_predictors)
  )

  return InnerWeightCache(
    data,
    predictor_weights,
    donor_weights,
    quadratic,
    active_set,
    predictor_search
  )
end

function normalize_weights!(dest::AbstractVector{T}, src::AbstractVector) where {T}
  s = zero(T)

  @inbounds for i in eachindex(dest, src)
    v = abs(T(src[i]))
    dest[i] = v
    s += v
  end

  if !(isfinite(s)) || s <= eps(T)
    fill!(dest, inv(T(length(dest))))
  else
    inv_s = inv(s)
    @inbounds for i in eachindex(dest)
      dest[i] *= inv_s
    end
  end

  return dest
end

function normalize_weight_column!(dest::AbstractVector{T}, src::AbstractMatrix{T}, col::Int) where {T}
  s = zero(T)

  @inbounds for i in eachindex(dest)
    v = abs(src[i, col])
    dest[i] = v
    s += v
  end

  if !(isfinite(s)) || s <= eps(T)
    fill!(dest, inv(T(length(dest))))
  else
    inv_s = inv(s)
    @inbounds for i in eachindex(dest)
      dest[i] *= inv_s
    end
  end

  return dest
end

function write_column!(dest::AbstractMatrix{T}, col::Int, src::AbstractVector{T}) where {T}
  @inbounds for i in eachindex(src)
    dest[i, col] = src[i]
  end
  return dest
end

function read_column!(dest::AbstractVector{T}, src::AbstractMatrix{T}, col::Int) where {T}
  @inbounds for i in eachindex(dest)
    dest[i] = src[i, col]
  end
  return dest
end

function update_donor_weight_objective!(cache::InnerWeightCache{T}) where {T}
  X0 = cache.data.X0_normalized
  X1 = cache.data.X1_normalized
  predictor_weights = cache.predictor_weights
  gram = cache.quadratic.gram
  linear = cache.quadratic.linear
  n_predictors, n_donors = size(X0)

  fill!(gram, zero(T))
  fill!(linear, zero(T))

  @inbounds for donor in 1:n_donors
    linear_donor = zero(T)
    for predictor in 1:n_predictors
      x0_donor = X0[predictor, donor]
      weighted_x0_donor = predictor_weights[predictor] * x0_donor
      linear_donor += weighted_x0_donor * X1[predictor]

      for other_donor in donor:n_donors
        gram[donor, other_donor] += weighted_x0_donor * X0[predictor, other_donor]
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

function solve_active_set_kkt!(cache::InnerWeightCache{T}, nactive::Int) where {T}
  gram = cache.quadratic.gram
  linear = cache.quadratic.linear
  active_set = cache.active_set
  indices = active_set.indices
  kkt = active_set.kkt
  rhs = active_set.rhs
  solution = active_set.solution
  ridge = sqrt(eps(T))

  fill!(kkt, zero(T))
  fill!(rhs, zero(T))

  @inbounds for a in 1:nactive
    donor_a = indices[a]
    rhs[a] = 2 * linear[donor_a]
    kkt[a, nactive+1] = one(T)
    kkt[nactive+1, a] = one(T)

    for b in 1:nactive
      donor_b = indices[b]
      kkt[a, b] = 2 * gram[donor_a, donor_b]
    end
    kkt[a, a] += ridge
  end
  rhs[nactive+1] = one(T)

  active_system = @views kkt[1:(nactive+1), 1:(nactive+1)]
  active_solution = @views solution[1:(nactive+1)]
  @inbounds for i in 1:(nactive+1)
    active_solution[i] = rhs[i]
  end
  pivots = active_set.pivots[nactive+1]
  LinearAlgebra.LAPACK.getrf!(active_system, pivots)
  LinearAlgebra.LAPACK.getrs!('N', active_system, pivots, active_solution)

  return solution[nactive+1]
end

function collect_active_indices!(indices::Vector{Int}, is_active::Vector{Bool})
  nactive = 0

  @inbounds for donor in eachindex(is_active)
    if is_active[donor]
      nactive += 1
      indices[nactive] = donor
    end
  end

  return nactive
end

function project_to_simplex!(weights::Vector{T}) where {T}
  s = zero(T)

  @inbounds for i in eachindex(weights)
    if weights[i] < zero(T)
      weights[i] = zero(T)
    end
    s += weights[i]
  end

  if s <= eps(T)
    fill!(weights, inv(T(length(weights))))
  else
    inv_s = inv(s)
    @inbounds for i in eachindex(weights)
      weights[i] *= inv_s
    end
  end

  return weights
end

function optimize_donor_weights!(cache::InnerWeightCache{T}) where {T}
  n_donors = length(cache.donor_weights)
  gram = cache.quadratic.gram
  linear = cache.quadratic.linear
  donor_weights = cache.donor_weights
  active_set = cache.active_set
  is_active = active_set.is_active
  indices = active_set.indices
  solution = active_set.solution
  tol = sqrt(eps(T))

  fill!(is_active, true)

  for _ in 1:(3n_donors)
    nactive = collect_active_indices!(indices, is_active)
    if nactive == 0
      fill!(donor_weights, inv(T(n_donors)))
      return donor_weights
    end

    lambda = solve_active_set_kkt!(cache, nactive)
    fill!(donor_weights, zero(T))

    min_weight = zero(T)
    donor_to_drop = 0
    @inbounds for a in 1:nactive
      donor = indices[a]
      weight = solution[a]
      donor_weights[donor] = weight
      if weight < min_weight
        min_weight = weight
        donor_to_drop = donor
      end
    end

    if donor_to_drop != 0 && min_weight < -tol
      is_active[donor_to_drop] = false
      continue
    end

    donor_to_add = 0
    min_multiplier = zero(T)
    @inbounds for donor in 1:n_donors
      if !is_active[donor]
        multiplier = lambda - 2 * linear[donor]
        for a in 1:nactive
          active_donor = indices[a]
          multiplier += 2 * gram[donor, active_donor] * donor_weights[active_donor]
        end

        if multiplier < min_multiplier
          min_multiplier = multiplier
          donor_to_add = donor
        end
      end
    end

    if donor_to_add != 0 && min_multiplier < -tol
      is_active[donor_to_add] = true
      continue
    end

    return project_to_simplex!(donor_weights)
  end

  return project_to_simplex!(donor_weights)
end

function evaluate_predictor_weights!(cache::InnerWeightCache{T}, raw_predictor_weights) where {T}
  normalize_weights!(cache.predictor_weights, raw_predictor_weights)
  update_donor_weight_objective!(cache)
  optimize_donor_weights!(cache)

  return calculate_mspe(cache.data.Y1, cache.data.Y0, cache.donor_weights)
end

function regression_predictor_weight_start(data::SyntheticControlData{T}) where {T}
  Xall = hcat(data.X1_normalized, data.X0_normalized) # K × (J+1)
  Xreg = hcat(ones(T, size(Xall, 2)), Xall')          # (J+1) × (K+1)
  Zall = hcat(data.Y1, data.Y0)                       # T0 × (J+1)

  β = try
    Xreg \ Zall'
  catch
    return nothing
  end

  β = β[2:end, :]              # drop intercept; K × T0
  Vmat = β * β'                # K × K
  v = abs.(diag(Vmat))

  s = sum(v)
  if !(isfinite(s)) || s <= eps(T)
    return nothing
  end

  return v ./ s
end

function push_unique_predictor_start!(starts::Vector{Vector{T}}, raw_weights, scratch::Vector{T}) where {T}
  normalize_weights!(scratch, raw_weights)
  threshold = T(10) * eps(T)

  for start in starts
    already_present = true
    @inbounds for i in eachindex(start, scratch)
      if abs(start[i] - scratch[i]) > threshold
        already_present = false
        break
      end
    end
    already_present && return starts
  end

  push!(starts, copy(scratch))
  return starts
end

function predictor_pair_from_ordinal(n_predictors::Int, ordinal::Int)
  remaining = ordinal

  for first_predictor in 1:(n_predictors-1)
    npairs = n_predictors - first_predictor
    if remaining <= npairs
      return first_predictor, first_predictor + remaining
    end
    remaining -= npairs
  end

  return n_predictors - 1, n_predictors
end

function push_pair_predictor_starts!(
  starts::Vector{Vector{T}},
  scratch::Vector{T},
  n_predictors::Int,
  max_pair_starts::Int
) where {T}
  total_pairs = div(n_predictors * (n_predictors - 1), 2)
  pair_start_count = min(total_pairs, max_pair_starts)
  pair = zeros(T, n_predictors)
  half = one(T) / T(2)

  for sample in 1:pair_start_count
    ordinal = pair_start_count == total_pairs ? sample : 1 + fld((sample - 1) * total_pairs, pair_start_count)
    first_predictor, second_predictor = predictor_pair_from_ordinal(n_predictors, ordinal)

    fill!(pair, zero(T))
    pair[first_predictor] = half
    pair[second_predictor] = half
    push_unique_predictor_start!(starts, pair, scratch)
  end

  return starts
end

function build_predictor_starts(data::SyntheticControlData{T}; max_pair_starts::Int=DEFAULT_MAX_PAIR_STARTS) where {T}
  n_predictors = length(data.X1)
  starts = Vector{Vector{T}}()
  scratch = Vector{T}(undef, n_predictors)

  push_unique_predictor_start!(starts, fill(one(T) / T(n_predictors), n_predictors), scratch)

  regression_start = regression_predictor_weight_start(data)
  if regression_start !== nothing
    push_unique_predictor_start!(starts, regression_start, scratch)
  end

  corner = zeros(T, n_predictors)
  for predictor in 1:n_predictors
    fill!(corner, zero(T))
    corner[predictor] = one(T)
    push_unique_predictor_start!(starts, corner, scratch)
  end

  push_pair_predictor_starts!(starts, scratch, n_predictors, max_pair_starts)

  starts_matrix = Matrix{T}(undef, n_predictors, length(starts))
  for start_idx in eachindex(starts)
    write_column!(starts_matrix, start_idx, starts[start_idx])
  end

  return starts_matrix
end

function select_largest_weights!(
  selected::Vector{Int},
  weights::Vector{T},
  max_count::Int,
  min_weight::T
) where {T}
  limit = min(max_count, length(weights))
  limit <= 0 && return 0
  nselected = 0

  @inbounds for idx in eachindex(weights)
    weight = weights[idx]
    weight <= min_weight && continue

    if nselected < limit
      nselected += 1
      pos = nselected
    elseif weight > weights[selected[nselected]]
      pos = nselected
    else
      continue
    end

    while pos > 1 && weight > weights[selected[pos-1]]
      selected[pos] = selected[pos-1]
      pos -= 1
    end
    selected[pos] = idx
  end

  return nselected
end

function select_spread_indices!(selected::Vector{Int}, nitems::Int, max_count::Int, offset::Int)
  count = min(max_count, nitems)
  count <= 0 && return 0

  @inbounds for sample in 1:count
    base_idx = 1 + fld((sample - 1) * nitems, count)
    selected[sample] = 1 + mod(base_idx + offset - 1, nitems)
  end

  return count
end

function search_depths(::Type{T}, n_predictors::Int) where {T<:AbstractFloat}
  return (
    SearchDepth{T}(
      DEFAULT_MAX_OUTER_ITERS,
      DEFAULT_MAX_OUTER_EVALUATIONS,
      min(DEFAULT_MAX_TRANSFER_DONORS, n_predictors),
      min(DEFAULT_MAX_TRANSFER_RECEIVERS, n_predictors),
      T(1.0e-6)
    ),
    SearchDepth{T}(
      100,
      600,
      min(8, n_predictors),
      min(12, n_predictors),
      T(1.0e-7)
    ),
    SearchDepth{T}(
      150,
      1600,
      min(12, n_predictors),
      min(16, n_predictors),
      T(1.0e-8)
    ),
    SearchDepth{T}(
      250,
      5000,
      n_predictors,
      n_predictors,
      zero(T)
    )
  )
end

function optimize_predictor_weights_from_start!(
  cache::InnerWeightCache{T},
  starts::AbstractMatrix{T},
  start_idx::Int,
  results::OuterSearchResults{T};
  depth::SearchDepth{T}=first(search_depths(T, length(cache.predictor_weights))),
  initial_step::T=T(0.25),
  min_step::T=T(1.0e-4),
  shrink::T=T(0.5),
  tol::T=sqrt(eps(T))
) where {T}
  n_predictors = length(cache.predictor_weights)
  workspace = cache.predictor_search
  current = workspace.current
  candidate = workspace.candidate
  best_candidate = workspace.best_candidate
  transfer_donors = workspace.transfer_donors
  transfer_receivers = workspace.transfer_receivers

  normalize_weight_column!(current, starts, start_idx)
  mspe = evaluate_predictor_weights!(cache, current)
  step = initial_step
  evaluations = 1

  for _ in 1:depth.max_iters
    (step < min_step || evaluations >= depth.max_evaluations) && break

    best_neighbor_mspe = mspe
    found_improvement = false
    ndonors = select_largest_weights!(
      transfer_donors,
      current,
      depth.max_transfer_donors,
      max(depth.min_transfer_weight, eps(T))
    )
    nreceivers = select_spread_indices!(
      transfer_receivers,
      n_predictors,
      depth.max_transfer_receivers,
      start_idx
    )

    @inbounds for donor_idx in 1:ndonors
      donor = transfer_donors[donor_idx]
      for receiver_idx in 1:nreceivers
        evaluations >= depth.max_evaluations && break
        receiver = transfer_receivers[receiver_idx]
        receiver == donor && continue
        delta = min(step, current[donor])
        delta <= depth.min_transfer_weight && continue

        copyto!(candidate, current)
        candidate[receiver] += delta
        candidate[donor] -= delta

        candidate_mspe = evaluate_predictor_weights!(cache, candidate)
        evaluations += 1
        if candidate_mspe + tol < best_neighbor_mspe
          best_neighbor_mspe = candidate_mspe
          copyto!(best_candidate, cache.predictor_weights)
          found_improvement = true
        end
      end
      evaluations >= depth.max_evaluations && break
    end

    if found_improvement
      copyto!(current, best_candidate)
      mspe = evaluate_predictor_weights!(cache, current)
      evaluations += 1
    else
      step *= shrink
    end
  end

  mspe = evaluate_predictor_weights!(cache, current)
  write_column!(results.predictor_weights, start_idx, cache.predictor_weights)
  write_column!(results.donor_weights, start_idx, cache.donor_weights)
  return mspe
end

function run_search_depth!(problem::SyntheticControlProblem{T}, depth::SearchDepth{T}) where {T}
  starts = problem.starts
  results = problem.search_results

  if nthreads() == 1
    for s in 1:problem.nstarts
      cache = problem.inner_caches[s]
      results.mspe[s] = optimize_predictor_weights_from_start!(cache, starts, s, results; depth=depth)
    end
  else
    @threads for s in 1:problem.nstarts
      cache = problem.inner_caches[s]
      results.mspe[s] = optimize_predictor_weights_from_start!(cache, starts, s, results; depth=depth)
    end
  end

  return nothing
end

function best_search_result_index(results::OuterSearchResults{T}, nstarts::Int) where {T}
  best_idx = 1
  best_mspe = results.mspe[1]

  @inbounds for s in 2:nstarts
    candidate_mspe = results.mspe[s]
    if candidate_mspe < best_mspe
      best_idx = s
      best_mspe = candidate_mspe
    end
  end

  return best_idx, best_mspe
end

function keep_best_search_result!(
  problem::SyntheticControlProblem{T},
  candidate_idx::Int,
  candidate_mspe::T,
  best_mspe::T
) where {T}
  if candidate_mspe < best_mspe
    results = problem.search_results
    read_column!(problem.best_predictor_weights, results.predictor_weights, candidate_idx)
    read_column!(problem.best_donor_weights, results.donor_weights, candidate_idx)
    problem.result.mspe = candidate_mspe
    return candidate_mspe
  end

  return best_mspe
end

function target_mspe_reached(problem::SyntheticControlProblem{T}, mspe::T) where {T}
  return problem.target_mspe > zero(T) && mspe <= problem.target_mspe
end

function mspe_improvement_is_small(previous_mspe::T, current_mspe::T, min_relative_improvement::T) where {T}
  !(isfinite(previous_mspe)) && return false
  improvement = previous_mspe - current_mspe
  improvement <= zero(T) && return true
  return improvement <= min_relative_improvement * max(abs(previous_mspe), eps(T))
end

function run_outer_search!(problem::SyntheticControlProblem{T}) where {T}
  depths = search_depths(T, length(problem.best_predictor_weights))
  best_mspe = typemax(T)
  previous_depth_mspe = typemax(T)

  for depth in depths
    run_search_depth!(problem, depth)
    best_idx, depth_mspe = best_search_result_index(problem.search_results, problem.nstarts)
    best_mspe = keep_best_search_result!(problem, best_idx, depth_mspe, best_mspe)

    target_mspe_reached(problem, best_mspe) && break
    mspe_improvement_is_small(previous_depth_mspe, best_mspe, problem.min_relative_mspe_improvement) && break
    previous_depth_mspe = best_mspe
  end

  return problem.result
end

function solve(prob::SyntheticControlProblem{T}) where {T}
  return run_outer_search!(prob)
end


end # module SyntheticControl
