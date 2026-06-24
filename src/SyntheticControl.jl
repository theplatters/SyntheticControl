module SyntheticControl

using CommonSolve, LinearAlgebra
import CommonSolve: solve
using Base.Threads: @threads, nthreads
using Statistics
export SyntheticControlData, SyntheticControlProblem, SyntheticControlResult
export solve
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

    std_divisor = vec(std(hcat(X0, X1), dims=2))
    any(<(eps(T)), std_divisor) && throw(ArgumentError("Predictors have almost zero variance"))
    return new{T}(X1, X1 ./ std_divisor, Y1, X0, X0 ./ std_divisor, Y0, predictor_names, donor_ids, treated_id)
  end
end

mutable struct InnerProblemCache{T<:AbstractFloat}
  data::SyntheticControlData{T}
  V::Vector{T}
  W::Vector{T}
  outer_v::Vector{T}
  Q::Matrix{T}
  c::Vector{T}
  active::Vector{Bool}
  active_idx::Vector{Int}
  kkt::Matrix{T}
  rhs::Vector{T}
  solution::Vector{T}
  ipiv::Vector{Vector{LinearAlgebra.BlasInt}}
  local_v::Vector{T}
  candidate::Vector{T}
  best_candidate::Vector{T}
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
  caches::Vector{InnerProblemCache{T}}
  result_V::Matrix{T}
  result_W::Matrix{T}
  result_mspe::Vector{T}
  best_V::Vector{T}
  best_W::Vector{T}
  result::SyntheticControlResult{T}

  function SyntheticControlProblem(data::SyntheticControlData{T}) where {T}
    K, J = size(data.X0)
    starts = make_outer_starts(data)
    nstarts = size(starts, 2)
    caches = [InnerProblemCache(data) for _ in 1:nstarts]
    result_V = Matrix{T}(undef, K, nstarts)
    result_W = Matrix{T}(undef, J, nstarts)
    result_mspe = Vector{T}(undef, nstarts)
    best_V = fill(one(T) / T(K), K)
    best_W = fill(one(T) / T(J), J)
    result = SyntheticControlResult(data, best_W, best_V, typemax(T))
    return new{T}(data, starts, nstarts, caches, result_V, result_W, result_mspe, best_V, best_W, result)
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
  predictor_names::Vector{String}, donor_ids::Vector{String}, treated_id::String
) where {T<:AbstractFloat}
  return SyntheticControlProblem(SyntheticControlData(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id))
end

InnerProblemCache(problem::SyntheticControlProblem{T}) where {T} = InnerProblemCache(problem.data)

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


function InnerProblemCache(data::SyntheticControlData{T}) where {T}
  K, J = size(data.X0)
  V = fill(one(T) / K, K)
  W = fill(one(T) / J, J)

  return InnerProblemCache(
    data,
    V,
    W,
    similar(V),
    zeros(T, J, J),
    zeros(T, J),
    fill(true, J),
    collect(1:J),
    zeros(T, J + 1, J + 1),
    zeros(T, J + 1),
    zeros(T, J + 1),
    [Vector{LinearAlgebra.BlasInt}(undef, n) for n in 1:(J + 1)],
    Vector{T}(undef, K),
    Vector{T}(undef, K),
    Vector{T}(undef, K)
  )
end

function normalize_v!(dest::AbstractVector{T}, src::AbstractVector) where {T}
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

function normalize_v_column!(dest::AbstractVector{T}, src::AbstractMatrix{T}, col::Int) where {T}
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

function copyto_column!(dest::AbstractMatrix{T}, col::Int, src::AbstractVector{T}) where {T}
  @inbounds for i in eachindex(src)
    dest[i, col] = src[i]
  end
  return dest
end

function copyfrom_column!(dest::AbstractVector{T}, src::AbstractMatrix{T}, col::Int) where {T}
  @inbounds for i in eachindex(dest)
    dest[i] = src[i, col]
  end
  return dest
end

function update_quadratic_terms!(cache::InnerProblemCache{T}) where {T}
  X0 = cache.data.X0_normalized
  X1 = cache.data.X1_normalized
  V = cache.V
  Q = cache.Q
  c = cache.c
  K, J = size(X0)

  fill!(Q, zero(T))
  fill!(c, zero(T))

  @inbounds for j in 1:J
    c_j = zero(T)
    for k in 1:K
      x0_j = X0[k, j]
      weighted_x0_j = V[k] * x0_j
      c_j += weighted_x0_j * X1[k]

      for l in j:J
        Q[j, l] += weighted_x0_j * X0[k, l]
      end
    end
    c[j] = c_j
  end

  @inbounds for j in 2:J
    for l in 1:(j-1)
      Q[j, l] = Q[l, j]
    end
  end

  return nothing
end

function solve_active_equality!(cache::InnerProblemCache{T}, nactive::Int) where {T}
  Q = cache.Q
  c = cache.c
  idx = cache.active_idx
  kkt = cache.kkt
  rhs = cache.rhs
  solution = cache.solution
  ridge = sqrt(eps(T))

  fill!(kkt, zero(T))
  fill!(rhs, zero(T))

  @inbounds for a in 1:nactive
    ja = idx[a]
    rhs[a] = 2 * c[ja]
    kkt[a, nactive+1] = one(T)
    kkt[nactive+1, a] = one(T)

    for b in 1:nactive
      jb = idx[b]
      kkt[a, b] = 2 * Q[ja, jb]
    end
    kkt[a, a] += ridge
  end
  rhs[nactive+1] = one(T)

  active_system = @views kkt[1:(nactive+1), 1:(nactive+1)]
  active_solution = @views solution[1:(nactive+1)]
  @inbounds for i in 1:(nactive+1)
    active_solution[i] = rhs[i]
  end
  ipiv = cache.ipiv[nactive+1]
  LinearAlgebra.LAPACK.getrf!(active_system, ipiv)
  LinearAlgebra.LAPACK.getrs!('N', active_system, ipiv, active_solution)

  return solution[nactive+1]
end

function active_indices!(idx::Vector{Int}, active::Vector{Bool})
  nactive = 0

  @inbounds for j in eachindex(active)
    if active[j]
      nactive += 1
      idx[nactive] = j
    end
  end

  return nactive
end

function renormalize_simplex!(W::Vector{T}) where {T}
  s = zero(T)

  @inbounds for j in eachindex(W)
    if W[j] < zero(T)
      W[j] = zero(T)
    end
    s += W[j]
  end

  if s <= eps(T)
    fill!(W, inv(T(length(W))))
  else
    inv_s = inv(s)
    @inbounds for j in eachindex(W)
      W[j] *= inv_s
    end
  end

  return W
end

function solve_inner_weights!(cache::InnerProblemCache{T}) where {T}
  J = length(cache.W)
  Q = cache.Q
  c = cache.c
  W = cache.W
  active = cache.active
  idx = cache.active_idx
  solution = cache.solution
  tol = sqrt(eps(T))

  fill!(active, true)

  for _ in 1:(3J)
    nactive = active_indices!(idx, active)
    if nactive == 0
      fill!(W, inv(T(J)))
      return W
    end

    lambda = solve_active_equality!(cache, nactive)
    fill!(W, zero(T))

    min_weight = zero(T)
    drop_j = 0
    @inbounds for a in 1:nactive
      j = idx[a]
      w_j = solution[a]
      W[j] = w_j
      if w_j < min_weight
        min_weight = w_j
        drop_j = j
      end
    end

    if drop_j != 0 && min_weight < -tol
      active[drop_j] = false
      continue
    end

    add_j = 0
    min_multiplier = zero(T)
    @inbounds for j in 1:J
      if !active[j]
        multiplier = lambda - 2 * c[j]
        for a in 1:nactive
          multiplier += 2 * Q[j, idx[a]] * W[idx[a]]
        end

        if multiplier < min_multiplier
          min_multiplier = multiplier
          add_j = j
        end
      end
    end

    if add_j != 0 && min_multiplier < -tol
      active[add_j] = true
      continue
    end

    return renormalize_simplex!(W)
  end

  return renormalize_simplex!(W)
end

function solve_inner!(cache::InnerProblemCache{T}, raw_v) where {T}
  normalize_v!(cache.V, raw_v)
  update_quadratic_terms!(cache)
  solve_inner_weights!(cache)

  return calculate_mspe(cache.data.Y1, cache.data.Y0, cache.W)
end

function regression_based_start(data::SyntheticControlData{T}) where {T}
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

function push_unique_start!(starts::Vector{Vector{T}}, raw_v, scratch::Vector{T}) where {T}
  normalize_v!(scratch, raw_v)
  threshold = T(10) * eps(T)

  for start in starts
    same = true
    @inbounds for i in eachindex(start, scratch)
      if abs(start[i] - scratch[i]) > threshold
        same = false
        break
      end
    end
    same && return starts
  end

  push!(starts, copy(scratch))
  return starts
end

function make_outer_starts(data::SyntheticControlData{T}) where {T}
  K = length(data.X1)
  starts = Vector{Vector{T}}()
  scratch = Vector{T}(undef, K)

  push_unique_start!(starts, fill(one(T) / T(K), K), scratch)

  start_reg = regression_based_start(data)
  if start_reg !== nothing
    push_unique_start!(starts, start_reg, scratch)
  end

  corner = zeros(T, K)
  for i in 1:K
    fill!(corner, zero(T))
    corner[i] = one(T)
    push_unique_start!(starts, corner, scratch)
  end

  if K <= 20
    pair = zeros(T, K)
    half = one(T) / T(2)
    for i in 1:(K-1)
      for j in (i+1):K
        fill!(pair, zero(T))
        pair[i] = half
        pair[j] = half
        push_unique_start!(starts, pair, scratch)
      end
    end
  end

  starts_matrix = Matrix{T}(undef, K, length(starts))
  for s in eachindex(starts)
    copyto!(@view(starts_matrix[:, s]), starts[s])
  end

  return starts_matrix
end

function local_outer_search!(
  cache::InnerProblemCache{T},
  starts::AbstractMatrix{T},
  start_idx::Int,
  result_V::AbstractMatrix{T},
  result_W::AbstractMatrix{T};
  max_iters::Int=200,
  initial_step::T=T(0.25),
  min_step::T=T(1.0e-4),
  shrink::T=T(0.5),
  tol::T=sqrt(eps(T))
) where {T}
  K = length(cache.V)
  v = cache.local_v
  candidate = cache.candidate
  best_candidate = cache.best_candidate

  normalize_v_column!(v, starts, start_idx)
  mspe = solve_inner!(cache, v)
  step = initial_step

  for _ in 1:max_iters
    step < min_step && break

    best_neighbor_mspe = mspe
    found_improvement = false

    @inbounds for i in 1:K
      for j in 1:K
        i == j && continue
        delta = min(step, v[j])
        delta <= eps(T) && continue

        copyto!(candidate, v)
        candidate[i] += delta
        candidate[j] -= delta

        candidate_mspe = solve_inner!(cache, candidate)
        if candidate_mspe + tol < best_neighbor_mspe
          best_neighbor_mspe = candidate_mspe
          copyto!(best_candidate, cache.V)
          found_improvement = true
        end
      end
    end

    if found_improvement
      copyto!(v, best_candidate)
      mspe = solve_inner!(cache, v)
    else
      step *= shrink
    end
  end

  mspe = solve_inner!(cache, v)
  copyto_column!(result_V, start_idx, cache.V)
  copyto_column!(result_W, start_idx, cache.W)
  return mspe
end

function threaded_outer_search!(problem::SyntheticControlProblem{T}) where {T}
  starts = problem.starts
  result_V = problem.result_V
  result_W = problem.result_W
  result_mspe = problem.result_mspe

  if nthreads() == 1
    for s in 1:problem.nstarts
      cache = problem.caches[s]
      result_mspe[s] = local_outer_search!(
        cache,
        starts,
        s,
        result_V,
        result_W
      )
    end
  else
    @threads for s in 1:problem.nstarts
      cache = problem.caches[s]
      result_mspe[s] = local_outer_search!(
        cache,
        starts,
        s,
        result_V,
        result_W
      )
    end
  end

  best_idx = 1
  best_mspe = result_mspe[1]
  @inbounds for s in 2:problem.nstarts
    candidate_mspe = result_mspe[s]
    if candidate_mspe < best_mspe
      best_idx = s
      best_mspe = candidate_mspe
    end
  end

  copyfrom_column!(problem.best_V, result_V, best_idx)
  copyfrom_column!(problem.best_W, result_W, best_idx)
  problem.result.mspe = best_mspe
  return problem.result
end

function solve(prob::SyntheticControlProblem{T}) where {T}
  return threaded_outer_search!(prob)
end


end # module SyntheticControl
