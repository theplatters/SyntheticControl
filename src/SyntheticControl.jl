module SyntheticControl

using CommonSolve, ForwardDiff, LinearAlgebra, Optimization, OptimizationOptimJL, OptimizationBBO
using ADTypes, DifferentiationInterface
using Statistics
export SyntheticControlProblem, SyntheticControlResult, SyntheticControlSolver
export solve
"""
    SyntheticControlProblem{T<:AbstractFloat}

Holds the cleanly aligned matrices and vectors required to solve a 
Synthetic Control Method optimization problem.
"""
struct SyntheticControlProblem{T<:AbstractFloat}
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
  function SyntheticControlProblem(
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


struct SyntheticControlResult{T<:AbstractFloat}
  problem::SyntheticControlProblem{T}
  W::Vector{T}                         # Optimal unit weights (J x 1)
  V::Vector{T}                         # Optimal predictor weights (diagonal elements, K x 1)
  mspe::T                              # Final pre-treatment Mean Squared Prediction Error

  function SyntheticControlResult(prob, W, V, mspe)
    length(W) == length(prob.donor_ids) || throw(DimensionMismatch("Weight vector W must match donor pool size"))
    return new{eltype(W)}(prob, W, V, mspe)
  end
end

struct SyntheticControlSolver{T<:AbstractFloat}
  prob::SyntheticControlProblem{T}
  V::Vector{T}                  # Holds the current outer-loop optimal V (diagonal elements K x 1)
  W::Vector{T}                  # Holds the current inner-loop optimal W (J x 1)

  function SyntheticControlSolver(prob::SyntheticControlProblem{T}) where {T}
    K, J = size(prob.X0)
    T0 = length(prob.Y1)

    # Initialize W evenly across the J donor units
    #
    V = fill(1.0 / K, K)
    W = fill(1.0 / J, J)

    return new{T}(prob, V, W)
  end
end

function weight_squared_distance(X0, X1, V, W)
  val = zero(eltype(W))
  for k in eachindex(X1)
    res_k = X1[k]
    for j in eachindex(W)
      @inbounds res_k -= X0[k, j] * W[j]
    end
    val += res_k * res_k * V[k]
  end
  return val
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


function cons(res, x, _p)
  res[1] = sum(x)
  return nothing
end

inner_objective(u, p) = weight_squared_distance(p.X0, p.X1, p.V, u)


function objective(u, p::SyntheticControlSolver{T}, opt_func) where {T}
  prob = p.prob
  _K, J = size(prob.X0)
  lower_bounds = zeros(T, J)
  upper_bounds = ones(T, J)


  u0 = ones(T, J) / J
  params = (; X1=prob.X1_normalized, X0=prob.X0_normalized, V=u)
  opt_prob = OptimizationProblem(opt_func, u0, params, lcons=[one(T)], ucons=[one(T)], lb=lower_bounds, ub=upper_bounds)

  sol = solve(opt_prob, IPNewton())
  if !SciMLBase.successful_retcode(sol)
    return Inf
  end
  p.W .= sol.u


  return calculate_mspe(p.prob.Y1, p.prob.Y0, p.W)
end


function regression_based_start(prob::SyntheticControlProblem{T}) where {T}
  Xall = hcat(prob.X1_normalized, prob.X0_normalized) # K × (J+1)
  Xreg = hcat(ones(T, size(Xall, 2)), Xall')          # (J+1) × (K+1)
  Zall = hcat(prob.Y1, prob.Y0)                       # T0 × (J+1)

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

function CommonSolve.solve(prob::SyntheticControlProblem{T}) where {T}
  solver = SyntheticControlSolver(prob)

  K, _J = size(prob.X0)

  lower_bounds = zeros(T, K)
  upper_bounds = ones(T, K)

  opt_func_inner = OptimizationFunction(
    inner_objective,
    DifferentiationInterface.SecondOrder(
      ADTypes.AutoForwardDiff(),
      ADTypes.AutoForwardDiff()
    ),
    cons=cons
  )

  opt_func = OptimizationFunction((u, p) -> objective(u, p, opt_func_inner))

  function run_outer(u0)
    solver.W .= fill(one(T) / length(solver.W), length(solver.W))
    opt_prob = OptimizationProblem(
      opt_func,
      u0,
      solver;
      lb=lower_bounds,
      ub=upper_bounds
    )
    sol = Optimization.solve(opt_prob, BBO_adaptive_de_rand_1_bin_radiuslimited())

    return (
      sol=sol,
      V=copy(sol.u),
      W=copy(solver.W),
      mspe=T(sol.objective),
    )
  end

  # Start 1: equal V weights
  start_equal = fill(one(T) / K, K)
  best = run_outer(start_equal)

  # Start 2: regression-based V weights, as in Synth::synth()
  start_reg = regression_based_start(prob)

  if start_reg !== nothing
    candidate = run_outer(start_reg)

    if candidate.mspe < best.mspe
      best = candidate
    end
  end

  return SyntheticControlResult(prob, best.W, best.V, best.mspe)
end


end # module SyntheticControl
