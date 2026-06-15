module SyntheticControl

using CommonSolve, ForwardDiff, LinearAlgebra, Optimization, OptimizationOptimJL, OptimizationBBO
using ADTypes, DifferentiationInterface
export SyntheticControlProblem, SyntheticControlResult, SyntheticControlSolver
export solve
"""
    SyntheticControlProblem{T<:AbstractFloat}

Holds the cleanly aligned matrices and vectors required to solve a 
Synthetic Control Method optimization problem.
"""
struct SyntheticControlProblem{T <: AbstractFloat}
    # 1. Target Data (Treated Unit)
    X1::Vector{T}      # K x 1 vector of pre-treatment predictors
    Y1::Vector{T}      # T_pre x 1 vector of pre-treatment outcomes

    # 2. Donor Pool Data (Control Units)
    X0::Matrix{T}      # K x J matrix of predictors for J control units
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
        ) where {T <: AbstractFloat}

        K, J = size(X0)
        T_pre = length(Y1)

        length(X1) == K  || throw(DimensionMismatch("X1 length must match rows of X0 (K)"))
        size(Y0, 1) == T_pre || throw(DimensionMismatch("Y0 rows must match length of Y1 (T_pre)"))
        size(Y0, 2) == J     || throw(DimensionMismatch("Y0 columns must match columns of X0 (J)"))
        length(predictor_names) == K || throw(DimensionMismatch("Must provide K predictor names"))
        length(donor_ids) == J       || throw(DimensionMismatch("Must provide J donor IDs"))

        return new{T}(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)
    end
end


struct SyntheticControlResult{T <: AbstractFloat}
    problem::SyntheticControlProblem{T}  # Link back to the original data
    W::Vector{T}                         # Optimal unit weights (J x 1)
    V::Vector{T}                         # Optimal predictor weights (diagonal elements, K x 1)
    mspe::T                              # Final pre-treatment Mean Squared Prediction Error

    # Convenient helper for evaluating post-treatment or plotting
    function SyntheticControlResult(prob, W, V, mspe)
        # Ensure the output weights map back perfectly to the donor pool size
        length(W) == length(prob.donor_ids) || throw(DimensionMismatch("Weight vector W must match donor pool size"))
        return new{eltype(W)}(prob, W, V, mspe)
    end
end

struct SyntheticControlSolver{T <: AbstractFloat}
    prob::SyntheticControlProblem{T}
    V::Vector{T}                  # Holds the current outer-loop optimal V (diagonal elements K x 1)
    W::Vector{T}                  # Holds the current inner-loop optimal W (J x 1)
    norm_cache::Vector{T}         # Cache for predictor matching residual (K x 1)
    mspe_cache::Vector{T}         # Cache for outcome tracking residual (T_pre x 1)

    function SyntheticControlSolver(prob::SyntheticControlProblem{T}) where {T}
        K, J = size(prob.X0)
        T0 = length(prob.Y1)

        # Initialize W evenly across the J donor units
        #
        V = fill(1.0 / K, K)
        W = fill(1.0 / J, J)
        norm_cache = zeros(T, K)
        mspe_cache = zeros(T, T0)

        return new{T}(prob, V, W, norm_cache, mspe_cache)
    end
end

function weight_squared_distance(X0, X1, V, W)
    residual = (X1 - X0 * W)
    return dot(residual, V, residual)
end


function calculate_mspe!(cache, Y1, Y0, W)
    cache .= Y1
    mul!(cache, Y0, W, -1.0, 1.0)
    return dot(cache, cache) / length(cache)
end
function calculate_mspe(Y1, Y0, W)
    residual = Y1 - Y0 * W
    return dot(residual, residual) / length(residual)
end


cons(res, x, _p) = (res .= [sum(x)])

inner_objective(u, p) = weight_squared_distance(p.X0, p.X1, Diagonal(p.V), u)


function objective(u, p::SyntheticControlSolver{T}, opt_func) where {T}
    prob = p.prob
    _K, J = size(prob.X0)
    lower_bounds = zeros(T, J)
    upper_bounds = ones(T, J)


    u0 = ones(T, J) / J
    params = (; X1 = prob.X1, X0 = prob.X0, V = u)
    opt_prob = OptimizationProblem(opt_func, u0, params, lcons = [one(T)], ucons = [one(T)], lb = lower_bounds, ub = upper_bounds)


    sol = solve(opt_prob, IPNewton())
    p.W .= sol.u


    return calculate_mspe(p.prob.Y1, p.prob.Y0, p.W)
end


function CommonSolve.solve(prob::SyntheticControlProblem{T}) where {T}
    solver = SyntheticControlSolver(prob)

    K, J = size(prob.X0)
    u0 = fill(1.0 / K, K)

    lower_bounds = zeros(T, K)
    upper_bounds = ones(T, K)

    opt_func_inner = OptimizationFunction(inner_objective, DifferentiationInterface.SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoForwardDiff()), cons = cons)
    opt_func = OptimizationFunction((u, p) -> objective(u, p, opt_func_inner))
    opt_prob = OptimizationProblem(opt_func, u0, solver, lb = lower_bounds, ub = upper_bounds)

    sol = Optimization.solve(opt_prob, BBO_adaptive_de_rand_1_bin_radiuslimited())

    # Extract the final optimized results
    optimal_V = sol.u
    optimal_W = solver.W
    final_mspe = sol.objective

    return SyntheticControlResult(prob, optimal_W, optimal_V, final_mspe)
end


end # module SyntheticControl
