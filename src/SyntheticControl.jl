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

Holds the cleanly aligned matrices and vectors used by synthetic control
estimators. Solver-specific problem types, such as the classic
`SyntheticControlProblem`, wrap this shared data interface.
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

include("classic_scm.jl")


end # module SyntheticControl
