"""
    QuadraticModelCache{T}

Mutable storage for the donor-weight quadratic program used inside the
classic and penalized solvers. `gram` stores the symmetric quadratic term and
`linear` stores the linear term, both with element type `T`.

This is an implementation detail. It is allocated by `InnerWeightCache` and
mutated by objective update functions. Callers must keep `gram` square with
one row and column per donor and `linear` with one entry per donor.

# Examples

```julia
cache = SyntheticControl.QuadraticModelCache(zeros(2, 2), zeros(2))
cache.gram[1, 1] = 1.0
cache.linear[1] = 0.5
```
"""
mutable struct QuadraticModelCache{T<:AbstractFloat}
  gram::Matrix{T}
  linear::Vector{T}
end

"""
    ActiveSetCache{T}

Scratch space for the active-set donor-weight optimizer. The cache tracks
which donor weights are currently active, their indices, a KKT matrix, a
right-hand side, a solution vector, and LAPACK pivot buffers.

This type is internal and is mutated by `optimize_donor_weights!` and
`solve_active_set_kkt!`. Buffers must be sized for the number of donors used
by the associated `InnerWeightCache`.

# Examples

```julia
cache = SyntheticControl.ActiveSetCache(
  trues(2), collect(1:2), zeros(3, 3), zeros(3), zeros(3),
  [Vector{LinearAlgebra.BlasInt}(undef, n) for n in 1:3],
)
length(cache.indices) == 2
```
"""
mutable struct ActiveSetCache{T<:AbstractFloat}
  is_active::Vector{Bool}
  indices::Vector{Int}
  kkt::Matrix{T}
  rhs::Vector{T}
  solution::Vector{T}
  pivots::Vector{Vector{LinearAlgebra.BlasInt}}
end

"""
    PredictorSearchWorkspace{T}

Mutable buffers used while moving mass between predictor weights during the
outer classic SCM search. `current`, `candidate`, and `best_candidate` store
simplex-valued predictor weights, while `transfer_donors` and
`transfer_receivers` store candidate coordinate indices.

This is an internal workspace with no validation in its default constructor.
It is normally created by `InnerWeightCache(data)`.

# Examples

```julia
workspace = SyntheticControl.PredictorSearchWorkspace(
  zeros(2), zeros(2), zeros(2), Vector{Int}(undef, 2), Vector{Int}(undef, 2),
)
workspace.current .= [0.5, 0.5]
```
"""
mutable struct PredictorSearchWorkspace{T<:AbstractFloat}
  current::Vector{T}
  candidate::Vector{T}
  best_candidate::Vector{T}
  transfer_donors::Vector{Int}
  transfer_receivers::Vector{Int}
end

"""
    InnerWeightCache(data::SyntheticControlData)
    InnerWeightCache(problem::SyntheticControlProblem)
    InnerWeightCache{T}

Preallocated state for solving the inner donor-weight problem at a fixed set
of predictor weights. The cache owns normalized predictor weights, donor
weights, the quadratic objective cache, the active-set cache, and the outer
predictor-search workspace.

The constructors allocate buffers sized from `data` or `problem.data` and
return an `InnerWeightCache{T}`. The cache is mutated by objective-update and
optimization routines and should not be shared across concurrent solves.

# Examples

```julia
data = SyntheticControlData(
  [1.0, 2.0], [1.0, 1.5],
  [0.5 1.5; 1.5 2.5], [0.8 1.2; 1.3 1.7],
  ["p1", "p2"], ["d1", "d2"], "treated",
)
cache = SyntheticControl.InnerWeightCache(data)
sum(cache.donor_weights) ≈ 1.0
```
"""
mutable struct InnerWeightCache{T<:AbstractFloat}
  data::SyntheticControlData{T}
  predictor_weights::Vector{T}
  donor_weights::Vector{T}
  quadratic::QuadraticModelCache{T}
  active_set::ActiveSetCache{T}
  predictor_search::PredictorSearchWorkspace{T}
end

"""
    OuterSearchResults{T}

Mutable matrices and vectors holding one row of search output per predictor
start. `predictor_weights` is `K × nstarts`, `donor_weights` is `J × nstarts`,
and `mspe` has length `nstarts`.

This is internal storage for `SyntheticControlProblem`. It has no side effects
other than being mutated by `optimize_predictor_weights_from_start!` and
`run_search_depth!`.

# Examples

```julia
results = SyntheticControl.OuterSearchResults(zeros(2, 3), zeros(4, 3), zeros(3))
size(results.predictor_weights) == (2, 3)
```
"""
mutable struct OuterSearchResults{T<:AbstractFloat}
  predictor_weights::Matrix{T}
  donor_weights::Matrix{T}
  mspe::Vector{T}
end

"""
    SearchDepth{T}

Configuration for one pass of the classic SCM outer search. Fields control
maximum coordinate-transfer iterations, objective evaluations, donor and
receiver coordinate counts, and the minimum transferable predictor weight.

Instances are immutable and returned by `search_depths`. Values must be
non-negative for meaningful search behavior; constructors do not validate
them.

# Examples

```julia
depth = SyntheticControl.SearchDepth{Float64}(10, 100, 2, 3, 1e-6)
depth.max_evaluations == 100
```
"""
struct SearchDepth{T<:AbstractFloat}
  max_iters::Int
  max_evaluations::Int
  max_transfer_donors::Int
  max_transfer_receivers::Int
  min_transfer_weight::T
end

"""
    SyntheticControlResult(data, W, V, mspe)
    SyntheticControlResult(problem, W, V, mspe)
    SyntheticControlResult{T}

Result from the classic synthetic-control solver. `W` contains donor weights
with one entry per donor, `V` contains predictor weights with one entry per
predictor, and `mspe` is the pre-treatment mean squared prediction error.

The constructor returns a `SyntheticControlResult{T}` and throws
`DimensionMismatch` if `W` or `V` does not match the donor or predictor count.
It stores the supplied vectors without copying, so later mutation of those
vectors is reflected in the result.

# Examples

```julia
data = SyntheticControlData(
  [1.0, 2.0], [1.0, 1.5],
  [0.5 1.5; 1.5 2.5], [0.8 1.2; 1.3 1.7],
  ["p1", "p2"], ["d1", "d2"], "treated",
)
result = SyntheticControl.SyntheticControlResult(data, [0.4, 0.6], [0.7, 0.3], 0.01)
result.mspe
```
"""
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

"""
    SyntheticControlProblem(data; max_pair_starts=0, target_mspe=0, min_relative_mspe_improvement=0.01)
    SyntheticControlProblem(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id; kwargs...)
    SyntheticControlProblem{T}

Cached problem definition for the classic synthetic-control estimator. The
problem stores `SyntheticControlData`, predictor-weight starts, per-start
inner caches, search results, best weights, stopping thresholds, and the
mutable result object returned by `solve`.

`max_pair_starts` adds two-predictor starting points to the default uniform,
regression, and corner starts. `target_mspe` stops the outer search once the
best MSPE is at or below the target. `min_relative_mspe_improvement` stops
later search-depth passes when improvement is small. These keyword arguments
must be non-negative.

The matrix constructor first creates `SyntheticControlData` and may throw the
same validation errors. `Base.getproperty` forwards data fields such as `X0`
or `donor_ids` to `problem.data`.

# Examples

```julia
using SyntheticControl, CommonSolve

problem = SyntheticControlProblem(
  [1.0, 2.0],
  [1.0, 1.5, 2.0],
  [0.5 1.5 2.5; 1.5 2.5 3.5],
  [0.8 1.2 1.8; 1.3 1.7 2.3; 1.8 2.2 2.8],
  ["level", "trend"],
  ["A", "B", "C"],
  "treated",
)
result = solve(problem)
sum(result.W) ≈ 1.0
```
"""
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

"""
    Base.getproperty(problem::SyntheticControlProblem, name::Symbol)

Return a field from `problem` or, when `name` is a field of
`SyntheticControlData`, return the corresponding field from `problem.data`.
This enables `problem.X0`, `problem.Y1`, and similar conveniences.

Throws the usual `type has no field` error for unknown names. It has no side
effects.

# Examples

```julia
data = SyntheticControlData([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                            ["p"], ["d1", "d2"], "treated")
problem = SyntheticControlProblem(data)
problem.X0 === data.X0
```
"""
function Base.getproperty(problem::SyntheticControlProblem, name::Symbol)
  if name in fieldnames(SyntheticControlData)
    return getproperty(getfield(problem, :data), name)
  end
  return getfield(problem, name)
end

"""
    SyntheticControlProblem(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id; kwargs...)

Construct a `SyntheticControlProblem` directly from aligned treated and donor
matrices. Arguments and keyword constraints are the same as
`SyntheticControlData` and `SyntheticControlProblem(data; kwargs...)`.

Returns a cached classic SCM problem. It may throw `DimensionMismatch` or
`ArgumentError` during data validation or keyword validation.

# Examples

```julia
problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
length(problem.donor_ids) == 2
```
"""
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

"""
    InnerWeightCache(problem::SyntheticControlProblem)

Allocate an `InnerWeightCache` using `problem.data`. The returned cache is
independent of the caches already stored inside `problem` and may be mutated
by inner-solver routines.

# Examples

```julia
problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
cache = SyntheticControl.InnerWeightCache(problem)
cache.data === problem.data
```
"""
InnerWeightCache(problem::SyntheticControlProblem{T}) where {T} = InnerWeightCache(problem.data)

"""
    SyntheticControlResult(problem, W, V, mspe)

Create a classic SCM result attached to `problem.data`. `W` must have one
entry per donor, `V` must have one entry per predictor, and `mspe` must have
the same floating-point element type as the problem.

Returns `SyntheticControlResult(problem.data, W, V, mspe)`. It may throw
`DimensionMismatch` for incompatible weight lengths and stores vectors without
copying.

# Examples

```julia
problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
result = SyntheticControl.SyntheticControlResult(problem, [0.5, 0.5], [1.0], 0.0)
result.data === problem.data
```
"""
function SyntheticControlResult(problem::SyntheticControlProblem{T}, W::Vector{T}, V::Vector{T}, mspe::T) where {T}
  return SyntheticControlResult(problem.data, W, V, mspe)
end

"""
    weight_squared_distance(X0, X1, V, W)

Compute `(X1 - X0 * W)' * Diagonal(V) * (X1 - X0 * W)` without allocating the
residual vector. `X0` must have one row per predictor and one column per donor,
`X1` and `V` must have one entry per predictor, and `W` must have one entry
per donor.

Returns a scalar with the element type of `W`. The method assumes compatible
dimensions and may throw bounds errors if inputs are inconsistent. It has no
side effects.

# Examples

```julia
X0 = [2.0 3.0; 4.0 5.0]
X1 = [1.0, 2.0]
V = [2.0, 0.5]
W = [0.3, 0.7]
SyntheticControl.weight_squared_distance(X0, X1, V, W) ≈ 9.425
```
"""
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


"""
    calculate_mspe(Y1, Y0, W)

Compute the mean squared prediction error between treated outcomes `Y1` and
the donor-weighted synthetic path `Y0 * W`. `Y0` must have one row per
pre-treatment period and one column per donor, and `W` must have one entry per
donor.

Returns `sum(abs2, Y1 - Y0 * W) / length(Y1)` without allocating the fitted
path. The method assumes compatible dimensions and may throw bounds errors for
inconsistent inputs. It has no side effects.

# Examples

```julia
Y1 = [10.0, 20.0]
Y0 = [5.0 15.0; 10.0 25.0]
W = [0.3, 0.7]
SyntheticControl.calculate_mspe(Y1, Y0, W) ≈ 2.125
```
"""
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


"""
    InnerWeightCache(data::SyntheticControlData)

Allocate all buffers needed by the inner donor-weight optimizer for `data`.
Initial predictor and donor weights are uniform simplexes.

Returns an `InnerWeightCache{T}`. It allocates memory but does not mutate
`data` and does not solve an optimization problem.

# Examples

```julia
data = SyntheticControlData([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                            ["p"], ["d1", "d2"], "treated")
cache = SyntheticControl.InnerWeightCache(data)
cache.predictor_weights == [1.0]
```
"""
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

"""
    normalize_weights!(dest, src)

Write a non-negative simplex normalization of `src` into `dest`. Each entry
is converted to the element type of `dest`, replaced by its absolute value,
and divided by the absolute sum.

Returns `dest`. If the absolute sum is non-finite or not larger than
`eps(eltype(dest))`, `dest` is filled uniformly. The function mutates only
`dest` and requires `dest` and `src` to share compatible indices.

# Examples

```julia
dest = zeros(3)
SyntheticControl.normalize_weights!(dest, [-2.0, 1.0, 0.0])
dest ≈ [2 / 3, 1 / 3, 0]
```
"""
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

"""
    normalize_weight_column!(dest, src, col)

Normalize column `col` of matrix `src` into vector `dest` as a non-negative
simplex. The absolute values in the selected column are used.

Returns `dest`. If the column sum is non-finite or too small, `dest` is
filled uniformly. Mutates `dest`; `src` is read-only. `col` must be a valid
column index.

# Examples

```julia
dest = zeros(2)
starts = [1.0 -2.0; 1.0 6.0]
SyntheticControl.normalize_weight_column!(dest, starts, 2)
dest ≈ [0.25, 0.75]
```
"""
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

"""
    write_column!(dest, col, src)

Copy vector `src` into column `col` of matrix `dest`.

Returns `dest`. The function mutates `dest`, assumes compatible dimensions,
and may throw a bounds error if `col` or vector indices are invalid.

# Examples

```julia
mat = zeros(2, 2)
SyntheticControl.write_column!(mat, 2, [0.25, 0.75])
mat[:, 2] == [0.25, 0.75]
```
"""
function write_column!(dest::AbstractMatrix{T}, col::Int, src::AbstractVector{T}) where {T}
  @inbounds for i in eachindex(src)
    dest[i, col] = src[i]
  end
  return dest
end

"""
    read_column!(dest, src, col)

Copy column `col` of matrix `src` into vector `dest`.

Returns `dest`. The function mutates `dest`, assumes compatible dimensions,
and may throw a bounds error if `col` or vector indices are invalid.

# Examples

```julia
dest = zeros(2)
SyntheticControl.read_column!(dest, [1.0 3.0; 2.0 4.0], 2)
dest == [3.0, 4.0]
```
"""
function read_column!(dest::AbstractVector{T}, src::AbstractMatrix{T}, col::Int) where {T}
  @inbounds for i in eachindex(dest)
    dest[i] = src[i, col]
  end
  return dest
end

"""
    update_donor_weight_objective!(cache)

Rebuild the quadratic and linear terms for the classic inner donor-weight
objective using `cache.data` and `cache.predictor_weights`.

Returns `nothing`. Mutates `cache.quadratic.gram` and
`cache.quadratic.linear`; all other fields are read. The cache dimensions
must match its data.

# Examples

```julia
data = SyntheticControlData([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                            ["p"], ["d1", "d2"], "treated")
cache = SyntheticControl.InnerWeightCache(data)
SyntheticControl.update_donor_weight_objective!(cache)
isfinite(cache.quadratic.gram[1, 1])
```
"""
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

"""
    solve_active_set_kkt!(cache, nactive)

Solve the KKT system for the currently active donor set in `cache`. `nactive`
is the number of valid entries in `cache.active_set.indices`.

Returns the Lagrange multiplier for the simplex equality constraint. Mutates
the active-set KKT matrix, right-hand side, solution buffer, and pivot buffer.
The quadratic objective must already have been updated.

# Examples

```julia
data = SyntheticControlData([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                            ["p"], ["d1", "d2"], "treated")
cache = SyntheticControl.InnerWeightCache(data)
SyntheticControl.update_donor_weight_objective!(cache)
nactive = SyntheticControl.collect_active_indices!(cache.active_set.indices, cache.active_set.is_active)
lambda = SyntheticControl.solve_active_set_kkt!(cache, nactive)
isfinite(lambda)
```
"""
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

"""
    collect_active_indices!(indices, is_active)

Write the indices whose corresponding `is_active` entry is `true` into the
front of `indices`.

Returns the number of active indices written. Mutates `indices` and reads
`is_active`; `indices` must be at least as long as `is_active`.

# Examples

```julia
indices = zeros(Int, 4)
n = SyntheticControl.collect_active_indices!(indices, [true, false, true, false])
n == 2 && indices[1:2] == [1, 3]
```
"""
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

"""
    project_to_simplex!(weights)

Project `weights` to the probability simplex by clipping negative entries to
zero and renormalizing the remaining mass.

Returns `weights`. Mutates `weights` in place. If the clipped sum is not
larger than `eps(eltype(weights))`, the vector is filled uniformly.

# Examples

```julia
weights = [-1.0, 2.0, 2.0]
SyntheticControl.project_to_simplex!(weights)
weights ≈ [0.0, 0.5, 0.5]
```
"""
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

"""
    optimize_donor_weights!(cache)

Solve the simplex-constrained donor-weight quadratic program represented by
`cache.quadratic`. The objective must already have been created with
`update_donor_weight_objective!` or `update_penalized_donor_weight_objective!`.

Returns `cache.donor_weights`. Mutates the donor weights and active-set
buffers. If numerical cycling prevents convergence within the internal
iteration limit, the current donor weights are projected back to the simplex.

# Examples

```julia
data = SyntheticControlData([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                            ["p"], ["d1", "d2"], "treated")
cache = SyntheticControl.InnerWeightCache(data)
SyntheticControl.update_donor_weight_objective!(cache)
w = SyntheticControl.optimize_donor_weights!(cache)
sum(w) ≈ 1.0
```
"""
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

"""
    evaluate_predictor_weights!(cache, raw_predictor_weights)

Normalize `raw_predictor_weights`, update the classic inner objective, solve
for donor weights, and compute the resulting pre-treatment MSPE.

Returns the scalar MSPE. Mutates `cache.predictor_weights`,
`cache.donor_weights`, and optimization buffers. `raw_predictor_weights` must
have one entry per predictor.

# Examples

```julia
data = SyntheticControlData([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                            ["p"], ["d1", "d2"], "treated")
cache = SyntheticControl.InnerWeightCache(data)
mspe = SyntheticControl.evaluate_predictor_weights!(cache, [1.0])
isfinite(mspe)
```
"""
function evaluate_predictor_weights!(cache::InnerWeightCache{T}, raw_predictor_weights) where {T}
  normalize_weights!(cache.predictor_weights, raw_predictor_weights)
  update_donor_weight_objective!(cache)
  optimize_donor_weights!(cache)

  return calculate_mspe(cache.data.Y1, cache.data.Y0, cache.donor_weights)
end

"""
    regression_predictor_weight_start(data)

Estimate a predictor-weight starting point by regressing pre-treatment
outcomes on standardized predictors across the treated unit and donors.

Returns a simplex vector of length `K`, or `nothing` if the regression cannot
be solved or produces an unusable diagonal. It does not mutate `data`.

# Examples

```julia
data = SyntheticControlData(
  [1.0, 2.0], [1.0, 1.5, 2.0],
  [0.5 1.5 2.5; 1.5 2.5 3.5],
  [0.8 1.2 1.8; 1.3 1.7 2.3; 1.8 2.2 2.8],
  ["level", "trend"], ["A", "B", "C"], "treated",
)
start = SyntheticControl.regression_predictor_weight_start(data)
start === nothing || sum(start) ≈ 1.0
```
"""
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

"""
    push_unique_predictor_start!(starts, raw_weights, scratch)

Normalize `raw_weights` into `scratch` and append a copy to `starts` only when
no existing start is equal within the internal tolerance.

Returns `starts`. Mutates `scratch` and possibly `starts`. `scratch` and every
stored start must have the same length as `raw_weights`.

# Examples

```julia
starts = Vector{Vector{Float64}}()
scratch = zeros(2)
SyntheticControl.push_unique_predictor_start!(starts, [1.0, 1.0], scratch)
SyntheticControl.push_unique_predictor_start!(starts, [2.0, 2.0], scratch)
length(starts) == 1
```
"""
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

"""
    predictor_pair_from_ordinal(n_predictors, ordinal)

Map a one-based ordinal over unordered predictor pairs to the corresponding
pair of one-based predictor indices.

Returns `(first_predictor, second_predictor)`. `n_predictors` should be at
least two and `ordinal` should be in `1:binomial(n_predictors, 2)`; invalid
inputs fall through to the final pair and are not separately validated.

# Examples

```julia
SyntheticControl.predictor_pair_from_ordinal(4, 3) == (1, 4)
```
"""
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

"""
    push_pair_predictor_starts!(starts, scratch, n_predictors, max_pair_starts)

Append up to `max_pair_starts` two-predictor starting weights to `starts`.
Each generated start puts half the mass on each selected predictor.

Returns `starts`. Mutates `starts` and `scratch`. `n_predictors` must be at
least two for meaningful pair starts; non-positive `max_pair_starts` appends
nothing.

# Examples

```julia
starts = Vector{Vector{Float64}}()
scratch = zeros(3)
SyntheticControl.push_pair_predictor_starts!(starts, scratch, 3, 2)
length(starts) == 2
```
"""
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

"""
    build_predictor_starts(data; max_pair_starts=0)

Build the matrix of classic SCM outer-search starting points. Starts include
the uniform simplex, a regression-derived start when available, all corner
starts, and optionally pair starts.

Returns a `K × nstarts` matrix. `max_pair_starts` must be non-negative when
called through `SyntheticControlProblem`; this helper does not validate it
separately. It allocates temporary vectors and does not mutate `data`.

# Examples

```julia
data = SyntheticControlData([1.0, 2.0], [1.0, 1.5],
                            [0.5 1.5; 1.5 2.5], [0.8 1.2; 1.3 1.7],
                            ["p1", "p2"], ["d1", "d2"], "treated")
starts = SyntheticControl.build_predictor_starts(data)
size(starts, 1) == 2
```
"""
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

"""
    select_largest_weights!(selected, weights, max_count, min_weight)

Select indices of the largest entries in `weights` that are greater than
`min_weight`, keeping at most `max_count` indices in descending weight order.

Returns the number of selected indices stored at the front of `selected`.
Mutates `selected` and reads `weights`. `selected` must be long enough for
`min(max_count, length(weights))` entries.

# Examples

```julia
selected = zeros(Int, 2)
n = SyntheticControl.select_largest_weights!(selected, [0.2, 0.7, 0.1], 2, 0.0)
n == 2 && selected == [2, 1]
```
"""
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

"""
    select_spread_indices!(selected, nitems, max_count, offset)

Choose up to `max_count` indices spread across `1:nitems`, rotated by
`offset`, and write them to `selected`.

Returns the number of indices written. Mutates `selected`. If `nitems` or
`max_count` is non-positive, returns zero without writing.

# Examples

```julia
selected = zeros(Int, 3)
n = SyntheticControl.select_spread_indices!(selected, 6, 3, 0)
n == 3 && selected == [1, 3, 5]
```
"""
function select_spread_indices!(selected::Vector{Int}, nitems::Int, max_count::Int, offset::Int)
  count = min(max_count, nitems)
  count <= 0 && return 0

  @inbounds for sample in 1:count
    base_idx = 1 + fld((sample - 1) * nitems, count)
    selected[sample] = 1 + mod(base_idx + offset - 1, nitems)
  end

  return count
end

"""
    search_depths(T, n_predictors)

Return the sequence of increasingly broad `SearchDepth{T}` configurations
used by the classic outer search for `n_predictors` predictor weights.

The returned tuple is immutable and has no side effects. `T` must be an
`AbstractFloat` type.

# Examples

```julia
depths = SyntheticControl.search_depths(Float64, 3)
length(depths) == 4
```
"""
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

"""
    optimize_predictor_weights_from_start!(cache, starts, start_idx, results; kwargs...)

Run one coordinate-transfer search over predictor weights starting from column
`start_idx` of `starts`. At each candidate predictor vector, the inner
donor-weight problem is solved using `cache`.

Returns the best MSPE found for that start. Mutates `cache`, writes the best
predictor and donor weights into `results`, and uses keyword parameters
`depth`, `initial_step`, `min_step`, `shrink`, and `tol` to control local
search. Input dimensions must match the cache and results matrices.

# Examples

```julia
problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
mspe = SyntheticControl.optimize_predictor_weights_from_start!(
  problem.inner_caches[1], problem.starts, 1, problem.search_results,
)
isfinite(mspe)
```
"""
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

"""
    run_search_depth!(problem, depth)

Evaluate every predictor start in `problem` using one `SearchDepth`. The
method uses Julia threads when more than one thread is available.

Returns `nothing`. Mutates `problem.search_results` and the per-start inner
caches. `problem.starts`, `problem.inner_caches`, and `problem.search_results`
must be internally consistent.

# Examples

```julia
problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
depth = first(SyntheticControl.search_depths(Float64, length(problem.best_predictor_weights)))
SyntheticControl.run_search_depth!(problem, depth)
all(isfinite, problem.search_results.mspe)
```
"""
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

"""
    best_search_result_index(results, nstarts)

Find the lowest MSPE among the first `nstarts` entries in `results.mspe`.

Returns `(best_idx, best_mspe)`. It has no side effects. `nstarts` must be at
least one and no larger than `length(results.mspe)`.

# Examples

```julia
results = SyntheticControl.OuterSearchResults(zeros(1, 3), zeros(2, 3), [2.0, 0.5, 1.0])
SyntheticControl.best_search_result_index(results, 3) == (2, 0.5)
```
"""
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

"""
    keep_best_search_result!(problem, candidate_idx, candidate_mspe, best_mspe)

Compare a candidate search result with the current best MSPE and copy the
candidate weights into `problem` when it improves the best value.

Returns the updated best MSPE. Mutates `problem.best_predictor_weights`,
`problem.best_donor_weights`, and `problem.result.mspe` only on improvement.
`candidate_idx` must identify a populated column in `problem.search_results`.

# Examples

```julia
problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
problem.search_results.predictor_weights[:, 1] .= [1.0]
problem.search_results.donor_weights[:, 1] .= [0.5, 0.5]
best = SyntheticControl.keep_best_search_result!(problem, 1, 0.1, 1.0)
best == 0.1
```
"""
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

"""
    target_mspe_reached(problem, mspe)

Return `true` when `problem.target_mspe` is positive and `mspe` is at or below
that target.

The function has no side effects. It is used as a stopping condition by
`run_outer_search!`.

# Examples

```julia
problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated"; target_mspe=0.5)
SyntheticControl.target_mspe_reached(problem, 0.4)
```
"""
function target_mspe_reached(problem::SyntheticControlProblem{T}, mspe::T) where {T}
  return problem.target_mspe > zero(T) && mspe <= problem.target_mspe
end

"""
    mspe_improvement_is_small(previous_mspe, current_mspe, min_relative_improvement)

Return `true` when the improvement from `previous_mspe` to `current_mspe` is
non-positive or no larger than `min_relative_improvement * abs(previous_mspe)`.

The first finite comparison is never considered small when `previous_mspe` is
not finite. The function has no side effects.

# Examples

```julia
SyntheticControl.mspe_improvement_is_small(10.0, 9.95, 0.01)
```
"""
function mspe_improvement_is_small(previous_mspe::T, current_mspe::T, min_relative_improvement::T) where {T}
  !(isfinite(previous_mspe)) && return false
  improvement = previous_mspe - current_mspe
  improvement <= zero(T) && return true
  return improvement <= min_relative_improvement * max(abs(previous_mspe), eps(T))
end

"""
    run_outer_search!(problem)

Run the full classic SCM outer search across all configured search depths.
Each depth evaluates all predictor starts, keeps the best result, and stops
early when the target MSPE or relative-improvement criterion is satisfied.

Returns `problem.result`. Mutates `problem` caches, search results, best
weights, and result MSPE. The problem should not be solved concurrently by
multiple tasks.

# Examples

```julia
problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
result = SyntheticControl.run_outer_search!(problem)
result === problem.result
```
"""
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

"""
    solve(prob::SyntheticControlProblem)

Solve a classic synthetic-control problem using the CommonSolve interface.

Returns a `SyntheticControlResult` containing donor weights `W`, predictor
weights `V`, and pre-treatment `mspe`. Mutates the problem's internal caches
and result object; repeated calls reuse those allocations. The returned
result is also associated with `prob` so fit-only orchestration APIs can
preserve its solver configuration.

# Examples

```julia
using SyntheticControl, CommonSolve

problem = SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                  ["p"], ["d1", "d2"], "treated")
result = solve(problem)
sum(result.W) ≈ 1.0
```
"""
function solve(prob::SyntheticControlProblem{T}) where {T}
  return _register_solution_problem!(run_outer_search!(prob), prob)
end
