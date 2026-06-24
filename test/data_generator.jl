module DataGenerator

using SyntheticControl
using Random
using LinearAlgebra

export generate_synthetic_data

"""
    generate_synthetic_data(; K=5, J=10, T_pre=15, noise_level=0.01, seed=42, T=Float64) -> (problem, true_W)

Generates a realistic `SyntheticControlProblem` of type `T` using a structural time-series model.
Returns a tuple `(problem, true_W)` where `true_W` is the sparse weight vector of size `J` 
used to construct the treated unit's data.
"""
function generate_synthetic_data(;
        K::Int=5,
        J::Int=10,
        T_pre::Int=15,
        noise_level::Real=0.01,
        seed::Int=42,
        T::Type{<:AbstractFloat}=Float64
    )
    rng = Random.MersenneTwister(seed)

    # 1. Generate sparse true weights W* that sum to 1
    # We choose 3 random donors to have positive weights
    true_W = zeros(T, J)
    active_indices = Random.shuffle(rng, 1:J)[1:min(3, J)]
    raw_weights = rand(rng, T, length(active_indices))
    true_W[active_indices] .= raw_weights ./ sum(raw_weights)

    # 2. Generate donor predictors X0 (K x J)
    # E.g. predictor values around some baseline
    X0 = zeros(T, K, J)
    for k in 1:K
        baseline = rand(rng, T) * 10
        X0[k, :] .= baseline .+ randn(rng, T, J) * T(1.5)
    end

    # 3. Generate donor pre-treatment outcomes Y0 (T_pre x J)
    # SCM outcomes typically have time trends, unit-specific intercepts, and unit-specific growth rates
    Y0 = zeros(T, T_pre, J)
    unit_intercepts = randn(rng, T, J) * T(10.0) .+ T(50.0)
    unit_growth = randn(rng, T, J) * T(0.5) .+ T(2.0)
    
    for j in 1:J
        for t in 1:T_pre
            common_trend = T(2.0) * t + T(5.0) * sin(t)
            noise = randn(rng, T) * T(0.5)
            Y0[t, j] = unit_intercepts[j] + unit_growth[j] * t + common_trend + noise[1]
        end
    end

    # 4. Generate treated unit predictors X1 (K) and outcomes Y1 (T_pre)
    # Treated unit is a convex combination of donors + small noise
    X1 = X0 * true_W + randn(rng, T, K) * T(noise_level)
    Y1 = Y0 * true_W + randn(rng, T, T_pre) * T(noise_level)

    # 5. Metadata
    predictor_names = ["Predictor_$i" for i in 1:K]
    donor_ids = ["Donor_$j" for j in 1:J]
    treated_id = "Treated_Unit"

    # 6. Construct problem
    prob = SyntheticControl.SyntheticControlProblem(
        X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id
    )

    return prob, true_W
end

end # module DataGenerator
