module DataGenerator

using SyntheticControl
using Random

export generate_synthetic_data

"""
    generate_synthetic_data(; K=5, J=10, T_pre=15, noise_level=0.01, seed=42, T=Float64) -> (problem, true_W)

Generates a realistic cached `SyntheticControlProblem` of type `T` using a structural time-series model.
The predictors are pre-treatment outcome summaries, so matching `X` is informative for matching
the full pre-treatment outcome path.
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

    # 2. Generate donor pre-treatment outcomes Y0 (T_pre x J)
    # SCM outcomes share common time shocks, but differ by latent level, trend, curvature, and seasonality.
    Y0 = zeros(T, T_pre, J)
    unit_intercepts = randn(rng, T, J) * T(10.0) .+ T(50.0)
    unit_growth = randn(rng, T, J) * T(0.5) .+ T(2.0)
    unit_curvature = randn(rng, T, J) * T(0.03)
    unit_seasonality = randn(rng, T, J) * T(0.4)
    
    for j in 1:J
        for t in 1:T_pre
            common_trend = T(2.0) * t + T(5.0) * sin(t)
            noise = randn(rng, T) * T(0.5)
            Y0[t, j] = unit_intercepts[j] +
                       unit_growth[j] * t +
                       unit_curvature[j] * t^2 +
                       unit_seasonality[j] * sin(T(0.5) * t) +
                       common_trend +
                       noise[1]
        end
    end

    # 3. Generate donor predictors X0 (K x J) from evenly spaced pre-treatment windows.
    # These special predictors mirror common SCM practice and make the synthetic benchmark identifiable.
    X0 = zeros(T, K, J)
    for k in 1:K
        first_t = floor(Int, (k - 1) * T_pre / K) + 1
        last_t = max(first_t, floor(Int, k * T_pre / K))
        window_length = T(last_t - first_t + 1)
        for j in 1:J
            window_sum = zero(T)
            for t in first_t:last_t
                window_sum += Y0[t, j]
            end
            X0[k, j] = window_sum / window_length
        end
    end

    # 4. Generate treated unit predictors X1 (K) and outcomes Y1 (T_pre).
    # The treated unit is a convex combination of donors plus optional measurement noise.
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
