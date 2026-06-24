using Test
using SyntheticControl
using LinearAlgebra
using CommonSolve
using Random
using Optimization
using OptimizationOptimJL
using ADTypes
using DifferentiationInterface
using ForwardDiff

# Include and use the data generator
include("data_generator.jl")
using .DataGenerator

function ipnewton_inner_solution(prob::Union{SyntheticControl.SyntheticControlData,SyntheticControl.SyntheticControlProblem}, raw_v)
    data = prob isa SyntheticControl.SyntheticControlProblem ? prob.data : prob
    _K, J = size(prob.X0)
    v = Vector{Float64}(undef, length(raw_v))
    SyntheticControl.normalize_v!(v, raw_v)

    inner_objective(u, p) = SyntheticControl.weight_squared_distance(p.X0, p.X1, p.V, u)
    cons(res, x, _p) = (res[1] = sum(x); nothing)

    opt_func = OptimizationFunction(
        inner_objective,
        DifferentiationInterface.SecondOrder(
            ADTypes.AutoForwardDiff(),
            ADTypes.AutoForwardDiff()
        ),
        cons=cons
    )
    opt_prob = OptimizationProblem(
        opt_func,
        fill(1.0 / J, J),
        (; X1=Float64.(data.X1_normalized), X0=Float64.(data.X0_normalized), V=v);
        lcons=[1.0],
        ucons=[1.0],
        lb=zeros(Float64, J),
        ub=ones(Float64, J)
    )

    sol = Optimization.solve(opt_prob, OptimizationOptimJL.IPNewton(); maxiters=1000)
    return collect(Float64, sol.u), v, Float64(sol.objective)
end

function active_set_inner_solution(prob::Union{SyntheticControl.SyntheticControlData{T},SyntheticControl.SyntheticControlProblem{T}}, raw_v) where {T}
    data = prob isa SyntheticControl.SyntheticControlProblem ? prob.data : prob
    cache = SyntheticControl.InnerProblemCache(prob)
    SyntheticControl.solve_inner!(cache, raw_v)
    loss = SyntheticControl.weight_squared_distance(
        data.X0_normalized,
        data.X1_normalized,
        cache.V,
        cache.W
    )

    return copy(cache.W), copy(cache.V), loss
end

function assert_simplex_weights(w; atol)
    @test all(isfinite, w)
    @test sum(w) ≈ 1 atol=atol
    @test minimum(w) >= -atol
end

function optimizer_fixture(X0::Matrix{T}, X1::Vector{T}) where {T}
    J = size(X0, 2)
    Y0 = Matrix{T}(I, J, J)
    Y1 = Y0 * fill(one(T) / J, J)
    predictor_names = ["P$i" for i in axes(X0, 1)]
    donor_ids = ["D$j" for j in axes(X0, 2)]
    return SyntheticControl.SyntheticControlData(X1, Y1, X0, Y0, predictor_names, donor_ids, "Treated")
end

@testset "SyntheticControl Package Tests" begin

    # 1. Compilation & Module Check
    @testset "Compilation" begin
        @test isdefined(SyntheticControl, :SyntheticControlProblem)
        @test isdefined(SyntheticControl, :SyntheticControlData)
        @test isdefined(SyntheticControl, :SyntheticControlResult)
        @test isdefined(SyntheticControl, :weight_squared_distance)
        @test isdefined(SyntheticControl, :calculate_mspe)
    end

    # 2. SyntheticControlData Construction & Validation
    @testset "SyntheticControlData Construction" begin
        # Valid inputs (Float64)
        X1 = [1.0, 2.0]
        Y1 = [10.0, 11.0, 12.0]
        X0 = [1.5 2.5; 3.5 4.5]
        Y0 = [10.5 11.5; 12.5 13.5; 14.5 15.5]
        predictor_names = ["P1", "P2"]
        donor_ids = ["D1", "D2"]
        treated_id = "T1"

        prob = SyntheticControl.SyntheticControlData(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)
        @test prob isa SyntheticControl.SyntheticControlData{Float64}
        @test prob.X1 == X1
        @test prob.Y1 == Y1
        @test prob.X0 == X0
        @test prob.Y0 == Y0
        @test prob.predictor_names == predictor_names
        @test prob.donor_ids == donor_ids
        @test prob.treated_id == treated_id

        # Valid inputs (Float32)
        X1_f32 = Float32[1.0, 2.0]
        Y1_f32 = Float32[10.0, 11.0, 12.0]
        X0_f32 = Float32[1.5 2.5; 3.5 4.5]
        Y0_f32 = Float32[10.5 11.5; 12.5 13.5; 14.5 15.5]
        prob_f32 = SyntheticControl.SyntheticControlData(X1_f32, Y1_f32, X0_f32, Y0_f32, predictor_names, donor_ids, treated_id)
        @test prob_f32 isa SyntheticControl.SyntheticControlData{Float32}

        # Dimension Mismatch Tests
        # X1 length mismatch (expected K = 2, got 3)
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlData(
            [1.0, 2.0, 3.0], Y1, X0, Y0, predictor_names, donor_ids, treated_id
        )

        # Y0 rows mismatch (expected T_pre = 3, got 2)
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlData(
            X1, Y1, X0, Y0[1:2, :], predictor_names, donor_ids, treated_id
        )

        # Y0 columns mismatch (expected J = 2, got 3)
        Y0_bad_cols = [Y0 [1.0; 2.0; 3.0]]
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlData(
            X1, Y1, X0, Y0_bad_cols, predictor_names, donor_ids, treated_id
        )

        # predictor_names length mismatch (expected K = 2, got 1)
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlData(
            X1, Y1, X0, Y0, ["P1"], donor_ids, treated_id
        )

        # donor_ids length mismatch (expected J = 2, got 3)
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlData(
            X1, Y1, X0, Y0, predictor_names, ["D1", "D2", "D3"], treated_id
        )
    end

    # 3. SyntheticControlResult Construction & Validation
    @testset "SyntheticControlResult Construction" begin
        X1 = [1.0, 2.0]
        Y1 = [10.0, 11.0, 12.0]
        X0 = [1.5 2.5; 3.5 4.5]
        Y0 = [10.5 11.5; 12.5 13.5; 14.5 15.5]
        predictor_names = ["P1", "P2"]
        donor_ids = ["D1", "D2"]
        treated_id = "T1"
        prob = SyntheticControl.SyntheticControlProblem(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)

        # Valid result
        W = [0.4, 0.6]
        V = [0.8, 0.2]
        mspe = 0.05
        res = SyntheticControl.SyntheticControlResult(prob, W, V, mspe)
        @test res isa SyntheticControl.SyntheticControlResult{Float64}
        @test res.data === prob.data
        @test res.W == W
        @test res.V == V
        @test res.mspe == mspe

        # Dimension mismatch in W (expected 2 elements, got 3)
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlResult(prob, [0.4, 0.4, 0.2], V, mspe)
    end

    # 4. SyntheticControlProblem Initialization
    @testset "SyntheticControlProblem Initialization" begin
        X1 = [1.0, 2.0]
        Y1 = [10.0, 11.0, 12.0]
        X0 = [1.5 2.5; 3.5 4.5]
        Y0 = [10.5 11.5; 12.5 13.5; 14.5 15.5]
        predictor_names = ["P1", "P2"]
        donor_ids = ["D1", "D2"]
        treated_id = "T1"
        data = SyntheticControl.SyntheticControlData(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)

        solver = SyntheticControl.SyntheticControlProblem(data)
        @test solver isa SyntheticControl.SyntheticControlProblem{Float64}
        @test solver.data === data
        # Initial V weights should be uniform (1 / K)
        @test solver.best_V ≈ [0.5, 0.5]
        # Initial W weights should be uniform (1 / J)
        @test solver.best_W ≈ [0.5, 0.5]
    end

    # 5. Core Mathematical Functions
    @testset "Mathematical Functions" begin
        X1 = [1.0, 2.0]
        Y1 = [10.0, 20.0]
        X0 = [2.0 3.0; 4.0 5.0]
        Y0 = [5.0 15.0; 10.0 25.0]
        predictor_names = ["P1", "P2"]
        donor_ids = ["D1", "D2"]
        treated_id = "T1"
        prob = SyntheticControl.SyntheticControlProblem(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)

        V = [2.0, 0.5]
        W = [0.3, 0.7]

        # 5.1 weight_squared_distance with vector V
        # Expected residual = X1 - X0 * W
        # X0 * W = [2*0.3 + 3*0.7, 4*0.3 + 5*0.7] = [2.7, 4.7]
        # residual = [1.0 - 2.7, 2.0 - 4.7] = [-1.7, -2.7]
        # sum(residual .^ 2 .* V) = (-1.7)^2 * 2.0 + (-2.7)^2 * 0.5 = 9.425
        expected_distance = 9.425
        actual_distance = SyntheticControl.weight_squared_distance(X0, X1, V, W)
        @test actual_distance ≈ expected_distance

        # 5.2 calculate_mspe
        # Expected residual_y = Y1 - Y0 * W
        # Y0 * W = [5*0.3 + 15*0.7, 10*0.3 + 25*0.7] = [12.0, 20.5]
        # residual_y = [10.0 - 12.0, 20.0 - 20.5] = [-2.0, -0.5]
        # mspe = dot(residual_y, residual_y) / length(residual_y) = (4.0 + 0.25) / 2 = 2.125
        expected_mspe = 2.125
        actual_mspe = SyntheticControl.calculate_mspe(Y1, Y0, W)
        @test actual_mspe ≈ expected_mspe
    end

    @testset "Inner Optimizer" begin
        @testset "Unique full-rank solution" begin
            X0 = [1.0 0.0 0.0;
                  0.0 1.0 0.0;
                  0.0 0.0 1.0]
            expected_w = [0.2, 0.3, 0.5]
            prob = optimizer_fixture(X0, X0 * expected_w)
            raw_v = [0.4, 0.2, 0.4]

            w_active, v_active, loss_active = active_set_inner_solution(prob, raw_v)
            w_ref, v_ref, loss_ref = ipnewton_inner_solution(prob, raw_v)

            assert_simplex_weights(w_active; atol=1e-10)
            @test v_active ≈ v_ref
            @test w_active ≈ expected_w atol=1e-5
            @test w_active ≈ w_ref atol=1e-5
            @test loss_active <= loss_ref + 1e-8
        end

        @testset "Boundary one-donor solution" begin
            X0 = [1.0 2.0 4.0;
                  2.0 0.0 1.0]
            prob = optimizer_fixture(X0, X0[:, 2])
            raw_v = [1.0, 1.0]

            w_active, _v_active, loss_active = active_set_inner_solution(prob, raw_v)
            w_ref, _v_ref, loss_ref = ipnewton_inner_solution(prob, raw_v)

            assert_simplex_weights(w_active; atol=1e-10)
            @test argmax(w_active) == 2
            @test w_active[2] ≈ 1.0 atol=1e-5
            @test loss_active <= loss_ref + 1e-8
        end

        @testset "Rank deficient duplicate donors" begin
            X0 = [1.0 1.0 2.0 3.0;
                  0.0 0.0 1.0 1.0]
            prob = optimizer_fixture(X0, [1.0, 0.0])
            raw_v = [0.6, 0.4]

            w_active, _v_active, loss_active = active_set_inner_solution(prob, raw_v)
            w_ref, _v_ref, loss_ref = ipnewton_inner_solution(prob, raw_v)

            assert_simplex_weights(w_active; atol=1e-10)
            @test w_active[1] + w_active[2] ≈ 1.0 atol=1e-5
            @test loss_active <= loss_ref + 1e-8
        end

        @testset "Raw V normalization edge cases" begin
            X0 = [0.0 1.0 2.0;
                  1.0 0.0 1.0]
            prob = optimizer_fixture(X0, [0.25, 0.75])

            for raw_v in ([0.0, 0.0], [-2.0, 3.0], [Inf, 1.0])
                w_active, v_active, loss_active = active_set_inner_solution(prob, raw_v)
                w_ref, v_ref, loss_ref = ipnewton_inner_solution(prob, raw_v)

                assert_simplex_weights(w_active; atol=1e-10)
                @test v_active ≈ v_ref
                @test sum(v_active) ≈ 1 atol=1e-12
                @test all(>=(0), v_active)
                @test loss_active <= loss_ref + 1e-8
            end
        end

        @testset "Float32 support" begin
            X0 = Float32[1 0 2 3;
                         0 1 1 2;
                         2 1 0 1]
            prob = optimizer_fixture(X0, Float32[0.7, 0.2, 1.4])
            raw_v = Float32[0.1, 0.7, 0.2]

            w_active, _v_active, loss_active = active_set_inner_solution(prob, raw_v)
            w_ref, _v_ref, loss_ref = ipnewton_inner_solution(prob, raw_v)

            assert_simplex_weights(w_active; atol=1f-4)
            @test loss_active <= loss_ref + 1f-3
        end

        @testset "Generated SCM problems match IPNewton loss" begin
            for seed in 1:5
                prob, _true_w = DataGenerator.generate_synthetic_data(
                    K=3 + seed % 3,
                    J=5 + seed % 4,
                    T_pre=10 + seed,
                    noise_level=0.01,
                    seed=100 + seed
                )
                raw_v = rand(MersenneTwister(seed), length(prob.X1))

                w_active, _v_active, loss_active = active_set_inner_solution(prob, raw_v)
                _w_ref, _v_ref, loss_ref = ipnewton_inner_solution(prob, raw_v)

                assert_simplex_weights(w_active; atol=1e-9)
                @test loss_active <= loss_ref + 1e-6 * max(1.0, abs(loss_ref))
            end
        end
    end

    # 6. Solver execution
    @testset "Solver Execution" begin
        prob, _ = DataGenerator.generate_synthetic_data(K=3, J=5, T_pre=10, noise_level=0.0)
        res = CommonSolve.solve(prob)
        @test res isa SyntheticControl.SyntheticControlResult{Float64}
        @test length(res.W) == 5
        @test length(res.V) == 3
        assert_simplex_weights(res.V; atol=1e-10)
        assert_simplex_weights(res.W; atol=1e-8)
        @test isfinite(res.mspe)

        @testset "J greater than K" begin
            prob_j_gt_k, _ = DataGenerator.generate_synthetic_data(K=3, J=8, T_pre=12, noise_level=0.01, seed=21)
            res_j_gt_k = CommonSolve.solve(prob_j_gt_k)
            assert_simplex_weights(res_j_gt_k.V; atol=1e-10)
            assert_simplex_weights(res_j_gt_k.W; atol=1e-8)
            @test isfinite(res_j_gt_k.mspe)
        end

        @testset "Collinear donors" begin
            X0 = [1.0 1.0 2.0 3.0 4.0;
                  0.0 0.0 1.0 1.0 2.0;
                  2.0 2.0 3.0 4.0 5.0]
            X1 = [1.0, 0.0, 2.0]
            Y0 = [1.0 1.0 2.0 3.0 4.0;
                  2.0 2.0 3.0 4.0 5.0;
                  3.0 3.0 4.0 5.0 6.0;
                  4.0 4.0 5.0 6.0 7.0]
            Y1 = Y0[:, 1]
            prob_collinear = SyntheticControl.SyntheticControlProblem(
                X1,
                Y1,
                X0,
                Y0,
                ["P1", "P2", "P3"],
                ["D1", "D2", "D3", "D4", "D5"],
                "Treated"
            )
            res_collinear = CommonSolve.solve(prob_collinear)
            assert_simplex_weights(res_collinear.V; atol=1e-10)
            assert_simplex_weights(res_collinear.W; atol=1e-8)
            @test isfinite(res_collinear.mspe)
        end

        @testset "Dominant predictor" begin
            X0 = [0.0 1.0 2.0 3.0;
                  3.0 2.0 1.0 0.0;
                  1.0 2.0 1.0 2.0]
            X1 = [2.0, 1.5, 1.2]
            Y0 = [0.0 1.0 2.0 3.0;
                  0.5 1.5 2.5 3.5;
                  1.0 2.0 3.0 4.0;
                  1.5 2.5 3.5 4.5]
            Y1 = Y0[:, 3]
            prob_dominant = SyntheticControl.SyntheticControlProblem(
                X1,
                Y1,
                X0,
                Y0,
                ["Dominant", "P2", "P3"],
                ["D1", "D2", "D3", "D4"],
                "Treated"
            )
            res_dominant = CommonSolve.solve(prob_dominant)
            assert_simplex_weights(res_dominant.V; atol=1e-10)
            assert_simplex_weights(res_dominant.W; atol=1e-8)
            @test isfinite(res_dominant.mspe)
        end
    end

    # 7. Data Generator Pipeline Tests
    @testset "Data Generator Pipeline" begin
        # Default construction
        prob, true_W = DataGenerator.generate_synthetic_data()
        @test prob isa SyntheticControl.SyntheticControlProblem{Float64}
        @test true_W isa Vector{Float64}
        @test length(prob.X1) == 5
        @test length(prob.Y1) == 15
        @test size(prob.X0) == (5, 10)
        @test size(prob.Y0) == (15, 10)
        @test length(true_W) == 10
        @test sum(true_W) ≈ 1.0
        @test SyntheticControl.calculate_mspe(prob.Y1, prob.Y0, true_W) < 1e-3
        # Weights should be sparse: at most 3 non-zero elements
        @test count(w -> w > 0.0, true_W) <= 3

        # Float32 construction with custom sizes
        prob_f32, true_W_f32 = DataGenerator.generate_synthetic_data(T=Float32, K=3, J=6, T_pre=12)
        @test prob_f32 isa SyntheticControl.SyntheticControlProblem{Float32}
        @test true_W_f32 isa Vector{Float32}
        @test length(prob_f32.X1) == 3
        @test length(prob_f32.Y1) == 12
        @test size(prob_f32.X0) == (3, 6)
        @test size(prob_f32.Y0) == (12, 6)
        @test length(true_W_f32) == 6
        @test sum(true_W_f32) ≈ 1.0f0
        @test SyntheticControl.calculate_mspe(prob_f32.Y1, prob_f32.Y0, true_W_f32) < 1f-3

        # Reproducibility check with seed
        prob1, true_W1 = DataGenerator.generate_synthetic_data(seed=100)
        prob2, true_W2 = DataGenerator.generate_synthetic_data(seed=100)
        @test true_W1 == true_W2
        @test prob1.X1 == prob2.X1
        @test prob1.Y1 == prob2.Y1
        @test prob1.X0 == prob2.X0
        @test prob1.Y0 == prob2.Y0

        # Different seed gives different weights
        _, true_W3 = DataGenerator.generate_synthetic_data(seed=200)
        @test true_W1 != true_W3
    end
end
