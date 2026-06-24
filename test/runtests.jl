using Test
using SyntheticControl
using LinearAlgebra
using CommonSolve

# Include and use the data generator
include("data_generator.jl")
using .DataGenerator

@testset "SyntheticControl Package Tests" begin

    # 1. Compilation & Module Check
    @testset "Compilation" begin
        @test isdefined(SyntheticControl, :SyntheticControlProblem)
        @test isdefined(SyntheticControl, :SyntheticControlResult)
        @test isdefined(SyntheticControl, :SyntheticControlSolver)
        @test isdefined(SyntheticControl, :weight_squared_distance)
        @test isdefined(SyntheticControl, :calculate_mspe)
    end

    # 2. SyntheticControlProblem Construction & Validation
    @testset "SyntheticControlProblem Construction" begin
        # Valid inputs (Float64)
        X1 = [1.0, 2.0]
        Y1 = [10.0, 11.0, 12.0]
        X0 = [1.5 2.5; 3.5 4.5]
        Y0 = [10.5 11.5; 12.5 13.5; 14.5 15.5]
        predictor_names = ["P1", "P2"]
        donor_ids = ["D1", "D2"]
        treated_id = "T1"

        prob = SyntheticControl.SyntheticControlProblem(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)
        @test prob isa SyntheticControl.SyntheticControlProblem{Float64}
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
        prob_f32 = SyntheticControl.SyntheticControlProblem(X1_f32, Y1_f32, X0_f32, Y0_f32, predictor_names, donor_ids, treated_id)
        @test prob_f32 isa SyntheticControl.SyntheticControlProblem{Float32}

        # Dimension Mismatch Tests
        # X1 length mismatch (expected K = 2, got 3)
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlProblem(
            [1.0, 2.0, 3.0], Y1, X0, Y0, predictor_names, donor_ids, treated_id
        )

        # Y0 rows mismatch (expected T_pre = 3, got 2)
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlProblem(
            X1, Y1, X0, Y0[1:2, :], predictor_names, donor_ids, treated_id
        )

        # Y0 columns mismatch (expected J = 2, got 3)
        Y0_bad_cols = [Y0 [1.0; 2.0; 3.0]]
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlProblem(
            X1, Y1, X0, Y0_bad_cols, predictor_names, donor_ids, treated_id
        )

        # predictor_names length mismatch (expected K = 2, got 1)
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlProblem(
            X1, Y1, X0, Y0, ["P1"], donor_ids, treated_id
        )

        # donor_ids length mismatch (expected J = 2, got 3)
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlProblem(
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
        @test res.problem === prob
        @test res.W == W
        @test res.V == V
        @test res.mspe == mspe

        # Dimension mismatch in W (expected 2 elements, got 3)
        @test_throws DimensionMismatch SyntheticControl.SyntheticControlResult(prob, [0.4, 0.4, 0.2], V, mspe)
    end

    # 4. SyntheticControlSolver Initialization
    @testset "SyntheticControlSolver Initialization" begin
        X1 = [1.0, 2.0]
        Y1 = [10.0, 11.0, 12.0]
        X0 = [1.5 2.5; 3.5 4.5]
        Y0 = [10.5 11.5; 12.5 13.5; 14.5 15.5]
        predictor_names = ["P1", "P2"]
        donor_ids = ["D1", "D2"]
        treated_id = "T1"
        prob = SyntheticControl.SyntheticControlProblem(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)

        solver = SyntheticControl.SyntheticControlSolver(prob)
        @test solver isa SyntheticControl.SyntheticControlSolver{Float64}
        @test solver.prob === prob
        # Initial V weights should be uniform (1 / K)
        @test solver.V ≈ [0.5, 0.5]
        # Initial W weights should be uniform (1 / J)
        @test solver.W ≈ [0.5, 0.5]
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

    # 6. Solver execution
    @testset "Solver Execution" begin
        prob, _ = DataGenerator.generate_synthetic_data(K=3, J=5, T_pre=10, noise_level=0.0)
        res = CommonSolve.solve(prob)
        @test res isa SyntheticControl.SyntheticControlResult{Float64}
        @test length(res.W) == 5
        @test length(res.V) == 3
        # Optimal weights W must sum to 1.0
        @test sum(res.W) ≈ 1.0 atol=1e-3
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
