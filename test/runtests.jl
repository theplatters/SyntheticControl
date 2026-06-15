using Test
using SyntheticControl
using LinearAlgebra
using CommonSolve

@testset "SyntheticControl Package Tests" begin

    # 1. Compilation & Module Check
    @testset "Compilation" begin
        @test isdefined(SyntheticControl, :SyntheticControlProblem)
        @test isdefined(SyntheticControl, :SyntheticControlResult)
        @test isdefined(SyntheticControl, :SyntheticControlSolver)
        @test isdefined(SyntheticControl, :weight_squared_distance)
        @test isdefined(SyntheticControl, :weight_squared_distance_cached!)
        @test isdefined(SyntheticControl, :calculate_mspe!)
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
        # Initial weights should be uniform (1 / J)
        @test solver.W ≈ [0.5, 0.5]
        # Check cache dimensions
        @test length(solver.norm_cache) == length(X1)
        @test length(solver.mspe_cache) == length(Y1)
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

        # 5.1 weight_squared_distance with Diagonal(V)
        # Expected residual = X1 - X0 * W
        # X0 * W = [2*0.3 + 3*0.7, 4*0.3 + 5*0.7] = [2.7, 4.7]
        # residual = [1.0 - 2.7, 2.0 - 4.7] = [-1.7, -2.7]
        # dot(residual, Diagonal(V), residual) = (-1.7)^2 * 2.0 + (-2.7)^2 * 0.5 = 9.425
        expected_distance = 9.425
        actual_distance = SyntheticControl.weight_squared_distance(X0, X1, Diagonal(V), W)
        @test actual_distance ≈ expected_distance

        # 5.2 weight_squared_distance_cached! with Diagonal(V)
        cache = zeros(2)
        cached_distance = SyntheticControl.weight_squared_distance_cached!(cache, X0, X1, Diagonal(V), W)
        @test cached_distance ≈ expected_distance
        @test cache ≈ [-1.7, -2.7]

        # 5.3 weight_squared_distance with Vector V (Throws MethodError due to package implementation)
        # V is expected to be a Vector, but dot(x, V, y) is not defined for Vector V.
        @test_throws MethodError SyntheticControl.weight_squared_distance(X0, X1, V, W)
        @test_throws MethodError SyntheticControl.weight_squared_distance_cached!(cache, X0, X1, V, W)

        # 5.4 calculate_mspe!
        # Expected residual_y = Y1 - Y0 * W
        # Y0 * W = [5*0.3 + 15*0.7, 10*0.3 + 25*0.7] = [12.0, 20.5]
        # residual_y = [10.0 - 12.0, 20.0 - 20.5] = [-2.0, -0.5]
        # mspe = dot(residual_y, residual_y) / length(residual_y) = (4.0 + 0.25) / 2 = 2.125
        cache_y = zeros(2)
        expected_mspe = 2.125
        actual_mspe = SyntheticControl.calculate_mspe!(cache_y, Y1, Y0, W)
        @test actual_mspe ≈ expected_mspe
        @test cache_y ≈ [-2.0, -0.5]
    end

    # 6. Solver execution (Known Failure Mode)
    @testset "Solver Execution (Known Failure)" begin
        X1 = [1.0, 2.0]
        Y1 = [10.0, 11.0, 12.0]
        X0 = [1.5 2.5; 3.5 4.5]
        Y0 = [10.5 11.5; 12.5 13.5; 14.5 15.5]
        predictor_names = ["P1", "P2"]
        donor_ids = ["D1", "D2"]
        treated_id = "T1"
        prob = SyntheticControl.SyntheticControlProblem(X1, Y1, X0, Y0, predictor_names, donor_ids, treated_id)

        # Verification of the internal bug causing UndefVarError (K used before definition in solve)
        @test_throws UndefVarError CommonSolve.solve(prob)
    end
end
