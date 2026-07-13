function robustness_panel(; times=[2001, 2003, 2006, 2010, 2015])
    return (
        unit=repeat([:treated, :a, :b, :c], inner=5),
        time=repeat(times, 4),
        outcome=[
            10.0, 13.0, 15.0, 20.0, 24.0,
             8.0,  9.0, 11.0, 12.0, 13.0,
            13.0, 14.0, 16.0, 17.0, 18.0,
            17.0, 19.0, 21.0, 22.0, 23.0,
        ],
        x=[
            2.0, 2.2, 2.4, 2.6, 2.8,
            1.0, 1.1, 1.2, 1.3, 1.4,
            3.0, 3.1, 3.2, 3.3, 3.4,
            5.0, 5.1, 5.2, 5.3, 5.4,
        ],
    )
end

function robustness_problem(; times=[2001, 2003, 2006, 2010, 2015])
    panel = robustness_panel(; times=times)
    problem = from_table(
        panel;
        unit=:unit,
        time=:time,
        outcome=:outcome,
        predictors=[:x],
        treated=:treated,
        treatment_time=times[4],
    )
    return problem, solve(problem)
end

function robustness_tests(; run_makie=true)
    @testset "Robustness and placebo inference" begin
        @testset "central RMSPE conventions" begin
            gaps = [3.0, 4.0, 0.0, 12.0]
            @test pre_treatment_rmspe(gaps, 3) ≈ 5 / sqrt(2)
            @test post_treatment_rmspe(gaps, 3) ≈ 12 / sqrt(2)
            @test rmspe_ratio(gaps, 3) ≈ 12 / 5
            @test rmspe_ratio(0.0, 0.0) == 1.0
            @test isinf(rmspe_ratio(0.0, 1.0))
        end

        @testset "in-space complete refits and inference" begin
            problem, solution = robustness_problem()
            placebos = in_space_placebos(problem, solution; units=[:a, :b], parallel=false)
            @test placebos isa InSpacePlaceboResult
            @test [refit.assignment for refit in placebos.refits] == [:a, :b]
            @test all(refit -> !(refit.assignment in refit.donor_ids), placebos.refits)
            @test all(refit -> !(:treated in refit.donor_ids), placebos.refits)
            @test all(refit -> refit.result !== solution, placebos.refits)
            @test all(refit -> refit.gap ≈ refit.actual .- refit.synthetic, placebos.refits)
            @test all(refit -> refit.pre_rmspe ≈ pre_treatment_rmspe(refit.gap, refit.treatment_index), placebos.refits)

            contaminated = in_space_placebos(problem, solution; units=[:a], include_treated=true)
            @test :treated in only(contaminated.refits).donor_ids

            penalized_problem = PenalizedSyntheticControlProblem(problem.data; lambda=2.0)
            penalized_solution = solve(penalized_problem)
            penalized_placebo = in_space_placebos(penalized_problem, penalized_solution; units=[:a])
            @test only(penalized_placebo.refits).result isa PenalizedSyntheticControlResult
            @test only(penalized_placebo.refits).result.lambda == 2.0

            filtered = in_space_placebos(problem, solution; rmspe_cutoff=0.0)
            @test length(filtered.refits) == 3
            @test all(refit -> refit.solver_status == :failed || !refit.included || refit.pre_rmspe == 0, filtered.refits)
            counts = robustness_counts(filtered)
            @test counts.successful + counts.failed == 3
            if placebos.p_value !== nothing
                eligible = [refit for refit in placebos.refits if refit.included]
                expected = (1 + count(refit -> refit.rmspe_ratio >= placebos.treated.rmspe_ratio, eligible)) /
                           (1 + length(eligible))
                @test randomization_p_value(placebos) == expected
            end

            serial = in_space_placebos(problem, solution)
            parallel = in_space_placebos(problem, solution; parallel=true)
            @test [refit.solver_status for refit in serial.refits] == [refit.solver_status for refit in parallel.refits]
            @test [refit.synthetic for refit in serial.refits] ≈ [refit.synthetic for refit in parallel.refits]
            @test_throws ArgumentError in_space_placebos(problem, solution; units=[:a, :a])
            @test_throws ArgumentError in_space_placebos(problem, solution; units=[:unknown])
        end

        @testset "leave-one-out refits" begin
            problem, solution = robustness_problem()
            loo = leave_one_out(problem, solution)
            @test loo isa LeaveOneOutResult
            @test [refit.assignment for refit in loo.refits] == [:a, :b, :c]
            @test all(length(refit.donor_ids) == 2 for refit in loo.refits)
            @test all(refit -> !(refit.assignment in refit.donor_ids), loo.refits)
            @test all(refit -> refit.max_path_deviation ≈ maximum(abs.(refit.synthetic .- loo.original_path.synthetic)), loo.refits)
            @test all(refit -> refit.cumulative_post_gap ≈ sum(refit.gap[refit.treatment_index:end]), loo.refits)
            active = leave_one_out(problem, solution; active_only=true, weight_tol=0.0)
            @test all(refit -> solution.W[findfirst(isequal(refit.assignment), [:a, :b, :c])] > 0, active.refits)
            @test_throws ArgumentError leave_one_out(problem, solution; donors=[:missing])
        end

        @testset "in-time generic windows and failures" begin
            problem, solution = robustness_problem()
            timing = in_time_placebos(problem, solution; placebo_times=[2006])
            @test timing isa InTimePlaceboResult
            refit = only(timing.refits)
            @test refit.time == [2001, 2003, 2006]
            @test refit.treatment_index == 3
            @test 2010 ∉ refit.time
            @test refit.result.data.Y1 == [10.0, 13.0]

            limited = in_time_placebos(problem, solution; placebo_times=[2003], post_window=1,
                                       minimum_pre_periods=1)
            @test only(limited.refits).time == [2001, 2003]
            selected = in_time_placebos(problem, solution; placebo_times=[2003],
                                        post_window=times -> times .== 2006,
                                        minimum_pre_periods=1)
            @test only(selected.refits).time == [2001, 2006]

            failure = in_time_placebos(problem, solution; placebo_times=[2003])
            @test only(failure.refits).solver_status == :failed
            @test occursin("minimum_pre_periods", only(failure.refits).failure_reason)
            @test_throws ArgumentError in_time_placebos(problem, solution; placebo_times=[2003], on_failure=:error)
            at_treatment = in_time_placebos(problem, solution; placebo_times=[2010])
            @test only(at_treatment.refits).solver_status == :failed
            @test_throws ArgumentError in_time_placebos(problem, solution; placebo_times=[2008])

            date_times = Date.(2020, 1, [1, 3, 6, 10, 15])
            date_problem, date_solution = robustness_problem(; times=date_times)
            date_result = in_time_placebos(date_problem, date_solution; placebo_times=[date_times[3]])
            @test only(date_result.refits).assignment == date_times[3]
            @test eltype(only(date_result.refits).time) == Date
        end

        @testset "recorded empty-pool failures and Tables schemas" begin
            panel = (
                unit=repeat([:treated, :only], inner=4),
                time=repeat(1:4, 2),
                outcome=[2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 4.0],
                x=[2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0],
            )
            problem = from_table(panel; unit=:unit, time=:time, outcome=:outcome,
                                 predictors=[:x], treated=:treated, treatment_time=3)
            solution = solve(problem)
            failed_placebo = in_space_placebos(problem, solution)
            @test only(failed_placebo.refits).solver_status == :failed
            @test occursin("donor pool is empty", only(failed_placebo.refits).failure_reason)
            @test_throws ArgumentError in_space_placebos(problem, solution; on_failure=:error)
            failed_loo = leave_one_out(problem, solution)
            @test only(failed_loo.refits).solver_status == :failed

            full_problem, full_solution = robustness_problem()
            placebos = in_space_placebos(full_problem, full_solution)
            loo = leave_one_out(full_problem, full_solution)
            timing = in_time_placebos(full_problem, full_solution; placebo_times=[2006])
            @test Tables.columnnames(placebo_summary(placebos)) ==
                  (:unit, :is_treated, :pre_rmspe, :post_rmspe, :rmspe_ratio,
                   :included, :exclusion_reason, :solver_status)
            @test Tables.columnnames(leave_one_out_summary(loo)) ==
                  (:omitted_donor, :original_weight, :pre_rmspe, :post_rmspe,
                   :rmspe_ratio, :mean_post_gap, :cumulative_post_gap,
                   :max_path_deviation, :solver_status)
            @test Tables.columnnames(in_time_summary(timing)) ==
                  (:placebo_time, :pre_rmspe, :post_rmspe, :rmspe_ratio,
                   :mean_post_gap, :cumulative_post_gap, :pre_periods,
                   :post_periods, :solver_status)
        end

        run_makie && @testset "Makie robustness recipes consume stored results" begin
            problem, solution = robustness_problem()
            placebos = in_space_placebos(problem, solution)
            loo = leave_one_out(problem, solution)
            timing = in_time_placebos(problem, solution; placebo_times=[2006])
            original_weights = copy(solution.W)
            @test placeboplot!(Axis(Figure()[1, 1]), placebos; show_treatment=false) isa Makie.Plot
            @test leaveoneoutplot!(Axis(Figure()[1, 1]), loo; refit_color=:red) isa Makie.Plot
            @test intimeplaceboplot!(Axis(Figure()[1, 1]), timing; gap_color=:green) isa Makie.Plot
            @test solution.W == original_weights
            @test_throws ArgumentError leaveoneoutplot!(Axis(Figure()[1, 1]), loo; refit_alpha=1.5)
        end
    end
end
