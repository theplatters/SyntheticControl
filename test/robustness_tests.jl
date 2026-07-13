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

        @testset "time-specific placebo inference" begin
            problem, solution = robustness_problem()
            placebos = in_space_placebos(problem, solution)
            @test all(refit -> refit.solver_status === :success, placebos.refits)
            @test all(refit -> refit.included, placebos.refits)

            placebos.treated.gap[4:5] .= [2.0, -3.0]
            placebos.refits[1].gap[4:5] .= [3.0, -3.0]
            placebos.refits[2].gap[4:5] .= [1.0, 4.0]
            placebos.refits[3].gap[4:5] .= [2.0, -2.0]

            greater = pointwise_placebo_inference(placebos)
            @test greater isa PointwisePlaceboInferenceResult{Float64,Symbol,Int}
            @test greater.source === placebos
            @test greater.statistic == :gap
            @test greater.alternative == :greater
            @test greater.time == [2010, 2015]
            @test greater.treated_statistic == [2.0, -3.0]
            @test greater.placebo_statistics == [3.0 1.0 2.0; -3.0 4.0 -2.0]
            @test greater.included_assignments == [:a, :b, :c]
            @test greater.extreme_count == [3, 4]
            @test greater.assignment_count == 4
            @test greater.p_value == [0.75, 1.0]

            less = pointwise_placebo_inference(placebos; alternative=:less)
            @test less.extreme_count == [3, 2]
            @test less.p_value == [0.75, 0.5]
            two_sided = pointwise_placebo_inference(placebos; alternative=:two_sided)
            @test two_sided.extreme_count == [3, 3]
            @test two_sided.p_value == [0.75, 0.75]
            absolute = pointwise_placebo_inference(placebos; statistic=:absolute_gap)
            @test absolute.treated_statistic == [2.0, 3.0]
            @test absolute.p_value == [0.75, 0.75]

            mean_gap = aggregate_placebo_inference(placebos)
            @test mean_gap isa AggregatePlaceboInferenceResult
            @test mean_gap.source === placebos
            @test mean_gap.specification == :mean_gap
            @test mean_gap.statistic == :mean_gap
            @test mean_gap.periods == [2010, 2015]
            @test mean_gap.treated_statistic == -0.5
            @test mean_gap.placebo_statistics == [0.0, 2.5, 0.0]
            @test mean_gap.extreme_count == 4
            @test mean_gap.assignment_count == 4
            @test mean_gap.p_value == 1.0

            cumulative = aggregate_placebo_inference(
                placebos; statistic=:cumulative_gap, alternative=:less,
            )
            @test cumulative.treated_statistic == -1.0
            @test cumulative.placebo_statistics == [0.0, 5.0, 0.0]
            @test cumulative.p_value == 0.25
            cumulative_two_sided = aggregate_placebo_inference(
                placebos; statistic=:cumulative_gap, alternative=:two_sided,
            )
            @test cumulative_two_sided.p_value == 0.5

            mean_absolute = aggregate_placebo_inference(
                placebos; statistic=:mean_absolute_gap,
            )
            @test mean_absolute.treated_statistic == 2.5
            @test mean_absolute.placebo_statistics == [3.0, 2.5, 2.0]
            @test mean_absolute.p_value == 0.75
            aggregate_rmspe = aggregate_placebo_inference(placebos; statistic=:rmspe)
            @test aggregate_rmspe.treated_statistic ≈ sqrt(6.5)
            @test aggregate_rmspe.placebo_statistics ≈ [3.0, sqrt(8.5), 2.0]
            @test aggregate_rmspe.p_value == 0.75

            selected = aggregate_placebo_inference(
                placebos; statistic=:cumulative_gap, periods=[2015], alternative=:less,
            )
            @test selected.periods == [2015]
            @test selected.treated_statistic == -3.0
            reordered = aggregate_placebo_inference(
                placebos; statistic=:cumulative_gap, periods=[2015, 2010],
            )
            @test reordered.periods == [2010, 2015]
            one_argument = aggregate_placebo_inference(
                placebos; statistic=gaps -> sum(gaps),
            )
            @test one_argument.statistic == :custom
            @test one_argument.treated_statistic == -1.0
            custom = aggregate_placebo_inference(
                placebos;
                statistic=(gaps, times) -> begin
                    @test times == [2010, 2015]
                    sum(gaps .* eachindex(gaps))
                end,
            )
            @test custom.statistic == :custom
            @test custom.treated_statistic == -4.0
            @test custom.placebo_statistics == [-3.0, 9.0, -2.0]
            @test custom.p_value == 1.0

            point_table = pointwise_placebo_summary(greater)
            @test Tables.istable(point_table)
            @test Tables.columnnames(point_table) ==
                  (:time, :statistic, :alternative, :treated_statistic,
                   :extreme_count, :assignment_count, :p_value,
                   :p_value_resolution)
            @test point_table.time == [2010, 2015]
            @test point_table.p_value == [0.75, 1.0]
            @test point_table.p_value_resolution == [0.25, 0.25]
            @test length(Tables.rowtable(point_table)) == 2

            aggregate_table = aggregate_placebo_summary(mean_absolute)
            @test Tables.istable(aggregate_table)
            @test Tables.columnnames(aggregate_table) ==
                  (:statistic, :alternative, :periods, :period_count,
                   :treated_statistic, :extreme_count, :assignment_count,
                   :p_value, :p_value_resolution)
            @test eltype(aggregate_table.periods) == Vector{Int}
            @test only(aggregate_table.periods) == [2010, 2015]
            @test only(aggregate_table.p_value) == 0.75
            @test only(aggregate_table.p_value_resolution) == 0.25
            @test only(Tables.rowtable(aggregate_table)).period_count == 2

            placebos.refits[3].included = false
            placebos.refits[3].exclusion_reason = :poor_pre_fit
            filtered = pointwise_placebo_inference(placebos)
            @test filtered.included_assignments == [:a, :b]
            @test filtered.assignment_count == 3
            @test filtered.p_value == [2 / 3, 1.0]
            placebos.refits[3].included = true
            placebos.refits[3].exclusion_reason = nothing

            placebos.refits[3].solver_status = :failed
            placebos.refits[3].included = false
            failed_assignment = pointwise_placebo_inference(placebos)
            @test failed_assignment.source.refits[3].solver_status == :failed
            @test failed_assignment.included_assignments == [:a, :b]
            @test failed_assignment.assignment_count == 3
            placebos.refits[3].solver_status = :success
            placebos.refits[3].included = true

            saved_gap = placebos.refits[2].gap[4]
            placebos.refits[2].gap[4] = NaN
            nonfinite = pointwise_placebo_inference(placebos)
            @test nonfinite.included_assignments == [:a, :c]
            @test nonfinite.assignment_count == 3
            placebos.refits[2].gap[4] = saved_gap

            saved_time = placebos.refits[1].time[end]
            placebos.refits[1].time[end] = 9999
            @test_throws ArgumentError pointwise_placebo_inference(placebos)
            placebos.refits[1].time[end] = saved_time

            saved_result = placebos.refits[1].result
            placebos.refits[1].result = nothing
            @test pointwise_placebo_inference(placebos).p_value == [0.75, 1.0]
            placebos.refits[1].result = saved_result

            @test_throws ArgumentError pointwise_placebo_inference(placebos; statistic=:unknown)
            @test_throws ArgumentError pointwise_placebo_inference(placebos; alternative=:unknown)
            @test_throws ArgumentError aggregate_placebo_inference(placebos; periods=Int[])
            @test_throws ArgumentError aggregate_placebo_inference(placebos; periods=[2010, 2010])
            @test_throws ArgumentError aggregate_placebo_inference(placebos; periods=[2006])
            @test_throws ArgumentError aggregate_placebo_inference(placebos; periods=[9999])
            @test_throws ArgumentError aggregate_placebo_inference(placebos; statistic=:unknown)
            @test_throws ArgumentError aggregate_placebo_inference(placebos; statistic=gaps -> "bad")
            @test_throws ArgumentError aggregate_placebo_inference(placebos; statistic=gaps -> Inf)

            for refit in placebos.refits
                refit.included = false
                refit.exclusion_reason = :poor_pre_fit
            end
            @test_throws ArgumentError pointwise_placebo_inference(placebos)

            date_times = Date.(2020, 1, [1, 3, 6, 10, 15])
            date_problem, date_solution = robustness_problem(; times=date_times)
            date_placebos = in_space_placebos(date_problem, date_solution)
            date_pointwise = pointwise_placebo_inference(date_placebos)
            @test eltype(date_pointwise.time) == Date
            date_aggregate = aggregate_placebo_inference(
                date_placebos; periods=[date_times[5]], statistic=:mean_absolute_gap,
            )
            @test date_aggregate.periods == [date_times[5]]
            @test eltype(only(aggregate_placebo_summary(date_aggregate).periods)) == Date
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

            @testset "stored robustness long paths" begin
                filtered = in_space_placebos(full_problem, full_solution; rmspe_cutoff=0.0)
                placebo_table = placebo_paths(filtered)
                @test Tables.istable(placebo_table)
                @test Tables.columnnames(placebo_table) ==
                      (:analysis_id, :time, :actual, :synthetic, :gap,
                       :is_post_treatment, :included, :solver_status,
                       :failure_reason, :is_treated, :exclusion_reason)
                @test eltype(placebo_table.analysis_id) == Symbol
                @test eltype(placebo_table.time) == Union{Missing,Int}
                @test eltype(placebo_table.actual) == Union{Missing,Float64}
                @test placebo_table.analysis_id[1:5] == fill(:treated, 5)
                @test placebo_table.is_treated[1:5] == trues(5)
                @test placebo_table.is_post_treatment[1:5] == [false, false, false, true, true]
                @test any(.!placebo_table.included .& .!placebo_table.is_treated)
                @test all(
                    ismissing(placebo_table.actual[index]) ||
                    placebo_table.gap[index] == placebo_table.actual[index] - placebo_table.synthetic[index]
                    for index in eachindex(placebo_table.gap)
                )
                @test length(Tables.rowtable(placebo_table)) == length(placebo_table.analysis_id)

                stored_refit = first(filtered.refits)
                stored_gap = stored_refit.gap[1]
                stored_refit.gap[1] = 1234.0
                stored_refit.result = nothing
                stored_table = placebo_paths(filtered)
                @test stored_table.gap[6] == 1234.0
                stored_refit.gap[1] = stored_gap

                failed_placebo_table = placebo_paths(failed_placebo)
                @test length(failed_placebo_table.analysis_id) == 5
                @test failed_placebo_table.analysis_id[end] == :only
                @test ismissing(failed_placebo_table.time[end])
                @test ismissing(failed_placebo_table.gap[end])
                @test failed_placebo_table.solver_status[end] == :failed
                @test !ismissing(failed_placebo_table.failure_reason[end])
                @test failed_placebo_table.exclusion_reason[end] == :failed

                loo_table = leave_one_out_paths(loo)
                @test Tables.istable(loo_table)
                @test Tables.columnnames(loo_table) ==
                      (:analysis_id, :time, :actual, :synthetic, :gap,
                       :is_post_treatment, :included, :solver_status,
                       :failure_reason, :original_weight)
                @test loo_table.analysis_id == vcat(fill(:a, 5), fill(:b, 5), fill(:c, 5))
                @test loo_table.original_weight == vcat(
                    fill(full_solution.W[1], 5), fill(full_solution.W[2], 5),
                    fill(full_solution.W[3], 5),
                )
                @test all(
                    loo_table.gap[index] == loo_table.actual[index] - loo_table.synthetic[index]
                    for index in eachindex(loo_table.gap)
                )
                @test Tables.columntable(loo_table).analysis_id == loo_table.analysis_id

                failed_loo_table = leave_one_out_paths(failed_loo)
                @test length(failed_loo_table.analysis_id) == 1
                @test failed_loo_table.analysis_id == [:only]
                @test ismissing(only(failed_loo_table.time))
                @test failed_loo_table.original_weight == [only(solution.W)]
                @test only(failed_loo_table.solver_status) == :failed

                timing_table = in_time_paths(timing)
                @test Tables.istable(timing_table)
                @test Tables.columnnames(timing_table) ==
                      (:analysis_id, :time, :actual, :synthetic, :gap,
                       :is_post_treatment, :included, :solver_status,
                       :failure_reason)
                @test timing_table.analysis_id == fill(2006, 3)
                @test timing_table.time == [2001, 2003, 2006]
                @test timing_table.is_post_treatment == [false, false, true]
                @test all(
                    timing_table.gap[index] == timing_table.actual[index] - timing_table.synthetic[index]
                    for index in eachindex(timing_table.gap)
                )

                failed_timing = in_time_placebos(full_problem, full_solution; placebo_times=[2003])
                failed_timing_table = in_time_paths(failed_timing)
                @test failed_timing_table.analysis_id == [2003]
                @test ismissing(only(failed_timing_table.time))
                @test only(failed_timing_table.solver_status) == :failed
                @test !only(failed_timing_table.included)

                date_times = Date.(2020, 1, [1, 3, 6, 10, 15])
                date_problem, date_solution = robustness_problem(; times=date_times)
                date_result = in_time_placebos(
                    date_problem, date_solution; placebo_times=[date_times[3]],
                )
                date_table = in_time_paths(date_result)
                @test eltype(date_table.analysis_id) == Date
                @test eltype(date_table.time) == Union{Missing,Date}
                @test date_table.analysis_id == fill(date_times[3], 3)
                @test collect(skipmissing(date_table.time)) == date_times[1:3]
            end
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
