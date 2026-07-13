function specification_panel(; times=collect(1:6))
  return (
    unit=repeat([:treated, :a, :b, :c], inner=6),
    time=repeat(times, 4),
    outcome=[
      10.0, 12.0, 15.0, 19.0, 25.0, 30.0,
       8.0, 10.0, 12.0, 14.0, 16.0, 18.0,
      12.0, 13.0, 16.0, 18.0, 20.0, 22.0,
      17.0, 20.0, 22.0, 24.0, 26.0, 28.0,
    ],
    x=[
      2.0, 2.5, 3.0, 4.0, 5.0, 6.0,
      1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
      3.0, 3.2, 3.6, 4.0, 4.4, 4.8,
      5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
    ],
    z=[
       9.0,  8.0,  7.0,  6.0,  5.0,  4.0,
       4.0,  4.5,  5.0,  5.5,  6.0,  6.5,
      10.0,  9.5,  9.0,  8.5,  8.0,  7.5,
      14.0, 13.0, 12.0, 11.0, 10.0,  9.0,
    ],
  )
end

function specification_problem(; times=collect(1:6))
  panel = specification_panel(; times=times)
  problem = from_table(
    panel;
    unit=:unit,
    time=:time,
    outcome=:outcome,
    predictors=[:x, :z],
    treated=:treated,
    treatment_time=times[5],
    max_pair_starts=2,
    target_mspe=1e-10,
    min_relative_mspe_improvement=0.005,
  )
  return problem, solve(problem)
end

@testset "Specification sensitivity" begin
  problem, solution = specification_problem()
  original_weights = copy(solution.W)
  original_predictor_weights = copy(solution.V)
  specs = [
    SCMSpecification(:baseline),
    SCMSpecification(:short_pre; pre_periods=[2, 3, 4]),
    SCMSpecification(:without_x; omitted_predictors=["x"]),
    SCMSpecification(:late_aggregation; predictor_periods=[3, 4]),
    SCMSpecification(:restricted_donors; donors=[:a, :c]),
  ]
  sensitivity = specification_sensitivity(problem, solution, specs)

  @test sensitivity isa SpecificationSensitivityResult{Float64,Symbol,Symbol,Int}
  @test sensitivity.baseline === solution
  @test sensitivity.baseline_diagnostics.pre_rmspe == fit_diagnostics(solution).pre_rmspe
  @test [refit.specification_id for refit in sensitivity.refits] ==
        [:baseline, :short_pre, :without_x, :late_aggregation, :restricted_donors]
  @test all(refit -> refit.solver_status === :success, sensitivity.refits)
  @test all(refit -> refit.included, sensitivity.refits)
  @test all(refit -> refit.time == collect(1:6), sensitivity.refits)
  @test all(refit -> refit.treatment_index == 5, sensitivity.refits)
  @test all(refit -> refit.gap == refit.actual .- refit.synthetic, sensitivity.refits)
  @test solution.W == original_weights
  @test solution.V == original_predictor_weights

  baseline, short_pre, without_x, late_aggregation, restricted = sensitivity.refits
  @test baseline.result.data.X1 == problem.X1
  @test baseline.result.data.Y1 == problem.Y1
  @test baseline.result.data.X0 == problem.X0
  @test baseline.result.data.Y0 == problem.Y0
  @test baseline.donor_weights ≈ solution.W
  @test baseline.predictor_weights ≈ solution.V
  @test baseline.result !== solution

  @test short_pre.result.data.Y1 == [12.0, 15.0, 19.0]
  @test short_pre.result.data.Y0 == problem.Y0[2:4, :]
  @test short_pre.result.data.X1 == problem.X1
  @test short_pre.result.data.X0 == problem.X0
  @test short_pre.donor_ids == [:a, :b, :c]
  @test short_pre.predictor_names == ["x", "z"]

  @test without_x.result.data.Y1 == problem.Y1
  @test without_x.result.data.Y0 == problem.Y0
  @test without_x.predictor_names == ["z"]
  @test without_x.result.data.X1 == problem.X1[2:2]
  @test without_x.result.data.X0 == problem.X0[2:2, :]
  @test length(without_x.predictor_weights) == 1

  expected_late_x = (3.0 + 4.0) / 2
  expected_late_z = (7.0 + 6.0) / 2
  @test late_aggregation.result.data.Y1 == problem.Y1
  @test late_aggregation.result.data.Y0 == problem.Y0
  @test late_aggregation.result.data.X1 == [expected_late_x, expected_late_z]
  @test late_aggregation.donor_ids == [:a, :b, :c]
  @test late_aggregation.predictor_names == ["x", "z"]

  @test restricted.result.data.Y1 == problem.Y1
  @test restricted.result.data.X1 == problem.X1
  @test restricted.donor_ids == [:a, :c]
  @test restricted.result.data.Y0 == problem.Y0[:, [1, 3]]
  @test restricted.result.data.X0 == problem.X0[:, [1, 3]]
  @test restricted.predictor_names == ["x", "z"]

  @testset "convenience specification grids" begin
    pre_specs = preperiod_specifications([[1, 2], [1, 2, 3]])
    @test [spec.id for spec in pre_specs] == [:preperiod_1, :preperiod_2]
    @test pre_specs[1].pre_periods == [1, 2]
    @test preperiod_specifications([[1, 2]]; ids=[:custom])[1].id == :custom

    omitted = leave_one_predictor_out(problem)
    @test [spec.omitted_predictors for spec in omitted] == [["x"], ["z"]]
    @test [spec.id for spec in omitted] == [:without_x, :without_z]

    aggregation = aggregation_period_specifications([[1, 2], [3, 4]])
    @test aggregation[2].predictor_periods == [3, 4]
    donor_specs = donor_pool_specifications([[:a, :b], [:a, :c]])
    @test donor_specs[1].donors == [:a, :b]
    @test_throws ArgumentError preperiod_specifications([])
    @test_throws DimensionMismatch aggregation_period_specifications([[1]]; ids=[:a, :b])

    one_predictor_problem = from_table(
      (unit=specification_panel().unit, time=specification_panel().time,
       outcome=specification_panel().outcome, x=specification_panel().x);
      unit=:unit, time=:time, outcome=:outcome, predictors=[:x],
      treated=:treated, treatment_time=5,
    )
    @test_throws ArgumentError leave_one_predictor_out(one_predictor_problem)
  end

  @testset "validation and retained failures" begin
    invalid = [
      SCMSpecification(:empty_pre; pre_periods=Int[]),
      SCMSpecification(:too_short; pre_periods=[4]),
      SCMSpecification(:last_predictor; omitted_predictors=["x", "z"]),
      SCMSpecification(:empty_donors; donors=Symbol[]),
      SCMSpecification(:unknown_predictor; omitted_predictors=["missing"]),
      SCMSpecification(:unknown_donor; donors=[:missing]),
      SCMSpecification(:post_leakage; predictor_periods=[4, 5]),
      SCMSpecification(:duplicate_period; pre_periods=[2, 2]),
      SCMSpecification(:treated_donor; donors=[:treated, :a]),
    ]
    failures = specification_sensitivity(problem, solution, invalid)
    @test length(failures.refits) == length(invalid)
    @test all(refit -> refit.solver_status === :failed, failures.refits)
    @test all(refit -> !refit.included && refit.exclusion_reason === :failed, failures.refits)
    @test all(refit -> refit.failure_reason !== nothing, failures.refits)
    @test occursin("fewer than 2", failures.refits[2].failure_reason)
    @test occursin("last predictor", failures.refits[3].failure_reason)
    @test occursin("pre-treatment", failures.refits[7].failure_reason)
    @test_throws ArgumentError specification_sensitivity(
      problem, solution, [SCMSpecification(:bad; donors=Symbol[])]; on_failure=:error,
    )
    @test_throws ArgumentError specification_sensitivity(problem, solution, SCMSpecification[])
    @test_throws ArgumentError specification_sensitivity(
      problem, solution, [SCMSpecification(:same), SCMSpecification(:same; donors=[:a])],
    )
    @test_throws ArgumentError specification_sensitivity(
      problem, solution, [SCMSpecification(:one), SCMSpecification("two")],
    )
    @test_throws ArgumentError specification_sensitivity(
      problem, solution, [SCMSpecification(:one; donors=[:a]), SCMSpecification(:two; donors=[:a])],
    )
    @test_throws ArgumentError specification_sensitivity(
      problem, solution, [SCMSpecification(:short; pre_periods=[4])]; minimum_pre_periods=0,
    )
  end

  @testset "Tables reporting uses stored results" begin
    failed_spec = SCMSpecification(:failed; donors=Symbol[])
    reported = specification_sensitivity(problem, solution, [specs[2], failed_spec])

    definitions = specification_definitions(reported)
    @test Tables.istable(definitions)
    @test Tables.columnnames(definitions) ==
          (:specification_id, :pre_periods, :omitted_predictors,
           :predictor_periods, :donors, :solver_status, :included,
           :failure_reason)
    @test definitions.specification_id == [:short_pre, :failed]
    @test definitions.pre_periods[1] == [2, 3, 4]
    @test ismissing(definitions.donors[1])
    @test definitions.donors[2] == Symbol[]

    diagnostic_table = specification_diagnostics(reported)
    @test Tables.istable(diagnostic_table)
    @test first(Tables.columnnames(diagnostic_table), 4) ==
          [:specification_id, :pre_rmspe, :pre_mae, :max_absolute_pre_gap]
    @test !ismissing(diagnostic_table.pre_rmspe[1])
    @test ismissing(diagnostic_table.pre_rmspe[2])
    @test diagnostic_table.solver_status == [:success, :failed]

    paths = specification_paths(reported)
    @test Tables.istable(paths)
    @test Tables.columnnames(paths) ==
          (:specification_id, :time, :actual, :synthetic, :gap,
           :is_post_treatment, :included, :solver_status, :failure_reason)
    @test paths.specification_id[1:6] == fill(:short_pre, 6)
    @test paths.is_post_treatment[1:6] == [false, false, false, false, true, true]
    @test all(paths.gap[index] == paths.actual[index] - paths.synthetic[index] for index in 1:6)
    @test paths.specification_id[end] == :failed
    @test ismissing(paths.time[end]) && ismissing(paths.gap[end])

    weights = specification_weights(reported)
    @test Tables.istable(weights)
    @test Tables.columnnames(weights) ==
          (:specification_id, :donor, :weight, :included, :solver_status,
           :failure_reason)
    @test collect(skipmissing(weights.donor[1:3])) == [:a, :b, :c]
    @test ismissing(weights.donor[end]) && ismissing(weights.weight[end])
    @test Tables.columntable(weights).solver_status == weights.solver_status

    balance = specification_balance(reported)
    @test Tables.istable(balance)
    @test Tables.columnnames(balance) ==
          (:specification_id, :predictor, :treated, :synthetic, :difference,
           :absolute_difference, :relative_difference, :predictor_weight,
           :included, :solver_status, :failure_reason)
    @test balance.predictor[1:2] == ["x", "z"]
    @test ismissing(balance.predictor[end])
    @test length(Tables.rowtable(balance)) == 3

    stored = reported.refits[1]
    saved_gap = stored.gap[1]
    stored.gap[1] = 987.0
    stored_result_weights = copy(stored.result.W)
    stored.result.W .= NaN
    @test specification_paths(reported).gap[1] == 987.0
    @test specification_diagnostics(reported).pre_rmspe[1] == stored.diagnostics.pre_rmspe
    stored.gap[1] = saved_gap
    stored.result.W .= stored_result_weights
  end

  @testset "penalized configuration and generic times" begin
    penalized_problem = PenalizedSyntheticControlProblem(problem.data; lambda=2.5)
    penalized_solution = solve(penalized_problem)
    penalized = specification_sensitivity(
      penalized_problem, penalized_solution,
      [SCMSpecification(:penalized; donors=[:a, :b])],
    )
    penalized_refit = only(penalized.refits)
    @test penalized_refit.result isa PenalizedSyntheticControlResult
    @test penalized_refit.result.lambda == 2.5
    @test penalized_refit.predictor_weights === nothing
    @test all(ismissing, specification_balance(penalized).predictor_weight)

    date_times = Date.(2021, 1, [1, 3, 6, 10, 15, 20])
    date_problem, date_solution = specification_problem(; times=date_times)
    date_result = specification_sensitivity(
      date_problem, date_solution,
      [SCMSpecification(:date_window; pre_periods=date_times[2:4])],
    )
    @test eltype(only(date_result.refits).time) == Date
    @test eltype(specification_paths(date_result).time) == Union{Missing,Date}
    @test only(specification_definitions(date_result).pre_periods) == date_times[2:4]
  end
end
