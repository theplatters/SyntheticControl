@testset "Unified robustness suite" begin
  problem, solution = robustness_problem()
  original_weights = copy(solution.W)
  original_predictor_weights = copy(solution.V)
  specifications = [SCMSpecification(:short_pre; pre_periods=[2001, 2003])]

  suite = robustness_suite(
    solution;
    in_space=true,
    leave_one_out=:active,
    placebo_times=[2006],
    specifications=specifications,
  )

  @test suite isa RobustnessSuiteResult
  @test suite.baseline === solution
  @test suite.diagnostics == fit_diagnostics(solution)
  @test suite.in_space isa InSpacePlaceboResult
  @test suite.leave_one_out isa LeaveOneOutResult
  @test suite.in_time isa InTimePlaceboResult
  @test suite.specification_sensitivity isa SpecificationSensitivityResult
  @test all(state -> state.status === :success, values(suite.states))
  @test solution.W == original_weights
  @test solution.V == original_predictor_weights

  direct_space = in_space_placebos(problem, solution)
  direct_leave = leave_one_out(problem, solution; active_only=true)
  direct_time = in_time_placebos(problem, solution; placebo_times=[2006])
  direct_specification = specification_sensitivity(problem, solution, specifications)
  @test [refit.assignment for refit in suite.in_space.refits] ==
        [refit.assignment for refit in direct_space.refits]
  @test [refit.synthetic for refit in suite.in_space.refits] ≈
        [refit.synthetic for refit in direct_space.refits]
  @test [refit.assignment for refit in suite.leave_one_out.refits] ==
        [refit.assignment for refit in direct_leave.refits]
  @test [refit.synthetic for refit in suite.leave_one_out.refits] ≈
        [refit.synthetic for refit in direct_leave.refits]
  @test [refit.synthetic for refit in suite.in_time.refits] ≈
        [refit.synthetic for refit in direct_time.refits]
  @test [refit.synthetic for refit in suite.specification_sensitivity.refits] ≈
        [refit.synthetic for refit in direct_specification.refits]

  @testset "disabled, skipped, and advanced configurations" begin
    disabled = robustness_suite(problem, solution)
    @test disabled.in_space === nothing
    @test disabled.leave_one_out === nothing
    @test disabled.in_time === nothing
    @test disabled.specification_sensitivity === nothing
    @test all(state -> !state.requested && state.status === :disabled, values(disabled.states))

    skipped = robustness_suite(
      problem, solution; in_time=true, specification_sensitivity=true,
    )
    @test skipped.states.in_time.status === :skipped
    @test skipped.states.specification_sensitivity.status === :skipped
    @test skipped.states.in_time.failure_reason !== nothing
    @test skipped.states.specification_sensitivity.failure_reason !== nothing

    configured = robustness_suite(
      problem,
      solution;
      in_space=InSpaceOptions(units=[:a], include_treated=true),
      leave_one_out=LeaveOneOutOptions(donors=[:b]),
      in_time=InTimeOptions(placebo_times=[2006], post_window=1),
      specification_sensitivity=SpecificationSensitivityOptions(specifications),
    )
    @test [refit.assignment for refit in configured.in_space.refits] == [:a]
    @test :treated in only(configured.in_space.refits).donor_ids
    @test [refit.assignment for refit in configured.leave_one_out.refits] == [:b]
    @test [refit.assignment for refit in configured.in_time.refits] == [2006]
    @test length(configured.specification_sensitivity.refits) == 1

    @test_throws ArgumentError robustness_suite(problem, solution; in_space=:invalid)
    @test_throws ArgumentError robustness_suite(problem, solution; leave_one_out=:invalid)
    @test_throws ArgumentError robustness_suite(
      problem, solution; in_time=InTimeOptions(placebo_times=[2006]), placebo_times=[2006],
    )
    @test_throws ArgumentError robustness_suite(
      problem, solution;
      specification_sensitivity=SpecificationSensitivityOptions(specifications),
      specifications=specifications,
    )
  end

  @testset "component failure policy and reproducibility" begin
    recorded = robustness_suite(
      problem,
      solution;
      in_space=InSpaceOptions(units=[:a]),
      in_time=InTimeOptions(placebo_times=[2010], on_failure=:error),
      suite_failure=:record,
    )
    @test recorded.in_space isa InSpacePlaceboResult
    @test recorded.states.in_space.status === :success
    @test recorded.in_time === nothing
    @test recorded.states.in_time.status === :failed
    @test occursin("before the actual treatment", recorded.states.in_time.failure_reason)
    @test_throws ArgumentError robustness_suite(
      problem,
      solution;
      in_time=InTimeOptions(placebo_times=[2010], on_failure=:error),
      suite_failure=:error,
    )

    first = robustness_suite(
      problem, solution; in_space=true, leave_one_out=true, placebo_times=[2006],
    )
    second = robustness_suite(
      problem, solution; in_space=true, leave_one_out=true, placebo_times=[2006],
    )
    @test [refit.assignment for refit in first.in_space.refits] ==
          [refit.assignment for refit in second.in_space.refits]
    @test [refit.synthetic for refit in first.in_space.refits] ≈
          [refit.synthetic for refit in second.in_space.refits]
    @test [refit.synthetic for refit in first.leave_one_out.refits] ≈
          [refit.synthetic for refit in second.leave_one_out.refits]
    @test [refit.synthetic for refit in first.in_time.refits] ≈
          [refit.synthetic for refit in second.in_time.refits]
  end

  @testset "stored Tables reports" begin
    diagnostics = diagnostics_table(suite)
    summary = robustness_summary(suite)
    @test Tables.istable(diagnostics)
    @test Tables.istable(summary)
    @test Tables.columnnames(diagnostics) == Tables.columnnames(diagnostics_table(solution))
    @test diagnostics.pre_rmspe == [suite.diagnostics.pre_rmspe]
    @test Tables.columnnames(summary) == (
      :analysis, :requested, :status, :successful, :failed, :filtered,
      :inference_eligible, :failure_reason,
    )
    @test summary.analysis == [
      :in_space, :leave_one_out, :in_time, :specification_sensitivity,
    ]
    @test summary.status == fill(:success, 4)
    @test summary.successful[1] == robustness_counts(suite.in_space).successful
    @test summary.successful[2] == robustness_counts(suite.leave_one_out).successful
    @test summary.successful[3] == robustness_counts(suite.in_time).successful
    @test ismissing(summary.inference_eligible[4])
    @test isequal(Tables.columntable(Tables.rowtable(summary)), summary)

    disabled_summary = robustness_summary(robustness_suite(problem, solution))
    @test disabled_summary.status == fill(:disabled, 4)
    @test all(ismissing, disabled_summary.successful)

    stored_weights = copy(solution.W)
    placebo_paths(suite.in_space)
    leave_one_out_paths(suite.leave_one_out)
    in_time_paths(suite.in_time)
    specification_paths(suite.specification_sensitivity)
    robustness_summary(suite)
    diagnostics_table(suite)
    placeboplot(suite.in_space)
    leaveoneoutplot(suite.leave_one_out)
    intimeplaceboplot(suite.in_time)
    @test solution.W == stored_weights
  end

  @testset "fit association" begin
    manual = SyntheticControlResult(
      solution.data, copy(solution.W), copy(solution.V), solution.mspe,
    )
    @test_throws ArgumentError robustness_suite(manual)
    explicit = robustness_suite(problem, manual)
    @test explicit.baseline === manual
    @test explicit.states.in_space.status === :disabled

    penalized_problem = PenalizedSyntheticControlProblem(problem.data; lambda=2.0)
    penalized_fit = solve(penalized_problem)
    penalized_suite = robustness_suite(
      penalized_fit; in_space=InSpaceOptions(units=[:a]),
    )
    @test penalized_suite.baseline === penalized_fit
    @test only(penalized_suite.in_space.refits).result isa PenalizedSyntheticControlResult
    @test only(penalized_suite.in_space.refits).result.lambda == 2.0
  end
end
