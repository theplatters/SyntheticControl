@testset "Fit diagnostics" begin
  data = SyntheticControlData(
    [2.0, 0.0],
    [2.0, 4.0, 8.0],
    [1.0 3.0; -1.0 1.0],
    [1.0 3.0; 3.0 5.0; 7.0 9.0],
    ["level", "zero baseline"],
    ["a", "b"],
    "treated",
  )
  fit = SyntheticControl.SyntheticControlResult(data, [0.25, 0.75], [0.5, 0.5], 0.25)

  @testset "analytical scalar and predictor values" begin
    diagnostics = fit_diagnostics(fit)
    @test diagnostics isa FitDiagnostics{Float64}
    @test diagnostics.pre_rmspe == 0.5
    @test diagnostics.pre_mae == 0.5
    @test diagnostics.max_absolute_pre_gap == 0.5
    @test diagnostics.effective_donor_count == 1.6
    @test diagnostics.largest_donor_weight == 0.75
    @test diagnostics.weight_sum == 1.0
    @test diagnostics.weight_sum_error == 0.0
    @test diagnostics.predictor_mae == 0.5
    @test diagnostics.predictor_max_absolute_imbalance == 0.5
    @test diagnostics.predictor_mean_relative_imbalance == Inf
    @test diagnostics.predictor_max_relative_imbalance == Inf
    @test diagnostics.solver == :classic
    @test ismissing(diagnostics.solver_success)
    @test ismissing(diagnostics.termination_status)
    @test ismissing(diagnostics.iteration_count)
    @test diagnostics.objective_value == 0.25
    @test ismissing(diagnostics.runtime_seconds)
    @test ismissing(diagnostics.objective_evaluations)

    predictor = predictor_diagnostics(fit)
    @test predictor isa PredictorDiagnostics{Float64}
    @test predictor.predictor == ["level", "zero baseline"]
    @test predictor.treated == [2.0, 0.0]
    @test predictor.synthetic == [2.5, 0.5]
    @test predictor.difference == [-0.5, -0.5]
    @test predictor.absolute_difference == [0.5, 0.5]
    @test predictor.relative_difference == [0.25, Inf]
  end

  @testset "raw weights and validation" begin
    nearly_normalized = SyntheticControl.SyntheticControlResult(
      data, [0.25, 0.750000001], [0.5, 0.5], 0.25,
    )
    diagnostics = fit_diagnostics(nearly_normalized)
    @test diagnostics.weight_sum > 1.0
    @test diagnostics.effective_donor_count == 1 / sum(abs2, nearly_normalized.W)

    @test_throws ArgumentError fit_diagnostics(
      SyntheticControl.SyntheticControlResult(data, [0.2, 0.7], [0.5, 0.5], 0.25)
    )
    @test fit_diagnostics(
      SyntheticControl.SyntheticControlResult(data, [0.2, 0.7], [0.5, 0.5], 0.25);
      weight_sum_tolerance=0.100001,
    ).weight_sum ≈ 0.9
    @test_throws ArgumentError fit_diagnostics(
      SyntheticControl.SyntheticControlResult(data, [-0.1, 1.1], [0.5, 0.5], 0.25)
    )
    @test_throws ArgumentError fit_diagnostics(
      SyntheticControl.SyntheticControlResult(data, [NaN, NaN], [0.5, 0.5], 0.25)
    )
    @test_throws ArgumentError fit_diagnostics(fit; weight_sum_tolerance=-1.0)

    no_pre_data = SyntheticControlData(
      [2.0, 0.0], Float64[], [1.0 3.0; -1.0 1.0], zeros(0, 2),
      ["level", "zero baseline"], ["a", "b"], "treated",
    )
    no_pre_fit = SyntheticControl.SyntheticControlResult(
      no_pre_data, [0.25, 0.75], [0.5, 0.5], 0.0,
    )
    @test_throws ArgumentError fit_diagnostics(no_pre_fit)

    empty_weights = SyntheticControl.SyntheticControlResult(data, [0.25, 0.75], [0.5, 0.5], 0.25)
    empty_weights.W = Float64[]
    @test_throws ArgumentError fit_diagnostics(empty_weights)
  end

  @testset "supported solver objectives" begin
    penalized = SyntheticControl.PenalizedSyntheticControlResult(
      data, [0.25, 0.75], 1.0, 0.5, 0.25, 0.75, 0.25,
    )
    diagnostics = fit_diagnostics(penalized)
    @test diagnostics.solver == :penalized
    @test diagnostics.objective_value == 0.75
  end

  @testset "Tables schemas and sinks" begin
    scalar_table = diagnostics_table(fit)
    @test Tables.istable(scalar_table)
    @test Tables.columnnames(scalar_table) == (
      :pre_rmspe, :pre_mae, :max_absolute_pre_gap, :effective_donor_count,
      :largest_donor_weight, :weight_sum, :weight_sum_error, :predictor_mae,
      :predictor_max_absolute_imbalance, :predictor_mean_relative_imbalance,
      :predictor_max_relative_imbalance, :solver, :solver_success,
      :termination_status, :iteration_count, :objective_value,
      :runtime_seconds, :objective_evaluations,
    )
    scalar_row = only(collect(Tables.rows(scalar_table)))
    @test scalar_row.pre_rmspe == 0.5
    @test ismissing(scalar_row.termination_status)
    @test eltype(scalar_table.termination_status) == Union{Missing,Symbol}

    predictor_table = predictor_diagnostics_table(fit)
    @test Tables.istable(predictor_table)
    @test Tables.columnnames(predictor_table) == (
      :predictor, :treated, :synthetic, :difference,
      :absolute_difference, :relative_difference,
    )
    @test length(collect(Tables.rows(predictor_table))) == 2
    @test Tables.columntable(predictor_table).difference == [-0.5, -0.5]
  end
end
