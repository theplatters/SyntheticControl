struct MinimalColumnTable
  columns::NamedTuple
end

Tables.istable(::Type{MinimalColumnTable}) = true
Tables.columnaccess(::Type{MinimalColumnTable}) = true
Tables.columns(table::MinimalColumnTable) = table.columns
Tables.schema(table::MinimalColumnTable) = Tables.schema(table.columns)

mutable struct SinglePassRows
  rows::Vector{NamedTuple}
  started::Bool
end

Tables.istable(::Type{SinglePassRows}) = true
Tables.rowaccess(::Type{SinglePassRows}) = true
Tables.rows(table::SinglePassRows) = begin
  table.started && error("single-pass source was iterated more than once")
  table.started = true
  table
end
Tables.schema(::SinglePassRows) = nothing
Base.iterate(table::SinglePassRows, state::Int=1) = state > length(table.rows) ? nothing : (table.rows[state], state + 1)

function table_panel()
  return (
    unit = repeat(["treated", "donor_b", "donor_a"], inner=5),
    year = repeat(2001:2005, 3),
    outcome = [
      10.0, 11.0, 12.0, 15.0, 17.0,
      11.0, 11.5, 12.0, 12.5, 13.0,
       9.0, 10.0, 11.0, 12.0, 13.0,
    ],
    income = [
      20.0, 21.0, 22.0, 23.0, 24.0,
      22.0, 22.0, 23.0, 23.0, 24.0,
      18.0, 19.0, 20.0, 21.0, 22.0,
    ],
    population = [
      100.0, 101.0, 103.0, 104.0, 105.0,
       92.0,  94.0,  95.0,  96.0,  98.0,
       80.0,  81.0,  82.0,  83.0,  84.0,
    ],
  )
end

function shuffle_table(table)
  order = [4, 1, 12, 7, 15, 2, 9, 5, 10, 13, 3, 6, 11, 14, 8]
  return NamedTuple{keys(table)}(Tuple(getproperty(table, name)[order] for name in keys(table)))
end

function row_table(table)
  names = keys(table)
  n = length(first(values(table)))
  return [NamedTuple{names}(Tuple(getproperty(table, name)[i] for name in names)) for i in 1:n]
end

@testset "Tables.jl extension" begin
  @test isdefined(SyntheticControl, :from_table)
  @test Tables.istable(weights_table(SyntheticControlProblem([1.0], [1.0], reshape([0.5, 1.5], 1, 2), [0.8 1.2],
                                                            ["p"], ["d1", "d2"], "treated").result))

  @testset "column table construction and orientation" begin
    problem = from_table(
      shuffle_table(table_panel());
      unit=:unit,
      time=:year,
      outcome=:outcome,
      predictors=[:income, :population],
      treated="treated",
      treatment_time=2004,
    )

    @test problem.predictor_names == ["income", "population"]
    @test problem.donor_ids == ["donor_a", "donor_b"]
    @test problem.treated_id == "treated"
    @test problem.X1 ≈ [21.0, 101.33333333333333]
    @test problem.X0 ≈ [19.0 22.333333333333332; 81.0 93.66666666666667]
    @test problem.Y1 ≈ [10.0, 11.0, 12.0]
    @test problem.Y0 ≈ [9.0 11.0; 10.0 11.5; 11.0 12.0]

    result = solve(problem)
    @test sum(result.W) ≈ 1.0

    wt = weights_table(result)
    @test Tables.istable(wt)
    @test Tables.columnnames(wt) == (:donor, :weight)
    @test wt.donor == ["donor_a", "donor_b"]
    @test wt.weight == result.W

    bt = balance_table(problem, result)
    @test Tables.istable(bt)
    @test Tables.columnnames(bt) == (:predictor, :treated, :synthetic, :difference)
    @test bt.synthetic ≈ problem.X0 * result.W

    pt = path_table(problem, result)
    @test Tables.istable(pt)
    @test Tables.columnnames(pt) == (:time, :actual, :synthetic, :gap, :post_treatment)
    @test pt.time == collect(2001:2005)
    @test pt.actual ≈ [10.0, 11.0, 12.0, 15.0, 17.0]
    @test pt.post_treatment == [false, false, false, true, true]
    @test pt.synthetic ≈ [9.0 11.0; 10.0 11.5; 11.0 12.0; 12.0 12.5; 13.0 13.0] * result.W
    @test collect(Tables.rows(wt))[1].donor == "donor_a"
  end

  @testset "row-oriented and custom sources" begin
    rows = row_table(table_panel())
    row_problem = from_table(rows; unit="unit", time="year", outcome="outcome",
                             predictors=["income"], treated="treated", treatment_time=2004)
    @test row_problem.X1 ≈ [21.0]

    custom_problem = from_table(MinimalColumnTable(table_panel()); unit=:unit, time=:year, outcome=:outcome,
                                predictors=[:income], treated="treated", treatment_time=2004)
    @test custom_problem.Y0 ≈ [9.0 11.0; 10.0 11.5; 11.0 12.0]

    single_pass = SinglePassRows(rows, false)
    single_pass_problem = from_table(single_pass; unit=:unit, time=:year, outcome=:outcome,
                                     predictors=[:income], treated="treated", treatment_time=2004)
    @test single_pass.started
    @test single_pass_problem.X0 ≈ [19.0 22.333333333333332]
  end

  @testset "identifier and time types" begin
    panel = (
      unit = repeat([:treated, :b, :a], inner=3),
      time = repeat([Date(2020, 1, 1), Date(2020, 1, 2), Date(2020, 1, 3)], 3),
      outcome = [3.0, 4.0, 6.0, 2.0, 2.5, 3.0, 1.0, 1.5, 2.0],
      x = [7.0, 8.0, 9.0, 6.0, 6.5, 7.0, 4.0, 5.0, 6.0],
    )
    problem = from_table(panel; unit=:unit, time=:time, outcome=:outcome,
                         predictors=[:x], treated=:treated, treatment_time=Date(2020, 1, 3))
    @test problem.donor_ids == ["a", "b"]
    @test path_table(problem, problem.result).time == collect(panel.time[1:3])
  end

  @testset "validation" begin
    panel = table_panel()
    duplicate = merge(panel, (; unit=copy(panel.unit), year=copy(panel.year)))
    duplicate.unit[2] = duplicate.unit[1]
    duplicate.year[2] = duplicate.year[1]
    @test_throws ArgumentError from_table(duplicate; unit=:unit, time=:year, outcome=:outcome,
                                          predictors=[:income], treated="treated", treatment_time=2004)

    unbalanced = NamedTuple{keys(panel)}(Tuple(getproperty(panel, name)[1:14] for name in keys(panel)))
    @test_throws ArgumentError from_table(unbalanced; unit=:unit, time=:year, outcome=:outcome,
                                          predictors=[:income], treated="treated", treatment_time=2004)

    missing_panel = merge(panel, (; income=Union{Missing,Float64}[panel.income...]))
    missing_panel.income[1] = missing
    @test_throws ArgumentError from_table(missing_panel; unit=:unit, time=:year, outcome=:outcome,
                                          predictors=[:income], treated="treated", treatment_time=2004)

    bad_outcome = merge(panel, (; outcome=Any[panel.outcome...]))
    bad_outcome.outcome[1] = "bad"
    @test_throws ArgumentError from_table(bad_outcome; unit=:unit, time=:year, outcome=:outcome,
                                          predictors=[:income], treated="treated", treatment_time=2004)

    @test_throws ArgumentError from_table(panel; unit=:unit, time=:year, outcome=:outcome,
                                          predictors=[:unknown], treated="treated", treatment_time=2004)
    @test_throws ArgumentError from_table(panel; unit=:unit, time=:year, outcome=:outcome,
                                          predictors=[:income], treated="missing", treatment_time=2004)
    @test_throws ArgumentError from_table(panel; unit=:unit, time=:year, outcome=:outcome,
                                          predictors=[:income], treated="treated", treatment_time=1999)
    @test_throws ArgumentError from_table((unit=String[], year=Int[], outcome=Float64[], income=Float64[]);
                                          unit=:unit, time=:year, outcome=:outcome,
                                          predictors=[:income], treated="treated", treatment_time=2004)
  end
end
