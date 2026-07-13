function visualization_fixture()
    time = 2001:2005
    treatment_index = 4
    treated_actual = [10.0, 11.0, 12.0, 16.0, 18.0]
    treated_synthetic = [9.0, 10.0, 11.0, 13.0, 14.0]
    placebo_actual = [
        10.0 10.0 30.0
        11.0 11.0 29.0
        12.0 12.0 31.0
        16.0 20.0 35.0
        18.0 22.0 36.0
    ]
    placebo_synthetic = [
        9.0 9.0 10.0
        10.0 10.0 10.0
        11.0 11.0 10.0
        13.0 15.0 10.0
        14.0 17.0 10.0
    ]
    placebo = SyntheticControlPlaceboResult(
        time,
        treatment_index,
        2004,
        "treated",
        ["tie", "small", "poor_fit"],
        treated_actual,
        treated_synthetic,
        placebo_actual,
        placebo_synthetic
    )
    paths = SyntheticControlPathData(
        collect(time),
        treatment_index,
        2004,
        treated_actual,
        treated_synthetic,
        "treated"
    )
    return paths, placebo
end

function child_plot_types(plot)
    return [Makie.plottype(child) for child in plot.plots]
end

function primitive_xy(plot)
    points = plot[1][]
    return [point[1] for point in points], [point[2] for point in points]
end

function visualization_tests()
    @testset "Visualization statistics" begin
        paths, placebo = visualization_fixture()
        gap = SyntheticControl.outcome_gap(paths)
        @test gap == [1.0, 1.0, 1.0, 3.0, 4.0]
        @test SyntheticControl.pre_treatment_rmspe(gap, paths.treatment_index) ≈ 1.0
        @test SyntheticControl.post_treatment_rmspe(gap, paths.treatment_index) ≈ sqrt((9 + 16) / 2)
        @test SyntheticControl.rmspe_ratio(gap, paths.treatment_index) ≈ sqrt(12.5)
        @test SyntheticControl.rmspe_ratio(0.0, 0.0) == 1.0
        @test isinf(SyntheticControl.rmspe_ratio(0.0, 1.0))

        placebo_gap = SyntheticControl.placebo_gaps(placebo)
        @test placebo_gap[:, 1] == gap
        treated_ratio, ratios = SyntheticControl.placebo_rmspe_ratios(placebo)
        @test treated_ratio ≈ ratios[1]
        @test ratios[2] > treated_ratio
        @test SyntheticControl.filter_placebos(placebo; pre_rmspe_threshold=5.0) == [true, true, false]
        @test SyntheticControl.randomization_p_value(placebo; pre_rmspe_threshold=5.0) ≈ 1.0
        @test_throws ArgumentError SyntheticControl.filter_placebos(placebo; pre_rmspe_threshold=-1.0)

        zero_pre = SyntheticControlPlaceboResult(
            1:3, 3, 3, "treated", ["inf"],
            [1.0, 1.0, 2.0], [1.0, 1.0, 1.5],
            reshape([1.0, 1.0, 2.0], 3, 1),
            reshape([1.0, 1.0, 1.0], 3, 1)
        )
        @test SyntheticControl.filter_placebos(zero_pre) == [false]
        @test_throws ArgumentError SyntheticControl.randomization_p_value(zero_pre)
    end

    @testset "Makie recipes" begin
        paths, placebo = visualization_fixture()

        fig1 = Figure()
        ax1 = Axis(fig1[1, 1])
        path_plot = pathplot!(ax1, paths; actual_color=:red, synthetic_color=:blue, show_treatment=true)
        @test path_plot isa Makie.Plot
        @test length(path_plot.plots) == 3
        @test path_plot.plots[1] isa Makie.Lines
        x_actual, y_actual = primitive_xy(path_plot.plots[1])
        x_synthetic, y_synthetic = primitive_xy(path_plot.plots[2])
        @test x_actual == Float64.(paths.time)
        @test y_actual == paths.actual
        @test x_synthetic == Float64.(paths.time)
        @test y_synthetic == paths.synthetic
        @test path_plot.plots[3][1][] == [paths.treatment_time]
        @test path_plot.plots[1].color[] == Makie.to_color(:red)
        @test path_plot.plots[2].color[] == Makie.to_color(:blue)

        fig2 = Figure()
        ax2 = Axis(fig2[1, 1])
        gap_plot = gapplot!(ax2, paths; gap_color=:green, treatment_color=:purple)
        @test gap_plot isa Makie.Plot
        @test length(gap_plot.plots) == 3
        @test gap_plot.plots[1][1][] == [0.0]
        _x_gap, y_gap = primitive_xy(gap_plot.plots[2])
        @test y_gap == SyntheticControl.outcome_gap(paths)
        @test gap_plot.plots[3][1][] == [paths.treatment_time]
        @test gap_plot.plots[2].color[] == Makie.to_color(:green)

        fig3 = Figure()
        ax3 = Axis(fig3[1, 1])
        placebo_plot = placeboplot!(ax3, placebo; pre_rmspe_threshold=5.0, treated_color=:orange)
        @test placebo_plot isa Makie.Plot
        @test length(placebo_plot.plots) == 5
        _x_treated, y_treated = primitive_xy(placebo_plot.plots[end-1])
        @test y_treated == placebo.treated_actual .- placebo.treated_synthetic
        @test placebo_plot.plots[end][1][] == [placebo.treatment_time]
        @test placebo_plot.plots[end-1].color[] == Makie.to_color(:orange)

        fig4 = Figure()
        ax4 = Axis(fig4[1, 1])
        distribution_plot = placebodistribution!(ax4, placebo; pre_rmspe_threshold=5.0, treated_color=:black)
        @test distribution_plot isa Makie.Plot
        @test length(distribution_plot.plots) == 4
        treated_ratio, ratios = SyntheticControl.placebo_rmspe_ratios(placebo)
        included = SyntheticControl.filter_placebos(placebo; pre_rmspe_threshold=5.0)
        _x_ratios, y_ratios = primitive_xy(distribution_plot.plots[1])
        _x_treated_ratio, y_treated_ratio = primitive_xy(distribution_plot.plots[2])
        @test y_ratios == sort(ratios[included])
        @test y_treated_ratio == [treated_ratio]
        @test distribution_plot.plots[3][1][] == [treated_ratio]

        nonmutating = pathplot(paths)
        @test nonmutating.plot isa Makie.Plot
        @test_throws DimensionMismatch SyntheticControlPathData(1:2, 2, 2, [1.0], [1.0], "bad")
        @test_throws ArgumentError SyntheticControlPlaceboResult(1:2, 3, 3, "t", ["p"], [1.0, 2.0], [1.0, 2.0], reshape([1.0, 2.0], 2, 1), reshape([1.0, 2.0], 2, 1))
    end
end
