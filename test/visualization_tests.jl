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

function legend_labels(plot)
    return [child.label[] for child in plot.plots if hasproperty(child, :label)]
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
        path_plot = pathplot!(
            ax1,
            paths;
            actual_color=:red,
            actual_linewidth=3,
            actual_linestyle=:dashdot,
            actual_label="Observed",
            synthetic_color=:blue,
            synthetic_linewidth=2,
            synthetic_linestyle=:dash,
            synthetic_label="Counterfactual",
            treatment_color=:purple,
            treatment_linewidth=1,
            treatment_linestyle=:dot,
            treatment_label=nothing,
            show_treatment=true
        )
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
        @test path_plot.plots[1].linewidth[] == 3
        @test path_plot.plots[2].linewidth[] == 2
        @test path_plot.plots[3].linewidth[] == 1
        @test path_plot.plots[1].linestyle[] isa AbstractVector
        @test path_plot.plots[2].linestyle[] isa AbstractVector
        @test path_plot.plots[3].linestyle[] == :dot
        @test legend_labels(path_plot) == ["Observed", "Counterfactual", nothing]
        @test_nowarn axislegend(ax1; position=:lt)

        fig2 = Figure()
        ax2 = Axis(fig2[1, 1])
        gap_plot = gapplot!(
            ax2,
            paths;
            gap_color=:green,
            gap_linewidth=4,
            gap_linestyle=:dash,
            gap_label="Effect",
            zero_color=:black,
            zero_linewidth=2,
            zero_linestyle=:dot,
            zero_label=nothing,
            treatment_color=:purple,
            treatment_linewidth=3,
            treatment_linestyle=:dashdot,
            treatment_label="Intervention"
        )
        @test gap_plot isa Makie.Plot
        @test length(gap_plot.plots) == 3
        @test gap_plot.plots[1][1][] == [0.0]
        _x_gap, y_gap = primitive_xy(gap_plot.plots[2])
        @test y_gap == SyntheticControl.outcome_gap(paths)
        @test gap_plot.plots[3][1][] == [paths.treatment_time]
        @test gap_plot.plots[2].color[] == Makie.to_color(:green)
        @test Makie.to_color(gap_plot.plots[1].color[]) == Makie.to_color(:black)
        @test Makie.to_color(gap_plot.plots[3].color[]) == Makie.to_color(:purple)
        @test gap_plot.plots[2].linewidth[] == 4
        @test gap_plot.plots[1].linewidth[] == 2
        @test gap_plot.plots[3].linewidth[] == 3
        @test legend_labels(gap_plot) == [nothing, "Effect", "Intervention"]

        fig3 = Figure()
        ax3 = Axis(fig3[1, 1])
        placebo_plot = placeboplot!(
            ax3,
            placebo;
            pre_rmspe_threshold=5.0,
            treated_color=:orange,
            treated_linewidth=4,
            treated_linestyle=:dashdot,
            treated_label="Treated unit",
            placebo_color=:navy,
            placebo_alpha=0.4,
            placebo_linewidth=2,
            placebo_linestyle=:dash,
            placebo_label="Placebo units",
            zero_color=:gray20,
            zero_linewidth=2,
            zero_linestyle=:dot,
            zero_label=nothing,
            treatment_color=:purple,
            treatment_linewidth=3,
            treatment_linestyle=:dash,
            treatment_label="Treatment"
        )
        @test placebo_plot isa Makie.Plot
        @test length(placebo_plot.plots) == 5
        _x_treated, y_treated = primitive_xy(placebo_plot.plots[end-1])
        @test y_treated == placebo.treated_actual .- placebo.treated_synthetic
        @test placebo_plot.plots[end][1][] == [placebo.treatment_time]
        @test placebo_plot.plots[end-1].color[] == Makie.to_color(:orange)
        @test placebo_plot.plots[2].color[] == Makie.to_color((:navy, 0.4))
        @test placebo_plot.plots[end-1].linewidth[] == 4
        @test placebo_plot.plots[2].linewidth[] == 2
        @test placebo_plot.plots[end].linewidth[] == 3
        @test legend_labels(placebo_plot) == [nothing, "Placebo units", nothing, "Treated unit", "Treatment"]

        fig4 = Figure()
        ax4 = Axis(fig4[1, 1])
        distribution_plot = placebodistribution!(
            ax4,
            placebo;
            pre_rmspe_threshold=5.0,
            placebo_color=:gray30,
            placebo_alpha=0.5,
            placebo_markersize=8,
            placebo_marker=:circle,
            placebo_label="Controls",
            treated_color=:black,
            treated_alpha=0.8,
            treated_markersize=14,
            treated_marker=:rect,
            treated_label="Treated",
            treated_line_color=:red,
            treated_linewidth=4,
            treated_linestyle=:dashdot,
            treated_line_label=nothing,
            p_value_label="p-value: ",
            p_value_color=:blue,
            p_value_fontsize=18,
            p_value_align=(:right, :top)
        )
        @test distribution_plot isa Makie.Plot
        @test length(distribution_plot.plots) == 4
        treated_ratio, ratios = SyntheticControl.placebo_rmspe_ratios(placebo)
        included = SyntheticControl.filter_placebos(placebo; pre_rmspe_threshold=5.0)
        _x_ratios, y_ratios = primitive_xy(distribution_plot.plots[1])
        _x_treated_ratio, y_treated_ratio = primitive_xy(distribution_plot.plots[2])
        @test y_ratios == sort(ratios[included])
        @test y_treated_ratio == [treated_ratio]
        @test distribution_plot.plots[3][1][] == [treated_ratio]
        @test distribution_plot.plots[1].color[] == Makie.to_color((:gray30, 0.5))
        @test distribution_plot.plots[2].color[] == Makie.to_color((:black, 0.8))
        @test all(distribution_plot.plots[1].markersize[] .== 8)
        @test all(distribution_plot.plots[2].markersize[] .== 14)
        @test Makie.to_color(distribution_plot.plots[3].color[]) == Makie.to_color(:red)
        @test distribution_plot.plots[3].linewidth[] == 4
        @test legend_labels(distribution_plot)[1:3] == ["Controls", "Treated", nothing]

        nonmutating = pathplot(paths; axis=(; title="Custom title", xlabel="Year", ylabel="Value", xticks=2001:2005))
        @test nonmutating.plot isa Makie.Plot
        @test nonmutating.axis.title[] == "Custom title"
        @test nonmutating.axis.xlabel[] == "Year"
        @test nonmutating.axis.ylabel[] == "Value"

        default_nonmutating = gapplot(paths)
        @test default_nonmutating.axis.title[] == "Synthetic-control gap"
        @test default_nonmutating.axis.xlabel[] == "Time"
        @test default_nonmutating.axis.ylabel[] == "Actual - synthetic"

        explicit_axis = Axis(
            Figure()[1, 1];
            title="Do not overwrite",
            xlabel="Calendar year",
            ylabel="Custom gap",
            xticks=2001:2005,
            yticks=-5:5,
            limits=(2001, 2005, -2, 5),
            xscale=identity,
            yscale=identity
        )
        gapplot!(explicit_axis, paths)
        @test explicit_axis.title[] == "Do not overwrite"
        @test explicit_axis.xlabel[] == "Calendar year"
        @test explicit_axis.ylabel[] == "Custom gap"
        @test explicit_axis.xticks[] == 2001:2005
        @test explicit_axis.yticks[] == -5:5
        @test explicit_axis.limits[] == (2001, 2005, -2, 5)
        @test explicit_axis.xscale[] === identity
        @test explicit_axis.yscale[] === identity

        no_optional = gapplot!(Axis(Figure()[1, 1]), paths; show_zero=false, show_treatment=false)
        @test length(no_optional.plots) == 1

        excluded_plot = placeboplot!(
            Axis(Figure()[1, 1]),
            placebo;
            pre_rmspe_threshold=5.0,
            show_excluded=true,
            excluded_color=:pink,
            excluded_alpha=0.25,
            excluded_linewidth=2,
            excluded_linestyle=:dash,
            excluded_label="Filtered out"
        )
        @test length(excluded_plot.plots) == 6
        @test excluded_plot.plots[4].color[] == Makie.to_color((:pink, 0.25))
        @test excluded_plot.plots[4].linewidth[] == 2
        @test excluded_plot.plots[4].linestyle[] isa AbstractVector
        @test "Filtered out" in legend_labels(excluded_plot)

        color_observable = Observable(:red)
        linewidth_observable = Observable(2)
        reactive_plot = pathplot!(
            Axis(Figure()[1, 1]),
            paths;
            actual_color=color_observable,
            actual_linewidth=linewidth_observable
        )
        @test reactive_plot.plots[1].color[] == Makie.to_color(:red)
        @test reactive_plot.plots[1].linewidth[] == 2
        color_observable[] = :green
        linewidth_observable[] = 5
        @test reactive_plot.plots[1].color[] == Makie.to_color(:green)
        @test reactive_plot.plots[1].linewidth[] == 5

        with_theme(Theme(linewidth=6, markersize=17)) do
            themed_path = pathplot!(Axis(Figure()[1, 1]), paths)
            @test themed_path.plots[1].linewidth[] == 6
            @test themed_path.plots[2].linewidth[] == 6
            themed_distribution = placebodistribution!(Axis(Figure()[1, 1]), placebo; show_p_value=false)
            @test all(themed_distribution.plots[1].markersize[] .== 17)
        end

        @test_throws ArgumentError pathplot!(Axis(Figure()[1, 1]), paths; actual_linewidth=-1)
        @test_throws ArgumentError placeboplot!(Axis(Figure()[1, 1]), placebo; placebo_alpha=1.2)
        @test_throws ArgumentError placebodistribution!(Axis(Figure()[1, 1]), placebo; treated_markersize=-1)
        @test_throws DimensionMismatch SyntheticControlPathData(1:2, 2, 2, [1.0], [1.0], "bad")
        @test_throws ArgumentError SyntheticControlPlaceboResult(1:2, 3, 3, "t", ["p"], [1.0, 2.0], [1.0, 2.0], reshape([1.0, 2.0], 2, 1), reshape([1.0, 2.0], 2, 1))
    end
end
