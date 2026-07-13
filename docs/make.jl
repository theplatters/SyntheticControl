using Documenter

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CairoMakie
using SyntheticControl
using CommonSolve
using LinearAlgebra
using Tables

CairoMakie.activate!()

if Base.get_extension(SyntheticControl, :SyntheticControlTablesExt) === nothing
  include(joinpath(@__DIR__, "..", "ext", "SyntheticControlTablesExt.jl"))
end

if Base.get_extension(SyntheticControl, :SyntheticControlMakieExt) === nothing
  include(joinpath(@__DIR__, "..", "ext", "SyntheticControlMakieExt.jl"))
end

DocMeta.setdocmeta!(
  SyntheticControl,
  :DocTestSetup,
  :(using SyntheticControl, CommonSolve, LinearAlgebra, CairoMakie, Tables);
  recursive=true,
)

makedocs(
  sitename="SyntheticControl.jl",
  modules=[SyntheticControl],
  remotes=nothing,
  format=Documenter.HTML(
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://example.com/SyntheticControl.jl/stable/",
  ),
  pages=[
    "Home" => "index.md",
    "Installation" => "installation.md",
    "Concepts" => "concepts.md",
    "End-to-End Example" => "example.md",
    "Solver Configuration" => "solver_configuration.md",
    "Interpreting Results" => "interpreting_results.md",
    "Fit Diagnostics" => "diagnostics.md",
    "Robustness and Placebo Inference" => "robustness.md",
    "Specification Sensitivity" => "specification_sensitivity.md",
    "Unified Robustness Suite" => "robustness_suite.md",
    "Tables Integration" => "tables.md",
    "Visualization" => "visualization.md",
    "API Reference" => "api.md",
    "Implementation Details" => "internals.md",
  ],
  doctest=true,
  checkdocs=:all,
  warnonly=false,
)
