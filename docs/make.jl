using Documenter

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CairoMakie
using SyntheticControl
using CommonSolve
using LinearAlgebra

CairoMakie.activate!()

DocMeta.setdocmeta!(
  SyntheticControl,
  :DocTestSetup,
  :(using SyntheticControl, CommonSolve, LinearAlgebra, CairoMakie);
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
    "Visualization" => "visualization.md",
    "API Reference" => "api.md",
    "Implementation Details" => "internals.md",
  ],
  doctest=true,
  checkdocs=:all,
  warnonly=false,
)
