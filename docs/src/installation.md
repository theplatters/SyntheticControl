# Installation

From the package repository root, instantiate the package environment:

```julia
using Pkg
Pkg.instantiate()
```

For a local checkout, activate the repository and load the package:

```jldoctest install
julia> using SyntheticControl

julia> isdefined(SyntheticControl, :SyntheticControlProblem)
true
```

To build this documentation, instantiate the documentation environment and run
Documenter:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The generated static site is written to `docs/build/`.

