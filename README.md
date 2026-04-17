# `Octopodes.jl` － Joint exoplanet modelling

## Getting started

Basic usage:

````julia
using Octopodes, Random

dict = Octopodes.Examples.small_dict()
runs = IndepRuns(dict)
b = Binning(runs, n_log_P_yr_intervals = 20, n_log_q_intervals = 20)
binned = bin(b, runs)
result = Octopodes.run_imh(Xoshiro(1), binned);
typeof(result)
````

````
@NamedTuple{psi_trace::Matrix{Float64}, pi_trace::Matrix{Float64}, accept_prs::Vector{Float64}}
````

## How to generate/preview doc

Use `include("docs/preview.jl")` or `include("docs/make.jl")`.

## How to develop

`JET` and `Revise` need to be in sync, so use the following to start Julia:
`./dev.sh`
This will load the Test environment.
Run individual tests with e.g., `include("test/test_imh.jl)`.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

