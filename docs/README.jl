# # `Octopodes.jl` － Joint exoplanet modelling

# ## Getting started

# Basic usage:


using Octopodes, Random

dict = Octopodes.Examples.small_dict()
runs = IndepRuns(dict)
b = Binning(runs, n_log_P_yr_intervals = 20, n_log_q_intervals = 20)
binned = bin(b, runs)
Octopodes.run_imh(Xoshiro(1), binned)

# ## How to generate/preview doc

# Use `include("docs/preview.jl")` or `include("docs/make.jl")`.


# ## How to develop

# `JET` and `Revise` need to be in sync, so use the following to start Julia:
# `./dev.sh`
# This will load the Test environment.
# Run individual tests with e.g., `include("test/test_imh.jl)`.

