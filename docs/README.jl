# # `Octopodes.jl` － Joint exoplanet modelling

# ## Getting started

# Basic usage 
# (using here a tiny input file only for a quick software demo, 
# that demo data is not suitable for scientific purpose, more 
# information on input format is explained in the 
# [documentation](https://julia-tempering.github.io/Octopodes.jl/dev/input)):

using Octopodes, Random

dict = Octopodes.Examples.small_dict()
input_data = IndepRuns(dict)

octopodes_result = octopodes(input_data; n_log_P_yr_intervals = 20, n_log_q_intervals = 20)
typeof(octopodes_result)

# 

# Summarize the joint-π posterior (warmup drop, per-bin rate density
# ``\lambda = \mathbb{E}[n]\,\pi``, ``P(n \ge c)`` and ``\mathbb{E}[n]``) and plot
# the population heatmap. The bin edges are read from `b`, so they never have to
# be passed by hand:

octopodes_result.population_posterior_plot

# If you also want the summarized quantities, use the `PopulationPosterior`:

post =  octopodes_result.posterior
post.lambda   # n_keep × n_log_P × n_log_q  (E[n]·π per bin)
post.P_geq    # max_n_companions × n_keep   (P(n ≥ c))
fig = population_posterior_plot(post)

# Bins whose posterior is driven by the prior rather than the data can be masked
# out using the per-bin relative prior-sensitivity. Masked bins are blanked on the
# heatmap and dropped from the marginal sums:

sens = relative_sensitivities(octopodes_result.binned, 1e-3)
fig = population_posterior_plot(post; sensitivity = sens, sensitivity_threshold = 2.0)


# ## How to preview/generate doc

# Use `include("docs/preview.jl")` or `include("docs/make.jl")` (the latter regenerates both 
# the `README.md` in the root of this repo, as well as the Documenter site). Note that the 
# changes to `README.md` need to be manually pushed to github.


# ## How to develop

# `JET` and `Revise` need to be in sync, so use the following to start Julia:
# `./dev.sh`
# This will load the Test environment.
# Run individual tests by running first `include("test/setup.jl")` and then the test file 
# you want to run, e.g., `include("test/test_imh.jl)`. To run all tests, use 
# `include("test/runtests.jl)`.

