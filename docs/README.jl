# # `Octopodes.jl` － Joint exoplanet modelling

# ## Getting started

# Basic usage:


using Octopodes, Random

dict = Octopodes.Examples.small_dict()
runs = IndepRuns(dict)
b = Binning(runs, n_log_P_yr_intervals = 20, n_log_q_intervals = 20)
binned = bin(b, runs)
result = run_imh(Xoshiro(1), binned);
typeof(result)

# Summarize the joint-π posterior (warmup drop, per-bin rate density
# ``\lambda = \mathbb{E}[n]\,\pi``, ``P(n \ge c)`` and ``\mathbb{E}[n]``) and plot
# the population heatmap. The bin edges are read from `b`, so they never have to
# be passed by hand:

fig = population_posterior_plot(result, b; warmup_frac = 0.2)

# If you also want the summarized quantities, build the `PopulationPosterior`
# explicitly and plot that:

post = population_posterior(result, b; warmup_frac = 0.2)
post.lambda   # n_keep × n_log_P × n_log_q  (E[n]·π per bin)
post.P_geq    # max_n_companions × n_keep   (P(n ≥ c))
fig = population_posterior_plot(post)

# Bins whose posterior is driven by the prior rather than the data can be masked
# out using the per-bin relative prior-sensitivity. Masked bins are blanked on the
# heatmap and dropped from the marginal sums:

sens = relative_sensitivities(binned, 1e-3)
fig = population_posterior_plot(post; sensitivity = sens, sensitivity_threshold = 2.0)

# ## How to generate/preview doc

# Use `include("docs/preview.jl")` or `include("docs/make.jl")`.


# ## How to develop

# `JET` and `Revise` need to be in sync, so use the following to start Julia:
# `./dev.sh`
# This will load the Test environment.
# Run individual tests with e.g., `include("test/test_imh.jl)`.

