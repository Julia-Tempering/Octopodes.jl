### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 626dcd7e-c4b7-11f0-02ee-69a651bb978d
begin
    import Pkg
    Pkg.activate(Base.current_project())
    
	using JLD2, CairoMakie, LogExpFunctions, Pigeons, DynamicPPL, Distributions, 
    Turing, Enzyme, LogDensityProblems, LogDensityProblemsAD, ForwardDiff, 
    LinearAlgebra, Random, BenchmarkTools, MCMCChains, Pluto, PlutoUI
	include("main.jl")
end 

# ╔═╡ 50a52357-6f34-4b7f-84bf-2197b785a338
md"""
## Goal

We consider the semi-hierchical model, i.e., sharing only the planet prevalence parameter $\pi$ across stars. 

The goal is to investigate the sensitivity of the posterior on $\pi$ with respect to the choice of prior, subsampling, and safe Bayes version of the octo-fitter posteriors.
"""

# ╔═╡ 687a0b79-41b2-44af-85b9-23b31465679d
@bind data_source Select(data_sources(); default=:dat_44k)

# ╔═╡ 4689f123-84a1-4ccf-8afd-0a711c714489
data = load_data_source(data_source)

# ╔═╡ 030d6d4c-8078-4ecd-b69e-c67c695ca239
shuffled_log_BF = shuffle(MersenneTwister(1), log_BF(data))

# ╔═╡ 89795bd3-6948-4f0d-91f9-fbd0eb6db712
md"""
## Subsampling and "safe Bayes"

We define subsampling as picking a subset of stars at random. 

Safe Bayes consists in the following: each octo-fitter contributes a posterior probability for the planet model, consider annealing these probabilities. 

The sufficient statistics for computing the semi-hierchical posterior consists in the individual octo-fitter posteriors on planet presence, with the annealing and subsampling applied. The sufficient statistics can be visualized by sorting stars with respect to these processed octo-fitter posteriors:
"""

# ╔═╡ 467d3f9d-cb2c-4acb-9989-2da1111ea579
@bind subsampling_strength PlutoUI.Slider(-4.0:0.01:0.0)

# ╔═╡ 4b336540-4d0b-4942-8d64-d2ca5c176142
subsampling = floor(Int, length(shuffled_log_BF) * 10.0^subsampling_strength)

# ╔═╡ cc30d78d-9e4e-4e81-855c-16a75173bebc
posterior_annealing = @bind posterior_annealing PlutoUI.Slider(0.0:0.01:1.0; default = 1.0, show_value = true)

# ╔═╡ b88d9aad-ee9f-4357-82c6-0a377fe8d83d
processed_log_BF = posterior_annealing*shuffled_log_BF[1:subsampling]

# ╔═╡ 7424e650-c1f2-4fb9-b964-ef2ca25397ee
let (fig, ax, _) = lines(planet_probabilities(sort(processed_log_BF)))
	ax.xlabel = "star indices (ranked by octofitter posterior on planet presence)"
	ax.ylabel = "octofitter posterior on planet presence"
	fig
end

# ╔═╡ 31895014-64ee-4c6e-9aa2-bcf739466be3
md"""
## Choice of prior
"""

# ╔═╡ cb74b9c0-0290-41f7-8e3f-dfef72ab8611
beta_prior_alpha = @bind beta_prior_alpha PlutoUI.Slider(0.5:0.01:2.0; default=2.0, show_value=true)

# ╔═╡ f4adb8b3-68f1-412b-8b83-076d1e74902a
beta_prior_beta = @bind beta_prior_beta PlutoUI.Slider(0.5:0.01:2.0; default=2.0, show_value=true)

# ╔═╡ 8bc5cba8-0615-4e01-9a47-f5115fec34c7
eps = 0.01

# ╔═╡ 99c7db20-1e46-4233-8e03-4e3176daf314
prior = semi_hierarchical_pi_posterior_density(eps, Float64[], beta_prior_alpha, beta_prior_beta)

# ╔═╡ 8d2068bc-efe0-49fa-aa7b-8743440a4bfe
posterior = semi_hierarchical_pi_posterior_density(eps, processed_log_BF, beta_prior_alpha, beta_prior_beta)

# ╔═╡ ff2225d1-e07c-4567-a807-fd56fabfa892
let (fig, ax, _) = lines(eps:eps:(1-eps), prior, linestyle = :dash)
	lines!(ax, eps:eps:(1-eps), posterior)
	ax.xlabel = "π  (solid: posterior; dashed: prior)" 
	ax.ylabel = "density"
	fig
end

# ╔═╡ Cell order:
# ╟─626dcd7e-c4b7-11f0-02ee-69a651bb978d
# ╟─50a52357-6f34-4b7f-84bf-2197b785a338
# ╟─687a0b79-41b2-44af-85b9-23b31465679d
# ╟─4689f123-84a1-4ccf-8afd-0a711c714489
# ╟─030d6d4c-8078-4ecd-b69e-c67c695ca239
# ╟─89795bd3-6948-4f0d-91f9-fbd0eb6db712
# ╟─7424e650-c1f2-4fb9-b964-ef2ca25397ee
# ╟─b88d9aad-ee9f-4357-82c6-0a377fe8d83d
# ╟─467d3f9d-cb2c-4acb-9989-2da1111ea579
# ╟─4b336540-4d0b-4942-8d64-d2ca5c176142
# ╟─cc30d78d-9e4e-4e81-855c-16a75173bebc
# ╟─ff2225d1-e07c-4567-a807-fd56fabfa892
# ╟─31895014-64ee-4c6e-9aa2-bcf739466be3
# ╟─cb74b9c0-0290-41f7-8e3f-dfef72ab8611
# ╟─f4adb8b3-68f1-412b-8b83-076d1e74902a
# ╟─8bc5cba8-0615-4e01-9a47-f5115fec34c7
# ╠═99c7db20-1e46-4233-8e03-4e3176daf314
# ╠═8d2068bc-efe0-49fa-aa7b-8743440a4bfe
