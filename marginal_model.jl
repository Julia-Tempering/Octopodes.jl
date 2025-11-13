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

# ╔═╡ ba56fbca-8edf-4e04-916b-ed02b967bd11
begin
    import Pkg
    Pkg.activate(Base.current_project())
    
	using JLD2, CairoMakie, LogExpFunctions, Pigeons, DynamicPPL, Distributions, 
    Turing, Enzyme, LogDensityProblems, LogDensityProblemsAD, ForwardDiff, 
    LinearAlgebra, Random, BenchmarkTools, MCMCChains, Pluto, PlutoUI
	include("main.jl")
end

# ╔═╡ 05993f6d-d590-4889-9e96-5cb23f84ecf0
md"""
### Data source
"""

# ╔═╡ 3eeac7d7-dfca-4d9c-be4b-805615782c88
@bind data_source Select(data_sources())

# ╔═╡ f8483d3c-3f88-4a65-92e2-036befa0440d
data = load_data_source(data_source)

# ╔═╡ c40df1d0-7911-4295-bcb9-112ba72fb0c3
sorted_log_BF = log_BF(data)

# ╔═╡ 049caba4-831a-47ae-b3e7-85bc00e93f5a
md"""
### Star mixture selector

Pick $N$ stars with logBF greater than right bound below, and $100-N$ with logBF smaller than left. $N$ can be picked in slider below.
"""

# ╔═╡ 491c2be6-a3cc-48bd-9ee8-c3a5ca94a2f1
left = 0

# ╔═╡ 787e4500-f426-4a00-a065-ab9251f4d126
right = 3

# ╔═╡ 1fdca356-c8d7-4b57-9cb1-55939ce9241b
@bind N NumberField(2:99, default=5)

# ╔═╡ 42cff42b-5e59-4a55-9c6b-831df83cdfd1
function subsample(x, size) 
	@assert size ≤ length(x) "$size $(length(x))"
	result = x[round.(Int, range(1, length(x), length=size))]
	@assert length(result) == size
	return result
end

# ╔═╡ 6bdd76b3-d1da-40b7-88e0-f14436e78fcb
low_BFs_indices = subsample(1:searchsortedfirst(sorted_log_BF, left), 100-N)

# ╔═╡ 1548bc74-051f-468f-a862-d56fdf9b63c7
high_BFs_indices = subsample((searchsortedlast(sorted_log_BF, right)):n_systems(data), N)

# ╔═╡ 7ca830ec-8f8a-41c1-b676-5c42dc6ab2de
subsampled_log_BFs_indices = [low_BFs_indices; high_BFs_indices]

# ╔═╡ 0d2f29c2-ab4d-490c-8764-f65025d19bd2
subsampled_log_BFs = sorted_log_BF[subsampled_log_BFs_indices]

# ╔═╡ 966f9272-cd74-4861-83b6-b47aa1e5695f
md"""
## Posterior over π (numerical integration)

Note: this model is slightly different. It is not hierarchical on the η which 
means that it is not learning where the population's mass and period are located.
So imagine we want to add one more star to the posterior: with the hierarchical 
model the marginal likelihood for the with-planet would not be computed from the uniform but from the predictive on η.
"""

# ╔═╡ c194518c-5482-441f-93c3-42662bd83e98
eps=0.001

# ╔═╡ 6ac4beac-d01f-494e-82f0-45114ece3be9
lines(0.0:eps:1.0, marginal_pi_posterior_density(eps, subsampled_log_BFs))

# ╔═╡ 089a4738-1308-4308-b754-a264af339e20
md"""
## Posterior using MCMC
"""

# ╔═╡ 3304a13b-c3be-46c5-ba71-977460723ed8
@bind algo Select([:pigeons, :turing])

# ╔═╡ e1cbce0b-de9a-4d13-b529-ab9fc94a3632
subsampled_data = subset(subsampled_log_BFs_indices, data)

# ╔═╡ 9129a7da-17a1-4a3c-aac2-2a19bfd1f130
chains = if algo == :turing 
		run_turing(subsampled_data, 10_000)
	elseif algo == :pigeons 
		to_chains(run_pigeons(subsampled_data))
	else
		error()
	end

# ╔═╡ b449658e-7f69-4713-a5cb-24105464191d
mcmc_pi_samples = vec(Array(chains[:, :π, :]))

# ╔═╡ 5c2bf009-af27-49a3-bb21-af8a281f713f
begin 
	fig = Figure()
	ax  = Axis(fig[1,1], limits = ((0.0, 1.0), nothing))
	hist!(ax, mcmc_pi_samples)
	fig
end

# ╔═╡ 35558c39-c1b3-46b1-97a0-ad1a979a9f17
mean(mcmc_pi_samples)

# ╔═╡ 0ee6d58f-db4c-47cc-a270-f5945b22fb96
lines(mcmc_pi_samples)

# ╔═╡ Cell order:
# ╟─ba56fbca-8edf-4e04-916b-ed02b967bd11
# ╟─05993f6d-d590-4889-9e96-5cb23f84ecf0
# ╠═3eeac7d7-dfca-4d9c-be4b-805615782c88
# ╟─f8483d3c-3f88-4a65-92e2-036befa0440d
# ╠═c40df1d0-7911-4295-bcb9-112ba72fb0c3
# ╟─049caba4-831a-47ae-b3e7-85bc00e93f5a
# ╟─491c2be6-a3cc-48bd-9ee8-c3a5ca94a2f1
# ╟─787e4500-f426-4a00-a065-ab9251f4d126
# ╠═1fdca356-c8d7-4b57-9cb1-55939ce9241b
# ╟─42cff42b-5e59-4a55-9c6b-831df83cdfd1
# ╠═6bdd76b3-d1da-40b7-88e0-f14436e78fcb
# ╠═1548bc74-051f-468f-a862-d56fdf9b63c7
# ╠═7ca830ec-8f8a-41c1-b676-5c42dc6ab2de
# ╠═0d2f29c2-ab4d-490c-8764-f65025d19bd2
# ╟─966f9272-cd74-4861-83b6-b47aa1e5695f
# ╠═6ac4beac-d01f-494e-82f0-45114ece3be9
# ╠═c194518c-5482-441f-93c3-42662bd83e98
# ╟─089a4738-1308-4308-b754-a264af339e20
# ╠═3304a13b-c3be-46c5-ba71-977460723ed8
# ╟─5c2bf009-af27-49a3-bb21-af8a281f713f
# ╠═35558c39-c1b3-46b1-97a0-ad1a979a9f17
# ╠═e1cbce0b-de9a-4d13-b529-ab9fc94a3632
# ╠═9129a7da-17a1-4a3c-aac2-2a19bfd1f130
# ╠═b449658e-7f69-4713-a5cb-24105464191d
# ╠═0ee6d58f-db4c-47cc-a270-f5945b22fb96
