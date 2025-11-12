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

# ╔═╡ 87fbed8c-582a-4d70-9bf6-ff565c41e8c4
sorted_log_BF = sort(log_BF(data))

# ╔═╡ 519504b3-dc75-40ae-b482-68c5b35f1676
@bind n_stars NumberField(0:100, default=20)

# ╔═╡ 049caba4-831a-47ae-b3e7-85bc00e93f5a
md"""
### Star range selector (ranked by Bayes Factor)

Here we will pick $n_stars stars below a given Bayes Factor (red dashed line). The  number of stars can be adjusted using the text field below. The BF threshold can be adjusted with the slider.
"""

# ╔═╡ 1fdca356-c8d7-4b57-9cb1-55939ce9241b
@bind star_rank PlutoUI.Slider(n_stars:length(data.log_Ni))

# ╔═╡ eb7b764e-156d-4673-8314-cdedd8a9e6ec
bf_plot(data, sorted_log_BF[star_rank])

# ╔═╡ 79974a36-2a53-4495-a460-1a3403528192
selected_log_BFs = sorted_log_BF[1:star_rank]

# ╔═╡ 42cff42b-5e59-4a55-9c6b-831df83cdfd1
subsample(x, size) = x[round.(Int, range(1, length(x), length=size))]

# ╔═╡ 7ca830ec-8f8a-41c1-b676-5c42dc6ab2de
subsampled_log_BFs = subsample(selected_log_BFs, n_stars)

# ╔═╡ 966f9272-cd74-4861-83b6-b47aa1e5695f
md"""
## Posterior over π (numerical integration)
"""

# ╔═╡ c194518c-5482-441f-93c3-42662bd83e98
eps=0.001

# ╔═╡ 6ac4beac-d01f-494e-82f0-45114ece3be9
lines(0.0:eps:1.0, marginal_pi_posterior_density(eps, subsampled_log_BFs))

# ╔═╡ 089a4738-1308-4308-b754-a264af339e20
md"""
## Posterior using MCMC
"""

# ╔═╡ 7f3dadd7-c885-479e-91d5-c2bef47e29b9
@assert length(unique(sorted_log_BF)) == length(sorted_log_BF)

# ╔═╡ 33e71a55-07e0-45a8-ba09-b6bb7ad8bf79
selected_indices = find_selected_indices(log_BF(data), subsampled_log_BFs)

# ╔═╡ e1cbce0b-de9a-4d13-b529-ab9fc94a3632
subsampled_data = subset(selected_indices, data)

# ╔═╡ 9129a7da-17a1-4a3c-aac2-2a19bfd1f130
chains = run_turing(subsampled_data)

# ╔═╡ b449658e-7f69-4713-a5cb-24105464191d
mcmc_pi_samples = vec(Array(chains[:, :π, :]))

# ╔═╡ 5c2bf009-af27-49a3-bb21-af8a281f713f
begin 
	fig = Figure()
	ax  = Axis(fig[1,1], limits = ((0.0, 1.0), nothing))
	hist!(ax, mcmc_pi_samples; bins = 50)
	fig
end

# ╔═╡ 0ee6d58f-db4c-47cc-a270-f5945b22fb96
lines(mcmc_pi_samples)

# ╔═╡ Cell order:
# ╠═ba56fbca-8edf-4e04-916b-ed02b967bd11
# ╟─05993f6d-d590-4889-9e96-5cb23f84ecf0
# ╠═3eeac7d7-dfca-4d9c-be4b-805615782c88
# ╠═f8483d3c-3f88-4a65-92e2-036befa0440d
# ╠═87fbed8c-582a-4d70-9bf6-ff565c41e8c4
# ╟─049caba4-831a-47ae-b3e7-85bc00e93f5a
# ╠═519504b3-dc75-40ae-b482-68c5b35f1676
# ╠═1fdca356-c8d7-4b57-9cb1-55939ce9241b
# ╠═eb7b764e-156d-4673-8314-cdedd8a9e6ec
# ╟─79974a36-2a53-4495-a460-1a3403528192
# ╟─42cff42b-5e59-4a55-9c6b-831df83cdfd1
# ╟─7ca830ec-8f8a-41c1-b676-5c42dc6ab2de
# ╟─966f9272-cd74-4861-83b6-b47aa1e5695f
# ╠═6ac4beac-d01f-494e-82f0-45114ece3be9
# ╠═c194518c-5482-441f-93c3-42662bd83e98
# ╟─089a4738-1308-4308-b754-a264af339e20
# ╠═5c2bf009-af27-49a3-bb21-af8a281f713f
# ╠═7f3dadd7-c885-479e-91d5-c2bef47e29b9
# ╠═33e71a55-07e0-45a8-ba09-b6bb7ad8bf79
# ╠═e1cbce0b-de9a-4d13-b529-ab9fc94a3632
# ╠═9129a7da-17a1-4a3c-aac2-2a19bfd1f130
# ╠═b449658e-7f69-4713-a5cb-24105464191d
# ╠═0ee6d58f-db4c-47cc-a270-f5945b22fb96
