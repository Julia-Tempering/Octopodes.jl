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

# ╔═╡ db7a3852-bbe9-11f0-0699-739e8aaf1015
begin
    import Pkg
    Pkg.activate(Base.current_project())
    
	using JLD2, CairoMakie, LogExpFunctions, Pigeons, DynamicPPL, Distributions, 
    Turing, Enzyme, LogDensityProblems, LogDensityProblemsAD, ForwardDiff, 
    LinearAlgebra, Random, BenchmarkTools, MCMCChains, Pluto, PlutoUI
	include("main.jl")
end 

# ╔═╡ 2b62f275-fb55-4fff-b995-a9aa5393a752
sorted_BF = sort(BF)

# ╔═╡ 6315055b-9a18-4a0f-857d-ce075586bf3d
@bind star_rank PlutoUI.Slider(1:length(data.log_Ni))

# ╔═╡ b71890ee-013f-4511-bea0-3b69bb298ee4
bf_plot(sorted_BF[star_rank]) 

# ╔═╡ 87125a02-e4b3-4ef8-8cdb-d094ec56c8c3
md"""
Let $M_i$ denote the mass bin $i$, where $M_0$ is the no planet event. We show here the marginal likelihoods $p(y | M_i)$ up to a proportionality constant. 
"""

# ╔═╡ 79298702-42f2-49b9-8848-8e827fcc10a4
lines(0:5, mass_histogram(index_of_ranks[star_rank]))

# ╔═╡ Cell order:
# ╠═db7a3852-bbe9-11f0-0699-739e8aaf1015
# ╠═2b62f275-fb55-4fff-b995-a9aa5393a752
# ╠═6315055b-9a18-4a0f-857d-ce075586bf3d
# ╠═b71890ee-013f-4511-bea0-3b69bb298ee4
# ╟─87125a02-e4b3-4ef8-8cdb-d094ec56c8c3
# ╠═79298702-42f2-49b9-8848-8e827fcc10a4
