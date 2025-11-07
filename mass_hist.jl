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

# ╔═╡ e932a371-9366-4818-a46a-72a1b258604f
md"""
### Data source
"""

# ╔═╡ 16843a78-2884-4eb8-b74f-4da1826ac4b0
@bind data_source Select([:real, :synthetic])

# ╔═╡ 048a6ebc-9007-4024-b46f-4d69ab50bef1
data = 
	if data_source == :real 
		load_data()
	elseif data_source == :synthetic 
		synt_data 
	else
		error() 
	end

# ╔═╡ 2b62f275-fb55-4fff-b995-a9aa5393a752
sorted_BF = sort(BF(data))

# ╔═╡ 45623653-ad5e-4772-8811-8d27576d7019
md"""
### Star selector (ranked by Bayes Factor)
"""

# ╔═╡ 6315055b-9a18-4a0f-857d-ce075586bf3d
@bind star_rank PlutoUI.Slider(1:length(data.log_Ni))

# ╔═╡ b71890ee-013f-4511-bea0-3b69bb298ee4
bf_plot(data, sorted_BF[star_rank]) 

# ╔═╡ 26b53966-bf90-4bfc-aefa-3df61c5dbcbb
star_index = index_of_ranks(data)[star_rank]

# ╔═╡ 09a1994a-ea5c-4885-83af-42334cd7fba7
md"""
### Marginal likelihood over mass bins (including the zero-mass bin)
"""

# ╔═╡ 87125a02-e4b3-4ef8-8cdb-d094ec56c8c3
md"""
Let $M_k$ denote the mass bin $k$, and $M_0$ is the no planet event. We show here the marginal likelihoods $p(y | M_k)$ up to a proportionality constant. 
"""

# ╔═╡ 79298702-42f2-49b9-8848-8e827fcc10a4
lines(0:5, mass_histogram(star_index, data))

# ╔═╡ c6f34447-453d-481b-b424-9b8da98e876b
md"""
For $p(y | M_0)$, this is just the marginal likehood for the no planet model, i.e., $N_i$.

For $p(y | M_k)$, $k > 0$ we compute it using $M_k \subset M_0^c$ and hence $M_k = M_k \cap M_0^c$ as follows:

```math
\begin{align*}
p(y | M_k) &= \frac{p(M_k, y)}{p(M_k)} \\ 
&= \frac{p(M_k, M_0^c, y)}{p(M_k)} \\ 
&= \frac{p(M_0^c) p(y | M_0^c) p(M_k | M_0^c, y)}{p(M_k)}. 
\end{align*}
```

Each term is computed from the single star models as follows: $p(M_0^c)$ is the prior on the presence of a planet, i.e., $1/2$; $p(y | M_0^c)$ is the marginal likelihood for the with-planet model, i.e., $Z_i$; and $p(M_k | M_0^c, y)$ is obtained from the fraction of draws in the with-planet model in bin $M_k$. 
"""

# ╔═╡ 2032fb51-b4c2-4ab9-9943-20c5ce06fb2b
specific_model(star_index, data)

# ╔═╡ Cell order:
# ╟─db7a3852-bbe9-11f0-0699-739e8aaf1015
# ╟─e932a371-9366-4818-a46a-72a1b258604f
# ╠═16843a78-2884-4eb8-b74f-4da1826ac4b0
# ╟─048a6ebc-9007-4024-b46f-4d69ab50bef1
# ╟─2b62f275-fb55-4fff-b995-a9aa5393a752
# ╟─45623653-ad5e-4772-8811-8d27576d7019
# ╠═2032fb51-b4c2-4ab9-9943-20c5ce06fb2b
# ╟─6315055b-9a18-4a0f-857d-ce075586bf3d
# ╠═b71890ee-013f-4511-bea0-3b69bb298ee4
# ╟─26b53966-bf90-4bfc-aefa-3df61c5dbcbb
# ╟─09a1994a-ea5c-4885-83af-42334cd7fba7
# ╟─87125a02-e4b3-4ef8-8cdb-d094ec56c8c3
# ╠═79298702-42f2-49b9-8848-8e827fcc10a4
# ╟─c6f34447-453d-481b-b424-9b8da98e876b
