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

# ╔═╡ a797d8cc-c502-11f0-3617-f901f90e0970
begin
    import Pkg
    Pkg.activate(Base.current_project())
    
	using JLD2, CairoMakie, LogExpFunctions, Pigeons, DynamicPPL, Distributions, 
    Turing, Enzyme, LogDensityProblems, LogDensityProblemsAD, ForwardDiff, 
    LinearAlgebra, Random, BenchmarkTools, MCMCChains, Pluto, PlutoUI
	include("main.jl")
end 

# ╔═╡ b9d8a86d-070a-4a20-839b-e152bc4f32a3
md"""

## Goal

We consider the semi-hierchical model, i.e., sharing only the planet prevalence parameter $\pi$ across stars. 

The goal is to investigate the sensitivity of the posterior mean of $\pi$ with respect to the Monte Carlo noise of the individual octofitter runs. 

Since $\pi$ is univariate, for a given set of octofitter posteriors, we can compute the posterior of the semi-hierchical model using numeric integration. This is a deterministic function of the octofitter posteriors, so we can do sensitivity analysis using the delta method. 

Before going into the math, it is useful to first qualitatively explore manual subsampling of the octofitter input and/or number of stars. 

As a preview of what follows, we also show the radius of the 95% confidence interval for the posterior mean, which we derive below.
"""

# ╔═╡ 002daebe-c270-40ee-8f0b-58670068c4c0
 mcmc_subsample = @bind mcmc_subsample PlutoUI.Slider(10:(size(assignments)[2]))

# ╔═╡ 95764e0b-ed3b-45ea-b34e-04611c635dc1
stars_subsample = @bind stars_subsample PlutoUI.Slider(10:300)

# ╔═╡ 9a7e5cba-ac46-49a7-8748-625ef003217b
md"""
It should be obvious from interaction with this widget that the posterior of the semi-hierchical model is robust to the Monte Carlo noise present in the octo-fitter runs. Nonetheless, the notes below derive a delta method confidence interval to quantify this.
"""

# ╔═╡ 082897eb-ec10-4726-9388-bdafcd197b60
md"""
## Derivation

We start with Markov chain CLT:

$$\sqrt{n} (F - f) \to N(0, A),$$

where:

- convergence is in distribution in this display and the rest of this note;
- the random variable $F_s = \frac{1}{n} \sum_i X_{s,i}$ is the fraction of the $n$ MCMC samples corresponding to star index $s$ where the planet model is selected;  
- and $A$ is the asymptotic variance, a diagonal matrix (since the MCMC runs are independent). 

We load the matrix $X$ (`bin-assignments-all-v0.0.jld2`, sent on Nov 18), and compute $F$ from it:
"""

# ╔═╡ a909ca64-05ac-45ab-a145-13f69830cf80
X = assignments[1:stars_subsample, 1:mcmc_subsample]

# ╔═╡ 8aa465a7-b5bc-4cc1-b5d2-1df91a51c0c1
n_stars, n_mcmc_iters = size(X)

# ╔═╡ 6b543109-fe14-46d5-8169-184c02c36e99
let (fig, ax, _) = heatmap(X)
	ax.xlabel = "star"
	ax.ylabel = "MCMC iteration"
	fig
end

# ╔═╡ 3596afaa-eea7-4e90-bd91-8abd566e8fc3
F = vec(sum(X, dims = 2) .+ 1) / (n_mcmc_iters + 2)

# ╔═╡ 1de999c8-505c-4d0b-a6c9-742c5993cf79
let (fig, ax, _) = lines(sort(F)) 
	ax.xlabel = "star (sorted by F)" 
	ax.ylabel = "F"
	fig
end

# ╔═╡ 15ab9b4a-35ad-4714-a661-f7cecf32a065
md"""
## Delta method statement 

If we can write the estimator of interest (posterior mean on $\pi$) as a function $h$ of $F$, we will be able to invoke the delta method:

$$\sqrt{n} (h(F) - h(f)) \to N(0, \Delta),$$

where the asymptotic variance of the transformed expression can be computed with:

$$\Delta = \nabla h(f)^T \cdot A \cdot \nabla h(f).$$

In our case $h: \mathbb{R}^S \to \mathbb{R}$ where $S$ is the number of stars.

From this, a 95% confidence interval can be constructed with a radius given by:

$$\text{CI radius} = 1.96 \sqrt{\frac{\Delta}{n}}$$.

"""

# ╔═╡ 34e9bab5-93fc-4796-b072-9a7c3405a780
md"""

## Writing the estimator as a function of $F$

We start by fixing a discretization mesh size for the numerical integration used within Bayes rule:
"""

# ╔═╡ 15063428-7a0c-477b-b46d-b030604b8a02
eps = 0.01

# ╔═╡ c92b7228-8156-492c-88ae-cbabfb6db7fc
md"""
Next, we write posterior mean as a function $h$ of $F$, in order to be able to apply the delta method. 

To do this, we have that $h$ is the composition of:

1. Transforming the `planet_probabilities` into `log_BFs`
2. Computing the hierarchical posterior using numerical integration
3. Computing the mean of that density.
"""

# ╔═╡ 134b1d8e-aeea-4afa-8126-62b72f3b8003
h(planet_probabilities) = numerical_mean(eps, semi_hierarchical_pi_posterior_density(eps, log_BF(planet_probabilities)))

# ╔═╡ b9b29f2b-3dc7-4859-8585-87fda6a349aa
estimator = h(F)

# ╔═╡ 10dc85e0-8ca2-4b78-a5a0-5ed614d967c0
posterior = semi_hierarchical_pi_posterior_density(eps, log_BF(F))

# ╔═╡ f403a3f1-1a02-474b-a78a-8c895817dab7
let (fig, ax, _) = lines(eps:eps:(1.0-eps), posterior)
	vlines!(ax, [estimator], color=:red, linestyle=:dash)
	ax.xlabel = "π (red dashed line shows E[π])"
	fig
end

# ╔═╡ fe2c7f84-223c-4cb2-a0c3-6eb51c53e4c1
md"""

## Computing the gradient using autodiff
"""

# ╔═╡ bed2a0c3-cbe9-41a3-ab5e-97c0df606cd2
grad = begin
	dx = zeros(length(F)) 
	Enzyme.autodiff(Enzyme.Reverse, h, Enzyme.Active, Enzyme.Duplicated(F, dx))
	dx
end

# ╔═╡ 4e0eee92-c653-4a54-8d3e-28b513e20af8
md"""
## Estimating the asymptotic variances

Let $n_e$ denote the Effective Sample Size (ESS). We first compute it based on Geyer's estimator and the FFT:
"""

# ╔═╡ c5775444-f4ae-4e7e-9857-488b6c27fe6a
n_e = ess(X)

# ╔═╡ 7f7f2202-b69f-4279-8582-9c9bcb3f9843
let (fig, ax, _) = lines(sort(n_e))
	ax.xlabel = "star (sorted by ESS)" 
	ax.ylabel = "ESS"
	fig
end

# ╔═╡ f9a71bc8-55ec-47dd-9278-a2024c5c7384
md"""

Next, we use the following relationship between asymptotic variance $\sigma_a$ (diagonal entries of $A$), variance $\sigma$, $n$ and $n_e$:

$$\sigma_a = \frac{\sigma^2 n}{n_e}.$$

As for $\sigma^2$, since the matrix $X$ is binary, it can be estimated with $F (1 - F)$. 

(The symbols in the above paragraph should be understood to have an implicit star: subscript $s$). 

Substituting into the expression for the radius of the confidence interval, we obtain 

$$\begin{align*}
\text{CI radius} &= 1.96 \sqrt{\frac{\Delta}{n}} \\ 
&= 1.96 \sqrt{\frac{\sum_s (\nabla_s h(F))^2 \left(\frac{\sigma_s^2 n}{(n_e)_s}\right)}{n}} \\ 
&= 1.96 \sqrt{\sum_s (\nabla_s h(F))^2 \left(\frac{\sigma_s^2}{(n_e)_s}\right)}.
\end{align*}$$
"""

# ╔═╡ 5a89fe33-ae68-4422-87fa-ff1da08b59b5
radius = begin
	s = 0.0 
	for star_index in 1:n_stars 
		n_e_s = isnan(n_e[star_index]) ? n_mcmc_iters : n_e[star_index]
		sigma2_s = F[star_index] * (1.0 - F[star_index]) 
		s += grad[star_index]^2 * sigma2_s / n_e_s
	end
	1.96 * sqrt(s)
end

# ╔═╡ 84c8cc6e-8f13-497c-b91a-0096dd7f8d79
radius

# ╔═╡ Cell order:
# ╟─a797d8cc-c502-11f0-3617-f901f90e0970
# ╟─b9d8a86d-070a-4a20-839b-e152bc4f32a3
# ╠═8aa465a7-b5bc-4cc1-b5d2-1df91a51c0c1
# ╠═84c8cc6e-8f13-497c-b91a-0096dd7f8d79
# ╟─002daebe-c270-40ee-8f0b-58670068c4c0
# ╟─95764e0b-ed3b-45ea-b34e-04611c635dc1
# ╠═f403a3f1-1a02-474b-a78a-8c895817dab7
# ╟─9a7e5cba-ac46-49a7-8748-625ef003217b
# ╟─082897eb-ec10-4726-9388-bdafcd197b60
# ╠═a909ca64-05ac-45ab-a145-13f69830cf80
# ╟─6b543109-fe14-46d5-8169-184c02c36e99
# ╠═3596afaa-eea7-4e90-bd91-8abd566e8fc3
# ╟─1de999c8-505c-4d0b-a6c9-742c5993cf79
# ╟─15ab9b4a-35ad-4714-a661-f7cecf32a065
# ╟─34e9bab5-93fc-4796-b072-9a7c3405a780
# ╟─15063428-7a0c-477b-b46d-b030604b8a02
# ╟─c92b7228-8156-492c-88ae-cbabfb6db7fc
# ╠═134b1d8e-aeea-4afa-8126-62b72f3b8003
# ╠═b9b29f2b-3dc7-4859-8585-87fda6a349aa
# ╠═10dc85e0-8ca2-4b78-a5a0-5ed614d967c0
# ╟─fe2c7f84-223c-4cb2-a0c3-6eb51c53e4c1
# ╠═bed2a0c3-cbe9-41a3-ab5e-97c0df606cd2
# ╟─4e0eee92-c653-4a54-8d3e-28b513e20af8
# ╠═c5775444-f4ae-4e7e-9857-488b6c27fe6a
# ╟─7f7f2202-b69f-4279-8582-9c9bcb3f9843
# ╟─f9a71bc8-55ec-47dd-9278-a2024c5c7384
# ╠═5a89fe33-ae68-4422-87fa-ff1da08b59b5
