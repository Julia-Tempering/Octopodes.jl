const default_eps = 0.0001

"""
$(SIGNATURES)

Assume that the provided `binned` is binary (see [`is_binary`](@ref)) and 
use numerical integration to obtain a discretized posterior. 

Use a uniform prior over the fraction of stars with at least one companion.
"""
numerical(binned::BinnedIndepRuns, eps = default_eps) =
    numerical(
        local_companionship_posteriors(binned), 
        binned.tilde_psi, 
        Uniform(0.0, 1.0), 
        eps,
        Float64
    )

numerical_joint_prediction(binned::BinnedIndepRuns, eps = default_eps) = 
    numerical_joint_prediction(
        local_companionship_posteriors(binned),
        binned.tilde_psi,
        Uniform(0.0, 1.0), 
        eps, 
        Float64
    )

"""
$SIGNATURES

A [`BinnedIndepRuns`](@ref) is binary if the maximum number of companions is one 
and the number of bins is one. 

See also [`binarize`](@ref) for two methods to get a binary binning from a non 
binary one (either by just looking at overall presence of at least on companion, or 
by focusing single bin).  
"""
is_binary(binned::BinnedIndepRuns) = binned.max_n_companions == 1 && binned.binning.n_bins == 1

function local_companionship_posteriors(binned::BinnedIndepRuns) 
    @assert is_binary(binned)
    return vec(mean(sample -> sample.n_companions, binned.samples, dims = 2))
end

to_logBF(local_companionship_posterior, tilde_psi) = 
    log(local_companionship_posterior) -
    log1p(-local_companionship_posterior) + 
    log1p(-tilde_psi) - 
    log(tilde_psi)

companion_probability(log_BF) = 1 / (1 + exp(-log_BF))

function numerical(local_companionship_posteriors::AbstractVector, tilde_psi::AbstractVector, psi_prior::Distribution, eps::Real, ::Type{T}) where {T}
    @assert eps > 0 
    @assert all(0 .<= local_companionship_posteriors .<= 1) 
    @assert length(tilde_psi) == 2 && sum(tilde_psi) ≈ 1 
    @assert Distributions.value_support(typeof(psi_prior)) == Distributions.Continuous && minimum(psi_prior) == 0 && maximum(psi_prior) == 1

    log_BFs = to_logBF.(local_companionship_posteriors, tilde_psi[2]) 

    grid = build_grid(eps) 
    result = zeros(T, length(grid))

    for posterior_discretization_index in eachindex(result)
        psi = grid[posterior_discretization_index]
        sum = logpdf(psi_prior, psi)
        
        for system_index in eachindex(log_BFs) 
            sum += log_BFs[system_index] == Inf ? log(psi) : logaddexp(log(psi) + log_BFs[system_index], log1p(-psi))
        end
        result[posterior_discretization_index] = sum
    end
    exp_normalize!(result)
    return result*(length(grid)+1) # Turn the PMF into a density
end

function standardized_local_posteriors(binned::BinnedIndepRuns)
    # raw local posterior might use a prior different than 1/2, 'standarize' it
    raw_local_posteriors = local_companionship_posteriors(binned)
    log_BFs = to_logBF.(raw_local_posteriors, binned.tilde_psi[2]) 
    return sort(companion_probability.(log_BFs))
end

build_grid(eps::Real) = eps:eps:(1.0-eps)
build_grid(posterior::AbstractVector) = build_grid(eps(posterior))

eps(n::Int) = 1.0 / (n + 1)
eps(posterior::AbstractVector) = eps(length(posterior))

numerical_mean(posterior) = numerical_mean(identity, posterior)
function numerical_mean(test_function, posterior::AbstractVector{T}) where {T}
    psis = build_grid(posterior) 
    eps = Octopodes.eps(posterior)
    sum = zero(T)
    for posterior_discretization_index in eachindex(posterior)
        sum += eps * test_function(psis[posterior_discretization_index]) * posterior[posterior_discretization_index]
    end
    return sum
end
function numerical_joint_prediction(local_companionship_posteriors::AbstractVector, tilde_psi::AbstractVector, psi_prior::Distribution, eps::Real, ::Type{T}) where {T}
    posterior = numerical(local_companionship_posteriors, tilde_psi, psi_prior, eps, T)
    BFs = exp.(to_logBF.(local_companionship_posteriors, tilde_psi[2]))
    return map(BFs) do BF 
        test_fct(psi) = 1/(1 + (1 - psi) / psi / BF)
        numerical_mean(test_fct, posterior)
    end
end


"""
$SIGNATURES

Produce a coarser binning based on the indicator function of whether there is 
at least one planet (irrespective of its position).
"""
function binarize(binned::BinnedIndepRuns)
    pr_no_companion = binned.tilde_psi[1]
    tilde_psi = [pr_no_companion, 1.0 - pr_no_companion]
    function binarize(s::BinnedSample)
        has_companion = UInt8(s.n_companions > 0 ? 1 : 0)
        
        return BinnedSample(has_companion, (has_companion,), s.indep_trace_index)
    end
    samples = map(binarize, binned.samples) 
    b = collapse(binned.binning)
    return BinnedIndepRuns(b, samples, tilde_psi, 1, binned.star_names)
end

"""
$SIGNATURES

Produce a coarser binning based on the indicator function of whether there is 
at least one planet in bin index `k`.
"""
function binarize(binned::BinnedIndepRuns, k::Int)
    K = binned.binning.n_bins 
    @assert K > 1 
    @assert 1 ≤ k ≤ K 

    # Prior probability of having no companion in bin k. 
    #  = E[P(no companion in bin k | number of companions)]
    # NOTE: Assuming a uniform prior here
    pr_no_companion = sum(c -> binned.tilde_psi[c] * ((K - 1.0)/K)^(c-1), eachindex(binned.tilde_psi))
    @assert 0 < pr_no_companion < 1

    tilde_psi = [pr_no_companion, 1.0 - pr_no_companion]
    
    function binarize(s::BinnedSample)
        has_companion_in_given_bin = UInt8(any(==(k), s.bin_memberships))
        return BinnedSample(has_companion_in_given_bin, (has_companion_in_given_bin,), s.indep_trace_index)
    end
    samples = map(binarize, binned.samples) 

    # keep track of the bin we reduced to 
    b = intervals_from_bin_index(binned.binning, k)
    return BinnedIndepRuns(b, samples, tilde_psi, 1, binned.star_names)
end

collapse(grid::StepRangeLen) = range(minimum(grid), maximum(grid), 2)
collapse(b::Binning) = Binning(collapse(b.log_P_yr_grid), collapse(b.log_q_grid), (1, 1), 1)

"""
$SIGNATURES 


"""
function compare_numerical_imh(rng, binned; 
        # The KS test assumes iid samples, so we need thin+burn-in
        thinning = 5, 
        burn_fraction = 0.5) 
    
    @assert is_binary(binned)
    true_posterior = numerical(binned)
    num_cdf_xs, num_cdf_ys = discrete_pdf_to_cdf(true_posterior) 

    imh_results = run_imh(rng, binned)
    _, n_iters = size(imh_results.psi_trace)
    first = ceil(Int, n_iters * burn_fraction)
    companion_presence_probability_samples = vec(imh_results.psi_trace[2, first:thinning:n_iters])
    imh_cdf_xs, imh_cdf_ys = samples_to_cdf(companion_presence_probability_samples)

    ks_dist = ks_distance(num_cdf_xs, num_cdf_ys, imh_cdf_xs, imh_cdf_ys) 
    ks_p = ks_p_value(ks_dist, length(companion_presence_probability_samples)) 

    return (; 
        true_posterior, 
        true_cdf = (num_cdf_xs, num_cdf_ys), 
        imh_results,
        samples_ecdf = (imh_cdf_xs, imh_cdf_ys),
        ks_distance = ks_dist,
        ks_p_value = ks_p
    )
end

function compare_numerical_imh_plot(compare_numerical_imh_results)
    (num_cdf_xs, num_cdf_ys) = compare_numerical_imh_results.true_cdf 
    (imh_cdf_xs, imh_cdf_ys) = compare_numerical_imh_results.samples_ecdf

    fig, ax = Makie.lines(num_cdf_xs, num_cdf_ys; label = "Numerical CDF")
    Makie.lines!(ax, imh_cdf_xs, imh_cdf_ys; label = "IMH ECDF")

    ax.xlabel = L"\psi"
    ax.ylabel = L"F_\psi(\cdot | y)"
    Makie.axislegend(ax, position = :lt)

    return fig
end