"""
$(SIGNATURES)

Assume that the provided `binned` is binary (see [`is_binary`](@ref)) and 
use numerical integration to obtain a discretized posterior. 

Use a uniform prior over the fraction of stars with at least one companion.
"""
numerical(binned::BinnedIndepRuns, eps = 0.001) =
    numerical(
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

function numerical(local_companionship_posteriors::Vector, tilde_psi::Vector, psi_prior::Distribution, eps::Real, ::Type{T}) where {T}
    @assert eps > 0 
    @assert all(0 .<= local_companionship_posteriors .<= 1) 
    @assert length(tilde_psi) == 2 && sum(tilde_psi) ≈ 1 
    @assert Distributions.value_support(typeof(psi_prior)) == Distributions.Continuous && minimum(psi_prior) == 0 && maximum(psi_prior) == 1

    log_tilde_psi_ratio = log(tilde_psi[1]) - log(tilde_psi[2]) 
    to_logBF(local_companionship_posterior) =  
        log(local_companionship_posterior) -
        log1p(-local_companionship_posterior) + 
        log_tilde_psi_ratio

    log_BFs = to_logBF.(local_companionship_posteriors) 

    grid = eps:eps:(1.0-eps) 
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

function numerical_mean(eps::Real, posterior::Vector{T}) where {T}
    psis = eps:eps:(1.0-eps) 
    sum = zero(T)
    for posterior_discretization_index in eachindex(posterior)
        sum += eps * psis[posterior_discretization_index] * posterior[posterior_discretization_index]
    end
    return sum
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
        has_companion = s.n_companions > 0 ? 1 : 0
        return BinnedSample{Tuple{Int64}}(has_companion, (has_companion,))
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
        has_companion_in_given_bin = any(==(k), s.bin_memberships)
        return BinnedSample{Tuple{Int64}}(has_companion_in_given_bin, (has_companion_in_given_bin,))
    end
    samples = map(binarize, binned.samples) 

    # keep track of the bin we reduced to 
    b = intervals_from_bin_index(binned.binning, k)
    return BinnedIndepRuns(b, samples, tilde_psi, 1, binned.star_names)
end

collapse(grid::StepRangeLen) = range(minimum(grid), maximum(grid), 2)
collapse(b::Binning) = Binning(collapse(b.log_P_yr_grid), collapse(b.log_q_grid), (1, 1), 1)

function compare_numerical_imh(rng, binned; 
        # The KS test assumes iid samples, so we need thin+burn-in
        thinning = 5, 
        burn_fraction = 0.5) 
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

    fig, ax = Makie.lines(num_cdf_xs, num_cdf_ys; label = "True CDF")
    Makie.lines!(ax, imh_cdf_xs, imh_cdf_ys; label = "IMH ECDF")

    ax.xlabel = L"\psi_{>0}"
    ax.ylabel = L"P(>\psi_{>0})"
    Makie.axislegend(ax, position = :lt)

    return fig
end