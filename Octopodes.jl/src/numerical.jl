

numerical(binned::BinnedIndepRuns, eps) = 
    numerical(
        local_companionship_posteriors(binned), 
        binned.tilde_psi, 
        Uniform(0.0, 1.0), 
        eps,
        Float64
    )

local_companionship_posteriors(binned::BinnedIndepRuns) = vec(mean(sample -> sample.n_companions, binned.samples, dims = 2))

function numerical(local_companionship_posteriors::Vector, tilde_psi::Vector, psi_prior::Distribution, eps::Real, ::Type{T}) where {T}
    @assert eps > 0 
    @assert all(0 .<= local_companionship_posteriors .<= 1) 
    @assert length(tilde_psi) == 2 && sum(tilde_psi) ≈ 1 "Incorrect tilde_psi: $tilde_psi"
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
            sum += log_BFs[system_index] == Inf ? log(psi) : logsumexp(log(psi) + log_BFs[system_index], log1p(-psi))
        end
        result[posterior_discretization_index] = sum
    end
    exp_normalize!(result)
    return result*(length(grid)+1) # Turn the PMF into a density
end

"""
Produce a coarser binning based on the indicator function of whether there is 
at least one planet (irrespective of its position).
"""
function binarize(binned::BinnedIndepRuns)
    pr_no_companion = binned.tilde_psi[1]
    tilde_psi = [pr_no_companion, 1.0 - pr_no_companion]
    samples = map(binarize, binned.samples) 
    b = collapse(binned.binning)
    return BinnedIndepRuns(b, samples, tilde_psi, 1, binned.star_names)
end

binarize(s::BinnedSample) = BinnedSample{Tuple{Int64}}(s.n_companions, (s.n_companions,))

collapse(grid::StepRangeLen) = range(minimum(grid), maximum(grid), 2)
collapse(b::Binning) = Binning(collapse(b.log_P_yr_grid), collapse(b.log_q_grid), (1, 1), 1)

function exp_normalize!(log_weights)
    m = maximum(log_weights)
    log_weights .= exp.(log_weights .- m) 
    return m + log(normalize!(log_weights))
end 

function normalize!(weights) 
    s = sum(weights)
    weights .= weights ./ s 
    return s
end