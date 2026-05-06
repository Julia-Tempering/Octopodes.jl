function sensitivity(binned::BinnedIndepRuns, eps)
    lcp = local_companionship_posteriors(binned)
    prior(mu) = Beta(mu, 1 - mu) 
    posterior(mu::T) where {T} = numerical_mean(eps,
            numerical(
                lcp,
                binned.tilde_psi,
                prior(mu), 
                eps, 
                T
            )
        )
    return ForwardDiff.derivative(posterior, 0.5)
end

sensitivities(binned::BinnedIndepRuns, eps) = 
    map(1:binned.binning.n_bins) do k 
        binarized = binarize(binned, k) 
        return sensitivity(binarized, eps)
    end