"""
$(SIGNATURES)
"""
function generate_binary_indep_runs(; 
                psi_some_companion_truth::Float64, 
                n_systems::Int,
                tilde_psi_some_companion::Float64 = 0.5, 
                mcmc_lazy_pr::Float64 = 0.5, 
                n_systems_iters::Int = 1000,
                rng = Xoshiro(1), 
                thinning::Int = 1, 
                shuffle_rng::Union{Nothing, AbstractRNG} = Xoshiro(1)
            )

    @assert 0 ≤ psi_some_companion_truth ≤ 1 
    @assert 0 ≤ tilde_psi_some_companion ≤ 1 
    @assert 0 ≤ mcmc_lazy_pr ≤ 1 
    @assert n_systems > 0 
    @assert n_systems_iters > 0 
    @assert thinning ≥ 1

    x_truth, data = _generate_binary_data(psi_some_companion_truth, n_systems, rng)
    thinned_indices = 1:thinning:n_systems_iters
    n_samples_after_thinning = length(thinned_indices)
    samples = Array{BinnedSample{Tuple{Int64}}}(undef, n_samples_after_thinning, n_systems)
    
    for s in 1:n_systems
        _generate_binary_trace!(@view(samples[:, s]), data[s], tilde_psi_some_companion, mcmc_lazy_pr, rng, thinned_indices, shuffle_rng)
    end
    transposed = permutedims(samples) 

    b = Binning(0.0:1.0:1.0, 0.0:1.0:1.0, (1, 1), 1)
    tilde_psi = [1-tilde_psi_some_companion, tilde_psi_some_companion]
    names = map(s -> "synthetic_$s", 1:n_systems)
    runs = BinnedIndepRuns(b, transposed, tilde_psi, 1, names)
    return (;
        runs, 
        psi_some_companion_truth, 
        x_truth, 
        data, 
    )
end

likelihood(x::Int) = Bernoulli(x == 0 ? 0.0 : 0.5)

function _generate_binary_data(psi_some_companion_truth, n_systems, rng)
    bern = Bernoulli(psi_some_companion_truth)
    x_truth = [rand(rng, bern) ? 1 : 0 for _ in 1:n_systems]
    data = rand.(rng, likelihood.(x_truth))
    return x_truth, data
end 

function _generate_binary_trace!(output, datum::Bool, tilde_psi_some_companion, mcmc_lazy_pr, rng, thinned_indices, shuffle_rng) 
    posterior = [
        (1 - tilde_psi_some_companion) * pdf(likelihood(0), datum), 
        tilde_psi_some_companion       * pdf(likelihood(1), datum)]
    posterior = posterior / sum(posterior) 
    some_comp_bern = Bernoulli(posterior[2])

    n_iterations = maximum(thinned_indices)
    samples_before_thinning = Array{BinnedSample{Tuple{Int64}}}(undef, n_iterations)
    for i in 1:n_iterations 
        self_transition = i == 1 ? false : rand(rng, Bernoulli(mcmc_lazy_pr)) 
        samples_before_thinning[i] = 
            if self_transition 
                samples_before_thinning[i-1] 
            else 
                has_companion = rand(rng, some_comp_bern) ? 1 : 0
                BinnedSample{Tuple{Int64}}(has_companion, (has_companion,))
            end
    end
    thinned_indices = shuffle_if_needed(shuffle_rng, thinned_indices) 
    output[:] = samples_before_thinning[thinned_indices]
    return nothing
end