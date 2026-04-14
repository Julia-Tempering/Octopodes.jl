

function run_imh(rng::AbstractRNG, b::Binning, runs::IndependentMCMCRuns, proposals = bin(b, runs)) 
    n_systems, n_iters = size(proposals)
    states = copy(proposals[:, 1])
    max_n_comp = max_n_companions(runs)
    n_bins = b.n_bins

    # for now, we skip computing tilde_psi since it is uniform so cancel each other in accept ratio
    @assert runs.log_P_yr_prior isa Uniform 
    @assert runs.log_q_prior isa Uniform 
    # we do not make that assumption the prior on the number of companions 
    tilde_psi = map(n -> pmf(run.n_companions_prior, n), 0:max_n_companions)

    psi_trace = zeros(max_n_companions + 1, n_iters - 1) 
    pi_trace = zeros(n_bins, n_iters - 1)
    accept_prs = zeros(n_systems)

    for iter in 2:n_iters
        # psi, pi | rest
        companion_counts, bin_membership_counts = gather_counts(states, max_n_companions, n_bins)
        psi = rand(rng, Dirichlet(1. .+ companion_counts))
        pi = rand(rng, Dirichlet(1. .+ bin_membership_counts)) 

        psi_trace[:, iter - 1] = psi 
        pi_trace[:, iter - 1] = pi

        # planet counts, memberships | rest
        sample_systems!(rng, states, accept_prs, @view(proposals[:, iter]), tilde_psi)
    end

    return (; psi_trace, pi_trace, accept_prs)
end

active_companions(s::BinnedSample) = 1:s.n_companions
product_pi(s::BinnedSample, pi_to_pi_tilde_ratios) = prod(i -> pi_to_pi_tilde_ratios[s.bin_memberships[i]], active_companions(s), init = 1.)
accept_pr(current::BinnedSample, proposed::BinnedSample, psi_to_tilde_psi_ratios, pi_to_pi_tilde_ratios) = min(1,
        psi_to_tilde_psi_ratios[proposed.n_companions + 1] / 
        psi_to_tilde_psi_ratios[current.n_companions  + 1] * 
        product_pi(proposed, pi_to_pi_tilde_ratios) / 
        product_pi(current, pi_to_pi_tilde_ratios)
    )
function sample_systems!(rng::AbstractRNG, states::AbstractVector{BinnedSample}, accept_prs::AbstractVector, proposals::AbstractVector{BinnedSample}, tilde_psi::AbstractVector)
    system_indices = eachindex(states)
    @assert system_indices == eachindex(proposals)
    
    psi_to_tilde_psi_ratios = psi ./ tilde_psi 

    # Assuming a uniform prior here (instead would be `pi ./ tilde_pi`)
    pi_to_pi_tilde_ratios = pi

    for s in system_indices
        pr = accept_pr(states[s], proposals[s], psi_to_tilde_psi_ratios, pi_to_pi_tilde_ratios) 
        accept_prs[s] += pr
        if rand(rng) < pr 
            states[s] = proposals[s]
        end
    end
end

function gather_counts(states::AbstractVector{BinnedSample}, max_n_companions::Int, n_bins::Int)
    system_indices = eachindex(states)
    companion_counts = zeros(Int, max_n_companions + 1)
    bin_membership_counts = zeros(Int, n_bins)

    for s in system_indices
        state = states[s]
        n_comp = state.n_companions 
        companion_counts[n_comp + 1] += 1 
        for c in 1:n_comp 
            bin_membership_counts[state.bin_memberships[c]] += 1
        end
    end
    return companion_counts, bin_membership_counts
end
