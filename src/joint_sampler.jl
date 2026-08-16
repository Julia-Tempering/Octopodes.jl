"""
$SIGNATURES
"""
function run_imh(rng::AbstractRNG, binned::BinnedIndepRuns; processor = (processor_context -> nothing), n_imh_iters = default_n_imh_iters(binned)) 
    proposals = copy(binned.samples) # copy since we shuffle in place
    n_systems, n_proposals = size(proposals)

    # Note: currently assuming a shuffle is already done, this will change if we stop pre-shfuffleing
    states = copy(@view(proposals[:, 1])) # iter 1: initialize with first system trace proposal

    max_n_comp = binned.max_n_companions
    n_bins = binned.binning.n_bins 
    tilde_psi = binned.tilde_psi
    
    psi_trace = zeros(max_n_comp + 1, n_imh_iters) 
    pi_trace = zeros(n_bins, n_imh_iters)
    states_trace = similar(proposals, (n_systems, n_imh_iters)) 
    accept_prs = zeros(n_systems)

    # Note: IMH iteration i uses proposal at iteration i+1 of the base sampler 
    #       (because sample at base iteration 1 is for initialization)
    proposal_index(imh_iter) = wrapped_index(imh_iter + 1, n_proposals)

    for imh_iter in 1:n_imh_iters

        if imh_iter > 1 &&  # Note: for now, it is already shuffled, so no need to redo, in the future may remove that 
           proposal_index(imh_iter) == 1 # otherwise, need to shuffle when we are about to access the first column of proposals which would be a repeat if we didn't reshuffle
            
           reshuffle!(rng, proposals)
        end

        # psi, pi | rest
        total_companion_counts, bin_membership_counts = gather_counts(states, max_n_comp, n_bins)
        psi = rand(rng, Dirichlet(1. .+ total_companion_counts))
        pi = rand(rng, Dirichlet(1. .+ bin_membership_counts)) 

        # planet counts, memberships | rest
        sample_systems!(rng, states, accept_prs, @view(proposals[:, wrapped_index(imh_iter + 1, n_proposals)]), tilde_psi, psi, pi)

        # collect samples
        psi_trace[:, imh_iter] = psi 
        pi_trace[:, imh_iter] = pi 
        states_trace[:, imh_iter] = states

        processor_context = (; imh_iter, n_imh_iters, psi, pi, states, total_companion_counts, bin_membership_counts)
        processor(processor_context)

    end
    accept_prs ./= n_imh_iters

    return (; psi_trace, pi_trace, states_trace, accept_prs, binning = binned.binning)
end

function default_n_imh_iters(binned)
    proposals = binned.samples
    _, n_iters = size(proposals)
    return n_iters - 1 # <- 1 pass on the base MCMC samples
end

function reshuffle!(rng, proposals)
    n_systems, n_base_mcmc_iter = size(proposals)
    # Note: after benchmarking it does not seem to be worth doing permutedims back and forth here (at least on M2 Pro)
    for s in 1:n_systems 
        shuffle!(rng, @view proposals[s, :])
    end
    return nothing
end

active_companions(s::BinnedSample) = 1:s.n_companions
product_pi(s::BinnedSample, pi_to_pi_tilde_ratios) = prod(i -> pi_to_pi_tilde_ratios[s.bin_memberships[i]], active_companions(s), init = 1.)
accept_pr(current::BinnedSample, proposed::BinnedSample, psi_to_tilde_psi_ratios, pi_to_pi_tilde_ratios) = min(1,
        psi_to_tilde_psi_ratios[proposed.n_companions + 1] / 
        psi_to_tilde_psi_ratios[current.n_companions  + 1] * 
        product_pi(proposed, pi_to_pi_tilde_ratios) / 
        product_pi(current, pi_to_pi_tilde_ratios)
    )
function sample_systems!(rng, states, accept_prs, proposals, tilde_psi, psi, pi)
    system_indices = eachindex(states)
    @assert system_indices == eachindex(proposals)
    
    psi_to_tilde_psi_ratios = psi ./ tilde_psi 

    # NOTE: Assuming a uniform prior here (instead would be `pi ./ tilde_pi`)
    pi_to_pi_tilde_ratios = pi * length(pi)

    for s in system_indices
        pr = accept_pr(states[s], proposals[s], psi_to_tilde_psi_ratios, pi_to_pi_tilde_ratios) 
        accept_prs[s] += pr
        if rand(rng) < pr
            states[s] = proposals[s]
        end
    end
end

function gather_counts(states, max_n_companions::Int, n_bins::Int)
    system_indices = eachindex(states)
    total_companion_counts = zeros(Int, max_n_companions + 1)
    bin_membership_counts = zeros(Int, n_bins)

    for s in system_indices
        state = states[s]
        n_comp = state.n_companions 
        total_companion_counts[n_comp + 1] += 1 
        for c in 1:n_comp 
            bin_membership_counts[state.bin_memberships[c]] += 1
        end
    end
    return total_companion_counts, bin_membership_counts
end
