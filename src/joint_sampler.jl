"""
$SIGNATURES

The dense per-iteration state trace costs `O(n_systems × n_imh_iters)` memory, which for
large runs (many systems and/or many IMH iterations) is by far the dominant allocation.
Because IMH proposals are exactly the system's own (binned) base draws, the post-warmup
states of a system form a multiset over those draws: acceptance counts indexed by
`indep_trace_index`, together with per-system multiplicity counts, are sufficient
statistics for everything the downstream post-processing computes, at
`O(n_systems × n_base_draws)` memory independent of `n_imh_iters`.

By default only these counts are collected (over the post-warmup window implied by
`warmup_frac`, matching [`population_posterior`](@ref)). Pass `store_states_trace = true`
(or `Val(true)`, for a fully inferred return type) to additionally record the dense
trace, needed only for custom statistics via [`joint_reconstructions`](@ref).
"""
function run_imh(rng::AbstractRNG, binned::BinnedIndepRuns; processor = (processor_context -> nothing), n_imh_iters = default_n_imh_iters(binned), warmup_frac::Real = 0.2, store_states_trace::Union{Bool, Val} = Val(false))
    0 ≤ warmup_frac < 1 || throw(ArgumentError("warmup_frac must be in [0, 1), got $warmup_frac"))
    return _run_imh(rng, binned, processor, n_imh_iters, warmup_frac, _to_val(store_states_trace))
end

_to_val(b::Bool) = Val(b)
_to_val(v::Val) = v

function _run_imh(rng::AbstractRNG, binned::BinnedIndepRuns, processor, n_imh_iters::Int, warmup_frac::Real, ::Val{store_states_trace}) where {store_states_trace}
    proposals = copy(binned.samples) # copy since we shuffle in place
    n_systems, n_proposals = size(proposals)

    # Note: currently assuming a shuffle is already done, this will change if we stop pre-shfuffleing
    states = copy(@view(proposals[:, 1])) # iter 1: initialize with first system trace proposal

    max_n_comp = binned.max_n_companions
    n_bins = binned.binning.n_bins
    tilde_psi = binned.tilde_psi

    psi_trace = zeros(max_n_comp + 1, n_imh_iters)
    pi_trace = zeros(n_bins, n_imh_iters)
    states_trace = store_states_trace ? similar(proposals, (n_systems, n_imh_iters)) : nothing
    accept_prs = zeros(n_systems)

    # Post-warmup sufficient statistics of the state trace (same warmup convention as
    # population_posterior: iterations warmup+1 … n_imh_iters are counted).
    @assert n_imh_iters < typemax(UInt32)
    warmup = max(1, floor(Int, warmup_frac * n_imh_iters))
    accept_counts = zeros(UInt32, n_proposals, n_systems)
    multiplicity_counts = zeros(Int, max_n_comp + 1, n_systems)

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
        if store_states_trace
            states_trace[:, imh_iter] = states
        end
        if imh_iter > warmup
            for s in 1:n_systems
                state = states[s]
                # synthetic runs carry indep_trace_index = 0 (no base trace to refer to)
                if state.indep_trace_index > 0
                    accept_counts[state.indep_trace_index, s] += one(UInt32)
                end
                multiplicity_counts[state.n_companions + 1, s] += 1
            end
        end

        processor_context = (; imh_iter, n_imh_iters, psi, pi, states, total_companion_counts, bin_membership_counts)
        processor(processor_context)

    end
    accept_prs ./= n_imh_iters

    return (; psi_trace, pi_trace, states_trace, accept_counts, multiplicity_counts,
              warmup_frac = Float64(warmup_frac), accept_prs, binning = binned.binning)
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
