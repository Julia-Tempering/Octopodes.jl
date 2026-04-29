"""
$SIGNATURES
"""
function run_imh(rng::AbstractRNG, binned::BinnedIndepRuns, n_levels::Int=1, hyperprior = (level -> Dirac(level^2)), processor = (processor_context -> nothing)) 
	@assert iszero(binned.binning.partition_sizes % 2^(n_levels-1)) "The number of bins in each dimension must be divisible by 2^(n_levels-1)"
    proposals = binned.samples
    n_systems, n_iters = size(proposals)
    states = copy(proposals[:, 1]) # iter 1: initialize with first system trace proposal
    max_n_comp = binned.max_n_companions
    n_bins = binned.binning.n_bins 
    tilde_psi = binned.tilde_psi
	alphas = zeros(n_levels)
	for n in 1:n_levels
		@assert minimum(hyperprior(n)) >= 0
		alphas[n] = rand(rng, hyperprior(n))
	end
    
    psi_trace = zeros(max_n_comp + 1, n_iters - 1) 
    pi_trace = zeros(n_bins, n_iters - 1)
    alpha_trace = zeros(length(alphas), n_iters - 1)
    accept_prs = zeros(n_systems)

    for iter in 2:n_iters
        # psi | rest
        total_companion_counts, bin_membership_counts = gather_counts(states, max_n_comp, n_bins)
        psi = rand(rng, Dirichlet(1. .+ total_companion_counts))

        # pi, alpha | rest, recursive
        # note: alpha resampling happens inside this function, no return alpha needed
		pi = sample_pi_alpha!(rng, alphas, n_levels, vector_to_array(binned.binning, bin_membership_counts), hyperprior)

        # planet counts, memberships | rest
        sample_systems!(rng, states, accept_prs, @view(proposals[:, iter]), tilde_psi, psi, pi)

        # collect samples
        psi_trace[:, iter - 1] = psi 
        pi_trace[:, iter - 1] = pi
        alpha_trace[:, iter - 1] = alphas

        processor_context = (; iter, psi, pi, alphas, states, total_companion_counts, bin_membership_counts)
        processor(processor_context)
    end
    accept_prs ./= (n_iters - 1)

    return (; psi_trace, pi_trace, alpha_trace, accept_prs)
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

    # Assuming a uniform prior here (instead would be `pi ./ tilde_pi`)
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

function logp_alpha(alpha, pi_vecs, hyperprior, bin_count_vecs)
	lp = logpdf(hyperprior, alpha)
	for i in 1:length(pi_vecs)
		lp += logpdf(Dirichlet(alpha .+ bin_count_vecs[i], pi_vecs[i]))
	end
	return lp
end

function sample_alpha(rng, alpha, pi_vecs, hyperprior, bin_count_vecs)
	@assert length(pi_vecs) == length(bin_count_vecs)
	# initial logprob computation with current alpha
	lp = logp_alpha(alpha, pi_vecs, hyperprior, bin_count_vecs)
	# sweep over a few reasonable scales with random walk MH
    for s in -3:2
    	alphap = alpha + 10.0^s * randn(rng)
		lpp = logp_alpha(alphap, pi_vecs, hyperprior, bin_count_vecs)
    	if log(rand(rng)) <= lpp - lp
    		alpha = alphap
    		lp = lpp
    	end
    end
    return alpha
end


function sample_pi_alpha!(rng, alphas, level, bin_counts, hyperprior)
	vec_bin_counts = vec(bin_membership_counts)
	if level == 1
		# if at the first level, just do a single flat dirichlet draw, and sample alpha based on that one dirichlet draw
		vecpi = rand(rng, Dirichlet(alphas[level] .+ vec_bin_counts))
		alphas[level] = sample_alpha(rng, alphas[level], [vecpi], hyperprior(level), [vec_bin_counts])
		return vecpi
	else
		# compute the bin membership counts for the level above
		# make sure that the bin counts grid is divisible by 2 to coarsen it
		@assert iszero(size(bin_counts,1) % 2)
		@assert iszero(size(bin_counts,2) % 2)
		bin_counts_above = zero(size(bin_counts) .÷ 2)
		for i in 1:size(bin_counts,1)
			for j in 1:size(bin_counts,2)
				bin_counts_above[(i+1)÷2,(j+1)÷2] += bin_counts[i,j]
			end
		end
		# draw the pi and alpha from one level up
		pi_above = reshape(sample_pi_alpha!(rng, alphas, level-1, bin_counts_above, hyperprior), size(bin_counts_above))
		# sample this level's pi and alpha
		# due to the properties of the dirichlet, we can sample the current pi all at once and adjust within quartet blocks
		pi = reshape(rand(rng, Dirichlet(alphas[level] .+ vec_bin_counts)), size(bin_counts))
		pi_quad_vecs = []
		count_quad_vecs = []
		for i in 1:2:size(bin_counts,1)
			for j in 1:2:size(bin_counts,2)
				# normalize each quadrant
				pi[i:i+1,j:j+1] ./= sum(pi[i:i+1,j:j+1])
				# store each quadrant separately for sampling alpha (need a copy for pi since about to overwrite)
				push!(pi_quad_vecs, copy(vec(pi[i:i+1,j:j+1])))
				push!(count_quad_vecs, vec(bin_counts[i:i+1,j:j+1]))
				# rescale pi for this layer based on higher level
				pi[i:i+1,j:j+1] .*= pi_above[(i+1)÷2, (j+1)÷2]
			end
		end
		# after all of this, pi should sum to 1 at each level
		@assert abs(sum(pi) - 1) < 1e-6
		# resample alpha
		alphas[level] = sample_alpha(rng, alphas[level], pi_quad_vecs, hyperprior, count_quad_vecs)
		return vec(pi)
	end
end
