"""
$SIGNATURES
"""
function run_imh(rng::AbstractRNG, binned::MultiBinnedIndepRuns, hyperprior = (level -> Dirac(level^2)), processor = (processor_context -> nothing)) 
	multiproposal = [b.samples for b in binned.multibinnedruns]
	n_systems, n_iters = size(multiproposal[end])
	multistate = [copy(p[:,1]) for p in multiproposal]
	n_levels = length(binned.multibinnedruns)
	max_n_comp = binned.multibinnedruns[end].max_n_companions
	n_bins_max = binned.multibinnedruns[end].binning.n_bins 
	tilde_psi = binned.multibinnedruns[end].tilde_psi
	alphas = zeros(n_levels)
	for n in 1:n_levels
		@assert minimum(hyperprior(n)) >= 0
		alphas[n] = rand(rng, hyperprior(n))
	end
    
    psi_trace = zeros(max_n_comp + 1, n_iters - 1) 
    pi_trace = zeros(n_bins_max, n_iters - 1)
    alpha_trace = zeros(n_levels, n_iters - 1)
    accept_prs = zeros(n_systems)

    for iter in 2:n_iters
        # psi, pi, alphas | rest
		multipi = Vector{Vector{Float64}}()
        for n in 1:n_levels
        	n_bins = binned.multibinnedruns[n].binning.n_bins
        	partition_sizes = binned.multibinnedruns[n].binning.partition_sizes
			# using the property of the Dirichlet that (normalized) marginals are still Dirichlet, 
			# we can sample the whole level of the hierarchy at once and normalize quartets within branches
			total_companion_counts, bin_membership_counts = gather_counts(multistate[n], max_n_comp, n_bins)
			push!(multipi, rand(rng, Dirichlet(alphas[n] .+ bin_membership_counts)))
			if n == 1
				# only need to do this once
        		psi = rand(rng, Dirichlet(1. .+ total_companion_counts))
				# at the top level alphas[1], no need to modify pi before sampling from conditional
				alphas[n] = sample_alpha(rng, alphas[n], multipi[n], hyperprior(n), bin_membership_counts)
			else
				normalize_quartets!(multipi[n], partition_sizes)
				alphas[n] = sample_alpha_quartets(rng, alphas[n], multipi[n], partition_sizes, hyperprior(n), bin_membership_counts)
			end
		end

		# compute total pi vector at the bottom based on multipi
		pi = aggregate_multipi(multipi, binned)

		# sample new planet counts, memberships at the bottom layer
		accepts = sample_systems!(rng, state_hierarchy[end], accept_prs, @view(multiproposal[end][:, iter]), tilde_psi, psi, pi)
		# propagate the same accept/reject decisions to other layers of the hierarchy
		propagate_accepts!(state_hierarchy, multiproposal, accepts, iter)

        # collect samples
        psi_trace[:, iter - 1] = psi 
        pi_trace[:, iter - 1] = pi
        alpha_trace[:, iter - 1] = alpha

        processor_context = (; iter, psi, pi, alphas, states, total_companion_counts, bin_membership_counts)
        processor(processor_context)
    end
    accept_prs ./= (n_iters - 1)

    return (; psi_trace, pi_trace, alpha_trace, accept_prs)
end

function propagate_accepts!(state_hierarchy, proposal_hierarchy, accepts, iter)
	system_indices = eachindex(state_hierarchy[end])
	@assert system_indices == eachindex(proposal_hierarchy[end])
	for n in 1:length(state_hierarchy)-1
		for s in system_indices
			if accepts[s]
				state_hierarchy[n][s] = proposal_hierarchy[n][s, iter]
			end
		end
	end
end

function normalize_quartets!(pi, partition_sizes)
	idcs = LinearIndices(partition_sizes)
	for i in 1:2:partition_sizes[1]
		for j in 1:2:partition_sizes[2]
			i1 = idcs[i,j]
			i2 = idcs[i+1,j]
			i3 = idcs[i,j+1]
			i4 = idcs[i+1,j+1]
			pi[[i1,i2,i3,i4]] ./= sum(pi[[i1,i2,i3,i4]])
		end
	end
end

function logp_alpha_quartets(alpha, pi, partition_sizes, hyperprior, bin_membership_counts)
	lp = logpdf(hyperprior, alpha)
	idcs = LinearIndices(partition_sizes)
	for i in 1:2:partition_sizes[1]
		for j in 1:2:partition_sizes[2]
			i1 = idcs[i,j]
			i2 = idcs[i+1,j]
			i3 = idcs[i,j+1]
			i4 = idcs[i+1,j+1]
			lp += logpdf(Dirichlet(alpha .+ bin_membership_counts[[i1,i2,i3,i4]]), pi[[i1,i2,i3,i4]])
		end
	end
	return lp
end

function sample_alpha_quartets(rng, alpha, pi, partition_sizes, hyperprior, bin_membership_counts)
	# initial logprob computation with current alpha
	lp = logp_alpha_quartets(alpha, pi, partition_sizes, hyperprior, bin_membership_counts)
	# sweep over a few reasonable scales with random walk MH
    for s in -3:2
    	alphap = alpha + 10.0^s * randn(rng)
		lpp = logp_alpha_quartets(alphap, pi, partition_sizes, hyperprior, bin_membership_counts)
    	if log(rand(rng)) <= lpp - lp
    		alpha = alphap
    		lp = lpp
    	end
    end
    return alpha
end

function aggregate_multipi(multipi, binned)
	n_levels = length(binned.multibinnedruns)
	pi = one(multipi)
	bottompartition = binned.multibinnedruns[end].binning.partition_sizes
	bottomidx = LinearIndices(bottompartition)
	for i in 1:bottompartition[1]
		for j in 1:bottompartition[2]
			icur = i
			jcur = j
			for n in n_levels:-1:1
				curidx = LinearIndices(binned.multibinnedruns[n].binning.partition_sizes)
				pi[bottomidx[icur, jcur]] *= multipi[n][curidx[icur, jcur]]
				icur = (icur+1) ÷ 2
				jcur = (jcur+1) ÷ 2	
			end
		end
	end
	@assert abs(sum(pi) - 1.0) < 1e-6
	return pi/sum(pi)
end
