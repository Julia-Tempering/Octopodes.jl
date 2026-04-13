
struct Binning{G <: StepRangeLen, S <: Tuple}
    log_P_yr_grid::G
    log_q_grid::G

    # derived quantities for quick access
    partition_sizes::S 
    n_bins::Int 
end
function Binning(log_P_yr_grid, log_q_grid)
    partition_sizes = n_intervals.((log_P_yr_grid, log_q_grid))
    n_bins = prod(partition_sizes)
    return Binning(log_P_yr_grid, log_q_grid, partition_sizes, n_bins)
end

# TODO: this will instead read in the prior objects and just ask for number of bins in each axis
default_binning() = Binning(
    range(log10(1/365.25), log10(10000.0), length=25), 
    range(log10(1e-5),     log10(10.0),     length=21),
)

function bin(b::Binning, values)
    interval_indices = interval_index.((b.log_P_yr_grid, b.log_q_grid), values)
    return LinearIndices(b.partition_sizes)[interval_indices...]
end

vector_to_array(b::Binning, vector::Vector) = reshape(vector, b.partition_sizes)

n_intervals(grid::StepRangeLen) = length(grid) - 1 

"""
Return the index of the intervals in the provided `StepRangeLen` the given point falls in, or 
throw `BoundsError` if the point falls in none of the intervals. 
"""
function interval_index(r::StepRangeLen, value::Real)
    if value == last(r) # last inteval we close on the right
        return length(r) - 1
    end
    result = Int(div(value - first(r), step(r), RoundDown)) + 1
    if !(1 ≤ result < length(r))
        error("Value $value out of range for $r")
    end
    return result
end



### data load (TODO: move to another file)


function bin_system_data(b::Binning, raw_system_data)
    log_P_yr::Matrix{Float64} = raw_system_data.log_P_yr
    log_q::Matrix{Float64} = raw_system_data.log_q
    n_planets::Vector{Int64} = raw_system_data.n_planets 
    n_samples::Int = raw_system_data.n_samples

    samples = Array{SystemSample}(undef, n_samples)
    comp_indices = companion_indices()
    max_n_companions = length(comp_indices)
    buffer = zeros(Int, max_n_companions)
    for iter in 1:n_samples
        n_companions = n_planets[iter]
        for c in 1:max_n_companions
            values = (log_P_yr[c, iter], log_q[c, iter]) 
            buffer[c] = c ≤ n_companions ? bin(b, values) : 0
        end
        bin_memberships = map(i -> buffer[i], comp_indices) 
        samples[iter] = SystemSample(n_companions, bin_memberships)
    end
    return samples
end

function bin_data(b::Binning, raw_data) 
    per_system_vectors = [bin_system_data(b, raw_system_data) for raw_system_data in raw_data]
    permutedims(stack(per_system_vectors))
end

companion_indices() = (1, 2, 3)
struct SystemSample{I <: Tuple}
    n_companions::Int
    bin_memberships::I # warning: this contains inactive ones (for type stability) - the inactive ones are set to zero
end
max_n_companions(sample) = length(sample.bin_memberships)


test_data() = bin_data(default_binning(), JLD2.load("multicomp/multicomp_v7_cache.jld2")["star_data"])

# TODO: add tests to make sure this code stays type stable (it currently is, as of first draft)

# Next: IMH

function imh(system_proposals, tilde_psi, tilde_pi, rng)  
    n_systems, n_iters = size(system_proposals)
    states = system_proposals[:, 1]
    max_n_companions = length(tilde_psi) - 1
    n_bins = length(tilde_pi)

    for iter in 2:n_iters
        companion_counts, bin_membership_counts = gather_counts(states, max_n_companions, n_bins)
        psi = rand(rng, Dirichlet(1. .+ companion_counts))
        pi = rand(rng, Dirichlet(1. .+ bin_membership_counts)) 

        sample_systems!(states, @view(system_proposals[:, iter]), psi, tilde_psi, pi, tilde_pi, rng)
    end
    return nothing
end

function test_imh()
    system_proposals = test_data() 
    tilde_psi = ones(4)
    tilde_pi = ones(default_binning().n_bins)
    rng = MersenneTwister(1)

    @show @timed imh(system_proposals, tilde_psi, tilde_pi, rng) 
    @show @timed imh(system_proposals, tilde_psi, tilde_pi, rng)  
end

active_companions(s::SystemSample) = 1:s.n_companions
product_pi(s::SystemSample, pi_to_pi_tilde_ratios) = prod(i -> pi_to_pi_tilde_ratios[s.bin_memberships[i]], active_companions(s), init = 1.)
accept_pr(current::SystemSample, proposed::SystemSample, psi_to_tilde_psi_ratios, pi_to_pi_tilde_ratios) = min(1,
        psi_to_tilde_psi_ratios[proposed.n_companions + 1] / 
        psi_to_tilde_psi_ratios[current.n_companions  + 1] * 
        product_pi(proposed, pi_to_pi_tilde_ratios) / 
        product_pi(current, pi_to_pi_tilde_ratios)
    )

function sample_systems!(states::AbstractVector{SystemSample}, proposals::AbstractVector{SystemSample}, psi, tilde_psi, pi, tilde_pi, rng)
    system_indices = eachindex(states)
    @assert system_indices == eachindex(proposals)
    
    psi_to_tilde_psi_ratios = psi ./ tilde_psi 
    pi_to_pi_tilde_ratios = pi ./ tilde_pi

    for s in system_indices
        pr = accept_pr(states[s], proposals[s], psi_to_tilde_psi_ratios, pi_to_pi_tilde_ratios) 
        if rand(rng) < pr 
            states[s] = proposals[s]
        end
    end

end

function gather_counts(states::AbstractVector{SystemSample}, max_n_companions, n_bins)
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

