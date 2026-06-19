
"""
$(TYPEDEF)

Posterior summary of the joint population model produced by [`run_imh`](@ref),
reduced to the quantities needed to describe and plot companion demographics.

Construct one with [`population_posterior`](@ref). The companion-rate density is
``\\lambda_x = \\mathbb{E}[n]\\,\\pi_x`` — the expected number of companions per
star in bin ``x`` — under the single shared bin distribution ``\\pi`` of the
joint model. Multiply by 100 for the conventional "companions per 100 stars per
bin" units.

$(FIELDS)
"""
struct PopulationPosterior{B <: Binning, S <: BinnedSample}
    """ The [`Binning`](@ref) the posterior is defined on (carries the grid edges). """
    binning::B
    """ Companion-count distribution ``\\psi``. Dims: `(max_n_companions + 1) × n_keep`. """
    psi::Matrix{Float64}
    """ Shared per-bin distribution ``\\pi`` (flattened). Dims: `n_bins × n_keep`. """
    pi::Matrix{Float64}
    """ Per-bin companion-rate density ``\\lambda = \\mathbb{E}[n]\\,\\pi``. Dims: `n_keep × n_log_P × n_log_q`. """
    lambda::Array{Float64, 3}
    """ Tail probabilities ``P(n \\ge c)`` for `c = 1 … max_n_companions`. Dims: `max_n_companions × n_keep`. """
    P_geq::Matrix{Float64}
    """ Expected companion count ``\\mathbb{E}[n]`` per retained draw. Length `n_keep`. """
    E_n::Vector{Float64}
    """ Values ``(n_s, x_s)`` for all systems and retained IMH iterations. Dims: `n_systems × n_keep`. """
    states_trace::Matrix{S}
    """ Number of post-warmup IMH iterations retained. """
    n_keep::Int
    """ Fraction of the trace discarded as warmup. """
    warmup_frac::Float64
end

Base.show(io::IO, p::PopulationPosterior) = print(io,
    "PopulationPosterior(n_bins=$(p.binning.n_bins), ",
    "max_n_companions=$(size(p.psi, 1) - 1), n_keep=$(p.n_keep))")

"""
$(SIGNATURES)

Summarize the raw output of [`run_imh`](@ref) into a [`PopulationPosterior`](@ref).

`result` is the named tuple returned by `run_imh` (with fields `psi_trace` and
`pi_trace`); `binning` is the [`Binning`](@ref) used to produce the binned runs.
The leading `warmup_frac` of each trace is discarded, then the per-draw companion
rate density ``\\lambda = \\mathbb{E}[n]\\,\\pi``, tail probabilities
``P(n \\ge c)`` and expected count ``\\mathbb{E}[n]`` are computed.

This is the standard reduction every downstream demographics plot needs, so it
lives here rather than in user scripts. See [`population_posterior_plot`](@ref).
"""
function population_posterior(result; warmup_frac::Real = 0.2)
    binning = result.binning
    0 ≤ warmup_frac < 1 || throw(ArgumentError("warmup_frac must be in [0, 1), got $warmup_frac"))
    psi_trace = result.psi_trace
    pi_trace  = result.pi_trace

    n_iters = size(psi_trace, 2)
    warmup  = max(1, floor(Int, warmup_frac * n_iters))
    keep    = (warmup + 1):n_iters
    psi     = psi_trace[:, keep]                # (max_n_comp + 1) × n_keep
    pi      = pi_trace[:, keep]                 # n_bins × n_keep
    n_keep  = length(keep)

    states_trace = result.states_trace[:, keep]

    n_per, n_mass = binning.partition_sizes
    max_n_comp = size(psi, 1) - 1

    # P(n ≥ c) tail probabilities, c = 1 … max_n_comp.
    P_geq = zeros(max_n_comp, n_keep)
    for c in 1:max_n_comp, s in 1:n_keep
        P_geq[c, s] = sum(@view psi[c+1:end, s])
    end

    # Expected companion count E[n] per retained draw.
    n_vals = collect(0:max_n_comp)
    E_n    = vec(n_vals' * psi)

    # Joint per-bin rate density λ_x = E[n] · π_x (single shared π).
    lambda = zeros(n_keep, n_per, n_mass)
    for s in 1:n_keep
        pi_mat = reshape(@view(pi[:, s]), n_per, n_mass)
        lambda[s, :, :] .= E_n[s] .* pi_mat
    end

    return PopulationPosterior(binning, psi, pi, lambda, P_geq, E_n, states_trace, n_keep, Float64(warmup_frac))
end



"""
$SIGNATURES

The `lambda` argument is a function that takes in a 
[`BinnedSample`](@ref) and compute a statistic. 
This function will return apply it to all samples, and 
compute a mean for each system. 
"""
joint_reconstructions(lambda::Function, pp::PopulationPosterior) = joint_reconstructions(lambda, pp.states_trace)
function joint_reconstructions(lambda::Function, states_trace::AbstractMatrix)
    broadcast_lambda(x) = lambda.(x)
    return mean(broadcast_lambda, eachcol(states_trace))
end




"""
$SIGNATURES

Suppose we wish to at reconstruction of individual systems 
based on the full dataset, `p(n_s, x_s | y_{1:S})`. 

In that case we can go back to the resolution of individual 
system traces, i.e., unbinned samples stored in 
`IndepRuns.traces`. Notice that the IMH sampler is based on 
proposing these system-level samples. So the number of times 
each is accepted (including possibly zero) is equivalent to  
a weighting scheme. While the preprocessing includes a shuffling 
of the traces (this was found to increase IMH sample quality), 
each `BinnedSample` keeps a record of its index in the original 
`IndepRuns.traces` (namely, `imh_sample.indep_trace_index`). 
We use that to create weights that are in the same order as 
those in `IndepRuns.traces`. 

Returns a matrix of dims `n_samples, n_systems`. The result is 
stored as `Float64` even though the entries are integers 
to facilitate in place normalization and emphasize that the 
fact the values are integers is in a sense an implementation detail. 
"""
joint_reconstruction_weights(pp::PopulationPosterior) = joint_reconstruction_weights(pp.states_trace)
function joint_reconstruction_weights(states_trace::AbstractMatrix) 
    transposed_states_trace = permutedims(states_trace) 

    n_samples, n_systems = size(transposed_states_trace)
    result = zeros(n_samples, n_systems)
    
    for system_index in 1:n_systems
        for iter in 1:n_samples 
            imh_sample = transposed_states_trace[iter, system_index]
            indep_run_sample_index = imh_sample.indep_trace_index 
            result[indep_run_sample_index, system_index] += 1
        end
    end
    return result 
end