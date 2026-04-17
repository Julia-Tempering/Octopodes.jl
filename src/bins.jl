"""
Discretization of the `(log_P_yr, log_q)` into regular grids. 

Each grid is specified using `StepRangeLen`, see 
[`interval_index`](@ref).
"""
struct Binning{G <: StepRangeLen, S <: Tuple}
    log_P_yr_grid::G
    log_q_grid::G

    # derived quantities for quick access
    partition_sizes::S 
    n_bins::Int 
end

"""
$(SIGNATURES)

Use the prior bounds from the independent MCMC runs and a requested number of intervals 
in each axis to build a binning.
"""
Binning(runs::IndepRuns; n_log_P_yr_intervals::Int, n_log_q_intervals::Int) =
    Binning(
        build_grid(runs.log_P_yr_prior, n_log_P_yr_intervals),
        build_grid(runs.log_q_prior,    n_log_q_intervals)
    )
build_grid(prior::Uniform, n_intervals) = range(prior.a, prior.b, n_intervals + 1)

function Binning(log_P_yr_grid::StepRangeLen, log_q_grid::StepRangeLen)
    partition_sizes = n_intervals.((log_P_yr_grid, log_q_grid))
    n_bins = prod(partition_sizes)
    return Binning(log_P_yr_grid, log_q_grid, partition_sizes, n_bins)
end

"""
Binned version of independent MCMC runs. 

$(FIELDS)
"""
struct BinnedIndepRuns{B <: Binning, M <: Matrix, V}
    binning::B

    """ Dims: (system index, MCMC iteration). """
    samples::M
    tilde_psi::V
    max_n_companions::Int
    star_names::Vector{String}
end

"""
$(SIGNATURES)

Perform binning on all the samples. Returns an Array of [`BinnedSample`](@ref).
Rows are systems, columns are MCMC iterations.

Thinning of `k`` refers to using only one every `k` MCMC samples. 

You can also shuffle each system independently. Provide a `shuffle_rng`
with the `AbstractRNG` object, set to `nothing` to skip shuffling. 

If both shuffling and thinning are requested, thinning is one first, then shuffling after. 

Assume at the moment that all traces have the same number of iterations. 
"""
function bin(
        b::Binning, runs::IndepRuns; 
        star_selector = (star_name::String -> true), 
        thinning::Int = 1, 
        shuffle_rng::Union{Nothing, AbstractRNG} = Xoshiro(1)
        ) 
    # for now, we skip computing tilde_psi since it is uniform so cancel each other in accept ratio
    @assert runs.log_P_yr_prior isa Uniform 
    @assert runs.log_q_prior isa Uniform 
    # we do not make that assumption on the prior on the number of companions 
    max_comp = max_n_companions(runs)
    tilde_psi = map(n -> pdf(runs.n_companions_prior, n), 0:max_comp)

    samples, star_names = _bin(b, runs, companion_indices(runs), star_selector, thinning, shuffle_rng)
    return BinnedIndepRuns(b, samples, tilde_psi, max_comp, star_names)
end

function _bin(b::Binning, runs::IndepRuns, comp_indices::T, star_selector, thinning::Int, shuffle_rng) where {T <: Tuple}
    @assert thinning ≥ 1
    n_systems = length(runs.traces)
    original_n_samples = n_samples(runs)
    thinned_indices = 1:thinning:original_n_samples
    n_samples_after_thinning = length(thinned_indices)
    samples = Array{BinnedSample{T}}(undef, n_samples_after_thinning, n_systems)
    star_names = String[]
    for s in 1:n_systems 
        star_name = runs.traces[s].name
        if star_selector(star_name)
            output = @view samples[:, s]
            _bin!(output, b, comp_indices, runs.traces[s], thinned_indices, shuffle_rng)
            push!(star_names, star_name)
        end
    end
    return permutedims(samples), star_names
end

shuffle_if_needed(::Nothing, indices) = indices 
shuffle_if_needed(rng::AbstractRNG, indices) = shuffle(rng, indices) 

function _bin!(output, b::Binning, comp_indices::T, system_trace::NamedTuple, thinned_indices, shuffle_rng) where {T <: Tuple}
    thinned_indices = shuffle_if_needed(shuffle_rng, thinned_indices)
    log_P_yr::Matrix{Float64} = @view system_trace.log_P_yr[:, thinned_indices]
    log_q::Matrix{Float64} = @view system_trace.log_q[:, thinned_indices]
    n_planets::Vector{Int64} = @view system_trace.n_planets[thinned_indices] 
    n_samples::Int = length(output)

    @assert size(log_P_yr) == size(log_q)
    @assert size(log_P_yr)[2] == size(log_q)[2] == length(n_planets) == n_samples
    @assert length(thinned_indices) == n_samples
    
    max_n_companions = length(comp_indices)
    @assert size(log_P_yr)[1] == size(log_q)[1] == max_n_companions

    buffer = zeros(Int, max_n_companions)
    for iter in 1:n_samples
        n_companions = n_planets[iter]
        @assert 0 ≤ n_companions ≤ max_n_companions
        for c in 1:max_n_companions
            values = (log_P_yr[c, iter], log_q[c, iter]) 
            buffer[c] = c ≤ n_companions ? bin(b, values) : 0
        end
        bin_memberships = map(i -> buffer[i], comp_indices) 
        output[iter] = BinnedSample(n_companions, bin_memberships)
    end
    return nothing 
end

companion_indices(runs::IndepRuns) = companion_indices(runs.mv)
companion_indices(::Val{N}) where {N} = tuple(1:N...)

"""
$(FIELDS)
"""
struct BinnedSample{T <: Tuple}
    n_companions::Int

    """ Contains inactive ones (for type stability) - the inactive ones are set to zero. """
    bin_memberships::T 
end

"""
$(SIGNATURES) 

Given a [`Binning`](@ref) and an iterable over reals, provide the index of the 
corresponding bin. 
"""
function bin(b::Binning, values) 
    @assert eltype(values) <: Real 
    @assert length(values) == 2
    interval_indices = interval_index.((b.log_P_yr_grid, b.log_q_grid), values)
    return LinearIndices(b.partition_sizes)[interval_indices...]
end

"""
$(SIGNATURES) 

Given a [`Binning`](@ref) and a vector of reals, reshape it into a matrix 
where the rows correspond to blocks in the `log_P_yr` partition, and the columns, 
to blocks in the `log_q` partition.

To perform the inverse operation, simply use `vec(matrix)`. 
"""
vector_to_array(b::Binning, vector::Vector) = reshape(vector, b.partition_sizes)

"""
The number of intervals (blocks) is one less than the number of grid (boundaries).
"""
n_intervals(grid::StepRangeLen) = length(grid) - 1 

"""
Return the index of the intervals in the provided `StepRangeLen` 
the given point falls in, or 
throw `BoundsError` if the point falls in none of the intervals. 

All the intervals are viewed as closed on the left and open to the right, 
except the last one which is closed on both sides. 
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