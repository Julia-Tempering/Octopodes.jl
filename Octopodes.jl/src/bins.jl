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
Binning(runs::IndependentMCMCRuns; n_log_P_yr_intervals::Int, n_log_q_intervals::Int) =
    Binning(
        build_grid(log_P_yr_prior(runs), n_log_P_yr_intervals),
        build_grid(log_q_prior(runs),    n_log_q_intervals)
    )
build_grid(prior::Uniform, n_intervals) = range(prior.a, prior.b, n_intervals + 1)

"""
$(SIGNATURES)
"""
function Binning(log_P_yr_grid::StepRangeLen, log_q_grid::StepRangeLen)
    partition_sizes = n_intervals.((log_P_yr_grid, log_q_grid))
    n_bins = prod(partition_sizes)
    return Binning(log_P_yr_grid, log_q_grid, partition_sizes, n_bins)
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