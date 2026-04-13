"""
A real valued parameter such as log10(mass), log10(period) which is partitioned 
over a regular grid. 
"""
struct PartitionedParameter{S <: StepRangeLen}
    """ Readable description for plotting, output, etc """
    name::Symbol 

    """
    View each interval in the `StepRangeLen` as closed on the left and open on the right, 
    except for the last one which we view as closed in each. 
    """
    partition::S
end 

struct Binning{P <: Tuple, S <: Tuple}
    parameters::P # iterable over PartitionedParameter

    # derived quantities for quick access
    partition_sizes::S 
    n_bins::Int 
end
function Binning(parameters::PartitionedParameter...)
    partition_sizes = n_intervals.(parameters)
    n_bins = prod(partition_sizes)
    return Binning(parameters, partition_sizes, n_bins)
end

default_binning() = Binning(
    PartitionedParameter(:log_P_yr, range(log10(1/365.25), log10(10000.0), length=25)), 
    PartitionedParameter(:log_q,    range(log10(1e-5),     log10(1.0),     length=21)),
)

function bin(b::Binning, values)
    interval_indices = interval_index.(b.parameters, values)
    return LinearIndices(b.partition_sizes)[interval_indices...]
end

vector_to_array(b::Binning, vector::Vector) = reshape(vector, b.partition_sizes)

n_intervals(param::PartitionedParameter) = length(param.partition) - 1 

"""
Return the index of the intervals in the provided `PartitionedParameter` the given point falls in, or 
throw `BoundsError` if the point falls in none of the intervals. 
"""
function interval_index(param::PartitionedParameter, value::Real)
    r = param.partition
    if value == last(r) # last inteval we close on the right
        return length(r) - 1
    end
    result = Int(div(value - first(r), step(r), RoundDown)) + 1
    if !(1 ≤ result < length(r))
        error("Value $value out of range for $param")
    end
    return result
end



### data load (TODO: move to another file)


struct SystemSample{I <: Tuple}
    companion_count::Int
    bin_memberships::I # warning: this contains inactive ones (for type stability)
end
max_n_companions(sample) = length(sample.bin_memberships)

load_value(system_data, iter::Int, c::Int, param::PartitionedParameter)::Float64 = system_data[param.name][c, iter]
load_values(b::Binning, system_data, iter::Int, c::Int) = [load_value(system_data, iter, c, p) for p in b.parameters]

loadbin_membership(b::Binning, system_data, iter::Int, c::Int) = bin(b, load_values(b, system_data, iter, c))
loadbin_memberships(b::Binning, system_data, iter::Int, companion_indices::Tuple) = [loadbin_membership(b, system_data, iter, c) for c in companion_indices]

max_n_companions_and_n_iters(b, system_data)::Tuple{Int, Int} = size(system_data[first(b.parameters).name])

function load_system_sample(b::Binning, system_data, iter::Int, companion_indices::Tuple)
    companion_count = system_data.n_planets[iter]
    bin_memberships = loadbin_memberships(b, system_data, iter, companion_indices)
    return SystemSample(companion_count, bin_memberships)
end

function load_system_trace(b::Binning, system_data, companion_indices::Tuple)
    max_n_companions, n_iters = max_n_companions_and_n_iters(b, system_data)
    if max_n_companions != length(companion_indices)
        error("Max number of companions differ: $max_n_companions vs $(length(companion_indices))")
    end
    return [load_system_sample(b, system_data, i, companion_indices) for i in 1:n_iters]
end

companion_indices() = (1, 2, 3)
load_system_traces(b::Binning, data) = [load_system_trace(b, system_data, companion_indices()) for system_data in data] 

test_system() = load_system_traces(default_binning(), JLD2.load("multicomp/multicomp_v7_cache.jld2")["star_data"][1:100])

# TODO: add tests to make sure this code stays type stable (it currently is, as of first draft)


# Next: IMH

active_companions(s::SystemSample) = 1:s.companion_count
product_eta(s::SystemSample, eta_to_eta_tilde_ratios) = prod(i -> eta_to_eta_tilde_ratios[s.bin_memberships[i]], active_companions(s))
accept_pr(current::SystemSample, proposed::SystemSample, psi_to_tilde_psi_ratios, eta_to_eta_tilde_ratios) = min(1,
        psi_to_tilde_psi_ratios[proposed.companion_count + 1] / 
        psi_to_tilde_psi_ratios[current.companion_count  + 1] * 
        product_eta(proposed, eta_to_eta_tilde_ratios) / 
        product_eta(current, eta_to_eta_tilde_ratios)
    )