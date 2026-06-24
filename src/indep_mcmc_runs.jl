"""
Information needed about the independent MCMC runs for each system in 
order to run joint inference. 

$(FIELDS)
"""
struct IndepRuns{V <: Vector, P <: Uniform, N, max_n_companions}
    """
    A vector of traces, one for each system. 

    Each trace is a `NamedTuple` assumed to have the following fields:

    - `n_planets::Vector{Int64}`,
    - `log_P_yr::Matrix{Float64}`, 
    - `log_q::Matrix{Float64}`, 
    - `n_samples::Int64`, 
    - `name::String`

    """
    traces::V
    mv::Val{max_n_companions}
    log_P_yr_prior::P
    log_q_prior::P
    n_companions_prior::N
    name_to_index::Dict{String,Int}
end
Base.show(io::IO, runs::IndepRuns) = print(io, "IndepRuns(max_n_companions=$(max_n_companions(runs)), n_systems=$(length(runs.traces)))")

"""
$(SIGNATURES)

Load the  data from an informal exchange format defined in the documentation.
"""
function IndepRuns(d::D) where {D <: Dict}
    NT = typeof(first(d["star_data"]))
    traces = Vector{NT}(d["star_data"])
    max = max_n_companions(traces)
    @assert max < 2^8 # we use UInt 8 for BinnedSample.n_companions
    n_companions_prior = d["n_planets_prior"]
    @assert Distributions.support(n_companions_prior) == 0:max 
    name_to_index = Dict(traces[i].name => i for i in eachindex(traces))
    result = IndepRuns(
        traces, 
        Val(max), 
        d["log_P_yr_prior"],
        d["log_q_prior"],
        n_companions_prior,
        name_to_index
    )
    @assert allunique(stars(result))
    return result
end

function stars(runs::IndepRuns)
    name(t) = t.name
    return map(name, runs.traces)
end

trace_by_name(ir::IndepRuns, star_name::String) = ir.traces[ir.name_to_index[star_name]]

max_n_companions(r) = max_n_companions_and_samples(r)[1]
n_samples(r) = max_n_companions_and_samples(r)[2]
max_n_companions_and_samples(r::IndepRuns) = max_n_companions_and_samples(r.traces)
function max_n_companions_and_samples(traces::Vector)
    system_trace = first(traces) 
    return size(system_trace.log_P_yr)
end

max_n_companions(runs::IndepRuns) = get_from_val(runs.mv)
get_from_val(::Val{x}) where {x} = x