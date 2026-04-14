"""
Wrapper around an informal, Dict-based interface to share 
independent MCMC runs over 
individual systems. 

We assume we can obtain traces for `log_P_yr` and 
`log_q` parameters (log period in years and log mass-ratio respectively, 
both in log base 10) for each system.

We also assume we can obtain the prior on `log_P_yr` and 
`log_q` parameters, shared across all runs.
"""
struct IndependentMCMCRuns{D <: Dict, max_n_companions}
    data::D
    function IndependentMCMCRuns(d::D) where {D <: Dict} 
        max_n_companions = max_n_companions_and_validate(d)
        new{D, max_n_companions}(d)
    end
end
Base.show(io::IO, runs::IndependentMCMCRuns{D, N}) where {D, N} = print(io, "IndependentMCMCRuns(max_n_companions=$N, n_systems=$(length(traces(runs))))")

function max_n_companions_and_validate(d::Dict)::Int
    system_trace = first(traces(d)) 
    max_n_companions, n_iters = size(system_trace.log_P_yr)
    return max_n_companions
end

"""
$(SIGNATURES)

A vector of traces, one for each system. 

Each trace is a `NamedTuple` assumed to have the following fields:

- `n_planets::Vector{Int64}`,
- `log_P_yr::Matrix{Float64}`, 
- `log_q::Matrix{Float64}`, 
- `n_samples::Int64`, 
- `name::String`

"""
traces(runs::IndependentMCMCRuns)::Vector{NamedTuple} = traces(runs.data)
traces(d::Dict) = d["star_data"]

max_n_companions(::IndependentMCMCRuns{D, max}) where {D, max} = max

log_P_yr_prior(runs::IndependentMCMCRuns)::Uniform = runs.data["log_P_yr_prior"]
log_q_prior(runs::IndependentMCMCRuns)::Uniform = runs.data["log_q_prior"]

