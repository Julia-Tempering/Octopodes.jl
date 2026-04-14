module Octopodes 

using   Distributions, 
        DocStringExtensions

include("indep_mcmc_runs.jl")
include("bins.jl")
include("joint_sampler.jl")

end
