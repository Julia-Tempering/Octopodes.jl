module Octopodes 

using   Distributions, 
        DocStringExtensions,
        Random

include("indep_mcmc_runs.jl")
include("bins.jl")
include("joint_sampler.jl")
include("numerical.jl")

export IndepRuns, Binning, bin

end
