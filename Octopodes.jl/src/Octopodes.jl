module Octopodes 

using   Distributions, 
        DocStringExtensions,
        CairoMakie,
        LaTeXStrings,
        Random

import LogExpFunctions: logsumexp

include("utils.jl")
include("indep_mcmc_runs.jl")
include("bins.jl")
include("joint_sampler.jl")
include("numerical.jl")

export IndepRuns, Binning, bin

end
