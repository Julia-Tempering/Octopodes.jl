module Octopodes 

using   Distributions, 
        DocStringExtensions,
        CairoMakie,
        LaTeXStrings,
        Random

import LogExpFunctions: logaddexp
import ForwardDiff

include("utils.jl")
include("indep_mcmc_runs.jl")
include("bins.jl")
include("joint_sampler.jl")
include("numerical.jl")
include("sensitivity.jl")
include("synthetic.jl")
include("examples.jl")

export IndepRuns, Binning, bin

end
