module Octopodes 

using   Distributions,
        DocStringExtensions,
        CairoMakie,
        LaTeXStrings,
        Printf,
        Random

import LogExpFunctions: logaddexp
import ForwardDiff
import JLD2

include("utils.jl")
include("indep_mcmc_runs.jl")
include("bins.jl")
include("joint_sampler.jl")
include("numerical.jl")
include("sensitivity.jl")
include("synthetic.jl")
include("examples.jl")
include("postprocessing.jl")
include("api.jl")
include("processors.jl")
include("visualization.jl")
include("dotplot.jl")


export IndepRuns, Binning, bin,
       run_imh, relative_sensitivities,
       population_posterior, population_posterior_plot,
       posterior_dotplot, posterior_dotplot!, population_heatmap_axis

end
