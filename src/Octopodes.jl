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

# Gallery (per-system mass–separation dotplot pages). These pull in the heavier
# domain dependencies (DataFrames for the per-star posterior table, PlanetOrbits
# for projecting catalog orbits / posterior separations). Kept as a `using` here
# so `src/gallery.jl` can call them unqualified.
using DataFrames
using PlanetOrbits

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
include("gallery.jl")


export octopodes, IndepRuns, Binning, bin,
       run_imh, relative_sensitivities,
       population_posterior, population_posterior_plot,
       posterior_dotplot, posterior_dotplot!, population_heatmap_axis,
       joint_reconstruction_weights,
       gallery_panel!, gallery_page, save_gallery, reweighted_samples,
       load_wds_catalog

end
