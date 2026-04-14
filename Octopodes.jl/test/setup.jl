#=
To run, use 
```
include("test/setup.jl")
```
=#

using Revise

import Octopodes: 
            Binning, bin, vector_to_array, companion_indices,
            IndependentMCMCRuns, max_n_companions, run_imh

using   Test,
        JLD2, 
        JET,
        Random
        

# using JLD2; big_run = IndependentMCMCRuns(JLD2.load("test/IndependentMCMCRuns_big.jld2"));
const dict = JLD2.load(joinpath(@__DIR__, "IndependentMCMCRuns_demo.jld2"))
const runs = IndependentMCMCRuns(dict) 
const b = Binning(runs, n_log_P_yr_intervals = 3, n_log_q_intervals = 2)

const rng = MersenneTwister(1)