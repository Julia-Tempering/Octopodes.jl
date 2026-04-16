using Octopodes
import Octopodes: 
            Binning, bin, vector_to_array, companion_indices,
            IndepRuns, max_n_companions, run_imh, binarize

using   Test,
        JLD2, 
        Random,
        CairoMakie,
        JET
        
const rng = MersenneTwister(1)
        
const dict = JLD2.load(joinpath(@__DIR__, "IndepRuns_demo.jld2"))
const runs = IndepRuns(dict) 
const b = Binning(runs, n_log_P_yr_intervals = 3, n_log_q_intervals = 2)

const big_run = IndepRuns(JLD2.load(joinpath(@__DIR__, "IndepRuns_big.jld2")))
const big_b =  Binning(big_run, n_log_P_yr_intervals = 20, n_log_q_intervals = 20)