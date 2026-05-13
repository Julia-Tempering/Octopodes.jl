using Octopodes
import Octopodes: 
            Binning, bin, vector_to_array, companion_indices,
            IndepRuns, max_n_companions, run_imh, binarize

using   Test,
        Random,
        CairoMakie,
        JLD2,
        JET
        
const rng = Xoshiro(1)
        
const dict = Octopodes.Examples.small_dict()
const runs = IndepRuns(dict) 
const b = Binning(runs, n_log_P_yr_intervals = 3, n_log_q_intervals = 2)
