"""
$SIGNATURES 

Convenient access to the main features of this package. 
"""
function octopodes(ir::IndepRuns;
        star_selector = (star_name::String -> true), 
        n_log_P_yr_intervals::Int = 20, 
        n_log_q_intervals::Int = 20,
        imh_rng::AbstractRNG = Xoshiro(41),
        warmup_frac::Real = 0.2, 
        max_n_dotplots::Int = 2
    ) 

    b = Binning(ir; n_log_P_yr_intervals, n_log_q_intervals)
    binned = bin(b, ir; star_selector)
    imh_output = run_imh(imh_rng, binned)

    posterior = population_posterior(imh_output; warmup_frac)
    multiplicities = joint_multiplicities(posterior)
    posterior_plot = population_posterior_plot(posterior)

    system_plots = joint_reconstruction_plot(posterior, binned, ir, 1:max_n_dotplots)

    # TODO: compute updated BFs using all the data (joint detection/multiplicity posterior probabilities)

    return (;
        imh_output,
        posterior,
        joint_multiplicities = multiplicities,
        population_posterior_plot = posterior_plot,
        system_plots
    )
end