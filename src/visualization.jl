""" 
$SIGNATURES 
"""
function numerical_posterior_plot(posterior::Vector, true_proportion = nothing)
    grid = build_grid(posterior)
    fig, ax, _ = lines(grid, posterior)
    ax.xlabel = L"\psi" 
    ax.ylabel = "posterior density"
    if !isnothing(true_proportion)
        vlines!(ax, [true_proportion], color=:green, linestyle=:dash)
    end
    return fig
end

function joint_detection_prior_sensitivity_synth_plot()

    psi_some_companion_truth = 0.8 
    generated = Octopodes.generate_binary_indep_runs(;
            psi_some_companion_truth, 
            n_systems = 100000,
            n_systems_iters = 10000,
            mcmc_lazy_pr = 0.5)

    # Only the curves corresponding to a no-companion are non-trivial
    s = findfirst(iszero, generated.x_truth)

    fig = Figure(size = (600, 450))
    ax1 = Axis(fig[1, 1], xscale = log10,                 ylabel = L"P(n_1 > 0|y_{1:S})")
    ax2 = Axis(fig[2, 1], xscale = log10, yscale = log10, ylabel = L"|\partial_\mu P_\mu(n_1 > 0|y_{1:S})|", xlabel = L"S")
    hidexdecorations!(ax1, grid = false, ticks = false)

    bayes_optimal = Octopodes.synthetic_local_posterior(psi_some_companion_truth, false)
    hlines!(ax1, [bayes_optimal[2]], linestyle = :dash, color = :gray, linewidth = 1.5)

    series = joint_detection_sensitivity_by_n_systems(generated.runs, s) 
    len = length(series.posteriors) 
    xs = 2 .^ (1:len)
    lines!(ax1, xs, series.posteriors)
    lines!(ax2, xs, abs.(series.derivatives))
    return fig
end


## Reproducing the paper's figures

function all_figures(jld2_file::String)
    real_data = IndepRuns(JLD2.load(jld2_file))
    # TODO: save jld2_file + commit + date
    validation_and_interpretation_figure(real_data) 
    full_run(real_data)
end

plots_folder() = mkpath("results/Planet-Demographics/plots")

function prior_sensitivity_figure()
    output_folder = plots_folder()

    synth_plot = joint_detection_prior_sensitivity_synth_plot() 
    save("$output_folder/prior_sensitivity_figure_synth.png", synth_plot)
end

function validation_and_interpretation_figure(real_data) 
    output_folder = plots_folder()

    # real data for comparison and setting number of systems, stars
    real_binarized = binarize(bin(Binning(real_data, n_log_P_yr_intervals = 1, n_log_q_intervals = 1), real_data))
    real_local_post = standardized_local_posteriors(real_binarized)
    local_fig, local_ax, _ = lines(real_local_post, label = "Real data")
    n_systems, n_systems_iters = size(real_binarized.samples)
    save_latex_key_values("$output_folder/validation_and_interpretation_figure_variables.tex";
        totalnsystems = n_systems,
        niters = n_systems_iters
    )
    real_posterior_plot = numerical_posterior_plot(numerical(real_binarized))
    save("$output_folder/validation_and_interpretation_figure__real.png", real_posterior_plot; size = (200, 200))
    
    # synthetic data on the same plot
    psis = [0.5, 0.9, 1.0] 
    for psi in psis 
        generated = generate_binary_indep_runs(; psi_some_companion_truth = psi, n_systems, n_systems_iters)
        posterior_plot = posterior_recovery_plot(generated)
        save("$output_folder/validation_and_interpretation_figure__psi=$psi.png", posterior_plot; size = (200, 200))
    
        synth_local_post = standardized_local_posteriors(generated.runs)
        lines!(local_ax, synth_local_post, label = L"Synthetic (\psi^* = $psi)", linestyle = :dash)
    end

    save("$output_folder/validation_and_interpretation_figure__locals.png", local_fig)
    return local_fig
end

save_latex_key_values(path::String; kwargs...) = 
    open(path, "w") do io
        for (k, v) in kwargs
            println(io, "\\newcommand{\\$k}{$v}")
        end
    end

function posterior_recovery_plot(generated, eps = default_eps)
    binned = generated.runs 
    true_proportion = generated.psi_some_companion_truth 
    posterior = numerical(binned, eps)
    return numerical_posterior_plot(posterior, true_proportion)
end

function full_run(real_data)
    output_folder = plots_foler()

    # real data for comparison and setting number of systems, stars
    real_binned = bin(Binning(real_data, n_log_P_yr_intervals = 1, n_log_q_intervals = 1), real_data)
    
    rng = Xoshiro(1)
    results = run_imh(rng, real_binned)
    @show mean(results.psi_trace, dims = 2)
    return results
end