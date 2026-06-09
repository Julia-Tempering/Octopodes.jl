@testset "IMH" begin
    binned = bin(b, runs)
    @test_opt run_imh(rng, binned)
    @inferred run_imh(rng, binned)
end

results_prefix = "$(Octopodes.plots_folder())/numerical_imh_check"

@testset "IMH and numerical agree on synthetic data" begin 
    generated = Octopodes.generate_binary_indep_runs(
        psi_some_companion_truth = 0.7, 
        n_systems = 20,
        n_systems_iters = 10000,
        mcmc_lazy_pr = 0.9)
    binned = generated.runs

    compare_numerical_imh_results = 
        Octopodes.compare_numerical_imh(Xoshiro(41), binned)

    @test @show(compare_numerical_imh_results.ks_p_value) > 0.01

    p = Octopodes.compare_numerical_imh_plot(compare_numerical_imh_results)
    Makie.save("$(results_prefix)_synthetic.png", p; size = (300, 300))
    Octopodes.save_latex_key_values("$(results_prefix)_synthetic.tex", 
        ksPValueSynthetic = round(compare_numerical_imh_results.ks_p_value, digits=3))
end

@testset "IMH and numerical agree on real data" begin 
    b = Binning(runs, n_log_P_yr_intervals = 1, n_log_q_intervals = 1)
    binned = binarize(Octopodes.bin(b, runs))

    compare_numerical_imh_results = 
        Octopodes.compare_numerical_imh(Xoshiro(41), binned)

    @test @show(compare_numerical_imh_results.ks_p_value) > 0.01

    p = Octopodes.compare_numerical_imh_plot(compare_numerical_imh_results)
    Makie.save("$(results_prefix)_real.png", p; size = (300, 300))
    Octopodes.save_latex_key_values("$(results_prefix)_real.tex", 
        ksPValueReal =  round(compare_numerical_imh_results.ks_p_value, digits=2))
end
