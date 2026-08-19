@testset "IMH" begin
    binned = bin(b, runs)
    @test_opt run_imh(rng, binned)
    @inferred run_imh(rng, binned)
end

@testset "Reshuffling" begin
    binned = bin(b, runs)
    result = run_imh(rng, binned, n_imh_iters = 3 * Octopodes.default_n_imh_iters(binned), store_states_trace = true)
    @test size(result.states_trace) == (2, 3 * Octopodes.default_n_imh_iters(binned))
end

@testset "Acceptance counts match the dense states trace" begin
    binned = bin(b, runs)
    result = run_imh(Xoshiro(7), binned; store_states_trace = true, warmup_frac = 0.2)
    n_iters = size(result.psi_trace, 2)
    warmup = max(1, floor(Int, result.warmup_frac * n_iters))
    kept = @view result.states_trace[:, (warmup + 1):n_iters]
    from_trace = Octopodes.joint_reconstruction_weights(kept, Octopodes.n_samples(runs))
    @test from_trace == Float64.(result.accept_counts)

    post = population_posterior(result)
    mult_from_counts = Octopodes.joint_multiplicities(post)
    max_n = Octopodes.max_n_companions(post)
    mult_from_trace = Octopodes.joint_reconstructions(post) do bs
        Octopodes.one_hot(Int(bs.n_companions) + 1, max_n + 1)
    end
    @test mult_from_counts == mult_from_trace
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
    binned = binarize(Octopodes.bin(b, runs))

    compare_numerical_imh_results = 
        Octopodes.compare_numerical_imh(Xoshiro(41), binned)

    @test @show(compare_numerical_imh_results.ks_p_value) > 0.01

    p = Octopodes.compare_numerical_imh_plot(compare_numerical_imh_results)
    Makie.save("$(results_prefix)_real.png", p; size = (300, 300))
    Octopodes.save_latex_key_values("$(results_prefix)_real.tex", 
        ksPValueReal =  round(compare_numerical_imh_results.ks_p_value, digits=2))
end

