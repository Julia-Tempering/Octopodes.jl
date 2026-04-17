@testset "IMH" begin
    binned = bin(b, runs)
    @test_opt run_imh(rng, binned)
    @inferred run_imh(rng, binned)
end



@testset "Agreement with numerical" begin
    binned = binarize(bin(b, runs, thinning = 3))
    compare_numerical_imh_results = 
        Octopodes.compare_numerical_imh(Xoshiro(1), binned)
    @test @show(compare_numerical_imh_results.ks_p_value) > 0.01

    # TODO: add compare_numerical_imh_plot(compare_numerical_imh_results) to documentation 
end

@testset "Synthetic data" begin 
    generated = Octopodes.generate_binary_indep_runs(
        psi_some_companion_truth = 0.7, 
        n_systems = 20,
        n_systems_iters = 2000,
        mcmc_lazy_pr = 0.9,
        thinning = 1)
    binned = generated.runs

    compare_numerical_imh_results = 
        Octopodes.compare_numerical_imh(Xoshiro(41), binned)

    @test @show(compare_numerical_imh_results.ks_p_value) > 0.01

#Octopodes.compare_numerical_imh_plot(compare_numerical_imh_results)
end

# big_thinned = Octopodes.binarize(bin(big_b, big_run; thinning = 10, shuffle_rng = nothing)) # Xoshiro(1)))

# compare_numerical_imh_results = 
#         Octopodes.compare_numerical_imh(Xoshiro(1), big_thinned)

# @show compare_numerical_imh_results.ks_p_value

# Octopodes.compare_numerical_imh_plot(compare_numerical_imh_results)