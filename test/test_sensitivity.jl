@testset "Synthetic data" begin 

    for n_systems in [1, 10, 100, 1000]
        generated = Octopodes.generate_binary_indep_runs(
            psi_some_companion_truth = 0.8, 
            n_systems = n_systems,
            n_systems_iters = 2000,
            mcmc_lazy_pr = 0.9)
        binned = generated.runs

        @show Octopodes.sensitivity(binned, 0.001)
        @test_opt  Octopodes.sensitivity(binned, 0.001)
    end

end