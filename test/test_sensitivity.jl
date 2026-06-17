@testset "Synthetic data sensitivity" begin 

    for n_systems in [1, 10, 100, 1000]
        generated = Octopodes.generate_binary_indep_runs(
            psi_some_companion_truth = 0.8, 
            n_systems = n_systems,
            n_systems_iters = 2000,
            mcmc_lazy_pr = 0.9)
        binned = generated.runs

        @show Octopodes.sensitivity(binned, 0.001)
        if n_systems == 1
            @test_opt Octopodes.sensitivity(binned, 0.001)
        end
    end

end

@testset "Binarized data sensitivity" begin     
    binned = bin(b, runs) 
    @show Octopodes.is_binary(binned)

    binarized1 = binarize(binned)
    @show Octopodes.sensitivity(binarized1, 0.001)
    @test_opt Octopodes.sensitivity(binarized1, 0.001)

    binarized2 = binarize(binned, 1)
    @test_opt binarize(binned, 1)
    @show Octopodes.sensitivity(binarized2, 0.001)
    @test_opt Octopodes.sensitivity(binarized2, 0.001)
end

@testset "Sensitivities" begin   
    @show Octopodes.relative_sensitivities(bin(b, runs), 0.001) 
    @test_opt Octopodes.relative_sensitivities(bin(b, runs), 0.001)
end

@testset "Sensitivities for joint reconstructions" begin  
    psi_some_companion_truth = 0.8
    generated = Octopodes.generate_binary_indep_runs(;
            psi_some_companion_truth, 
            n_systems = 100000,
            n_systems_iters = 2000,
            mcmc_lazy_pr = 0.5)

    @show bayes_optimal = Octopodes.synthetic_local_posterior(psi_some_companion_truth, false)

    binned = generated.runs

    out = Octopodes.joint_detection_sensitivity_by_n_systems(binned, 3)

    @test abs(last(out.posteriors) - bayes_optimal[2]) < 0.05
    @test last(out.derivatives) < 0.05
end