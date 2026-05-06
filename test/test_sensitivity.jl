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