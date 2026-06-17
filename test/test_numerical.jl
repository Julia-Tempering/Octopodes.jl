@testset "Numerical methods run and are type stable" begin
    binned = binarize(Octopodes.bin(b, runs))

    Octopodes.compare_numerical_imh(Xoshiro(41), binned)
    Octopodes.numerical_joint_prediction(binned)
    Octopodes.sensitivity(binned) 
    Octopodes.joint_detection_sensitivities(binned)

    @test_opt Octopodes.compare_numerical_imh(Xoshiro(41), binned)
    @test_opt Octopodes.numerical_joint_prediction(binned)
    @test_opt Octopodes.sensitivity(binned) 
    @test_opt Octopodes.joint_detection_sensitivities(binned)
end

@testset "Approx agreement of joint reconstruction IMH vs numerical" begin
    n_systems = 10
    generated = Octopodes.generate_binary_indep_runs(;
        psi_some_companion_truth = 1., 
        n_systems,
        n_systems_iters = 100000,
        mcmc_lazy_pr = 0.9)
    binned = generated.runs 

    numerical = Octopodes.numerical_joint_prediction(binned)

    processor = Octopodes.JointDetection(n_systems, 2) 
    run_imh(rng, binned, processor)
    imh = [Octopodes.posterior_detection(processor, 1, s) for s in 1:n_systems]

    @test maximum(abs.(numerical - imh)) < 0.01
end

