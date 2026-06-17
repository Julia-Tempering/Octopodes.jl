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

@testset "Approx agreement of joint detection IMH, reconstruction IMH, and numerical" begin
    n_systems = 10
    generated = Octopodes.generate_binary_indep_runs(;
        psi_some_companion_truth = 1., 
        n_systems,
        n_systems_iters = 100000,
        mcmc_lazy_pr = 0.9)
    binned = generated.runs 

    numerical = Octopodes.numerical_joint_prediction(binned)

    processor_detection = Octopodes.JointDetection(n_systems, 2) 
    run_imh(Xoshiro(41), binned, processor_detection)
    imh = [Octopodes.posterior_detection(processor_detection, 1, s) for s in 1:n_systems]

    @test maximum(abs.(numerical - imh)) < 0.01

    processor_reconstruction = Octopodes.JointReconstuction(binned) 
    run_imh(Xoshiro(41), binned, processor_reconstruction)

    hascompanions(s::Octopodes.BinnedSample) = s.n_companions > 0 ? 1.0 : 0.0
    imh2 = Octopodes.joint_reconstructions(hascompanions, processor_reconstruction) 
    @test maximum(abs.(imh2 - imh)) < 0.0001

    @test_opt Octopodes.JointReconstuction(binned)
    @test_opt run_imh(Xoshiro(41), binned, processor_reconstruction)
    @test_opt Octopodes.joint_reconstructions(hascompanions, processor_reconstruction) 
end

