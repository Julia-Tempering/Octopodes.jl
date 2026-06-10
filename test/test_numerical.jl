@testset "Numerical methods run and are type stable" begin
    binned = binarize(Octopodes.bin(b, runs))

    Octopodes.compare_numerical_imh(Xoshiro(41), binned)
    Octopodes.numerical_joint_prediction(binned, 1)
    Octopodes.sensitivity(binned) 
    Octopodes.joint_detection_sensitivity(binned, 1)

    @test_opt Octopodes.compare_numerical_imh(Xoshiro(41), binned)
    @test_opt Octopodes.numerical_joint_prediction(binned, 1)
    @test_opt Octopodes.sensitivity(binned) 
    @test_opt Octopodes.joint_detection_sensitivity(binned, 1)
end

@testset "Approx agreement of joint reconstruction IMH vs numerical" begin
    n_systems = 1000
    generated = Octopodes.generate_binary_indep_runs(;
        psi_some_companion_truth = 1., 
        n_systems,
        n_systems_iters = 1000,
        mcmc_lazy_pr = 0.9)
    binned = generated.runs 

    stars = 1:n_systems 

    t = @timed numerical = [Octopodes.numerical_joint_prediction(binned, star) for star in stars]
    @show t
    t = @timed numerical = [Octopodes.numerical_joint_prediction(binned, star) for star in stars]
    @show t 
end