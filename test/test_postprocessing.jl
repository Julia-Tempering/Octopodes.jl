@testset "Joint reconstruction processor" begin

    binned = bin(b, runs)

    
    results = run_imh(Xoshiro(41), binned)

    @test_opt Octopodes.joint_reconstruction_weights(results.states_trace, Octopodes.n_samples(runs))

    weights = Octopodes.joint_reconstruction_weights(results.states_trace, Octopodes.n_samples(runs))[:, 1]
    n = length(weights)
    nz = sum(iszero, weights)
    hard_accept_rate = (n - 1 - nz) / (n - 1)

    soft_accept_rate = results.accept_prs[1]

    # only approx b/c accept_prs uses Rao-Blackwellized version
    @test abs(hard_accept_rate - soft_accept_rate) < 0.01
end

@testset "Multiplicities" begin
    binned = bin(b, runs) 
    imh_output = run_imh(rng, binned)
    posterior = population_posterior(imh_output)

    mult = Octopodes.joint_multiplicities(posterior)
    @test sum(mult[:, 1]) ≈ 1

    @test_opt Octopodes.joint_multiplicities(posterior)
end