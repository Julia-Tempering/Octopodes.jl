@testset "Test joint reconstruction processor" begin

    binned = bin(b, runs)

    processor_reconstruction = Octopodes.JointReconstuction(binned) 
    results = run_imh(Xoshiro(41), binned, processor_reconstruction)

    wc = Octopodes.JointReconstructionWeightCalculator(processor_reconstruction)

    @test_opt Octopodes.joint_reconstruction_weights(wc, 1)

    weights = Octopodes.joint_reconstruction_weights(wc, 1)
    n = length(weights)
    nz = sum(iszero, weights)
    hard_accept_rate = (n - 1 - nz) / (n - 1)

    soft_accept_rate = results.accept_prs[1]

    # only approx b/c accept_prs uses Rao-Blackwellized version
    @test abs(hard_accept_rate - soft_accept_rate) < 0.01
end