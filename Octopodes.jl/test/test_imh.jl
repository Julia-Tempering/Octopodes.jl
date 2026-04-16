@testset "IMH" begin
    binned = bin(b, runs)
    @test_opt run_imh(rng, binned)
    @inferred run_imh(rng, binned)
end



@testset "Agreement with numerical" begin
    binned = binarize(bin(b, runs))
    compare_numerical_imh_results = 
        Octopodes.compare_numerical_imh(MersenneTwister(1), binned)
    @test compare_numerical_imh_results.ks_p_value > 0.01

    # TODO: add compare_numerical_imh_plot(compare_numerical_imh_results) to documentation 
end
