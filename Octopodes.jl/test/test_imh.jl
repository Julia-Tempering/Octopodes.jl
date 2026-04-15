@testset "IMH" begin
    binned = bin(b, runs)
    @test_opt run_imh(rng, binned)
    @inferred run_imh(rng, binned)
end