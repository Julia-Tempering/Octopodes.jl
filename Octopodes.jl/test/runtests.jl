include("setup.jl")

@testset "Octopodes.jl Tests" begin

    function test_imh()
        @test_opt run_imh(rng, b, runs)
        result = @inferred run_imh(rng, b, runs)
        
    end
    @testset "IMH" test_imh()

    function test_run()
        @test max_n_companions(runs) == 3 
        @test first(runs.traces) isa NamedTuple 
        @test b.partition_sizes == (3, 2)
        @test @inferred companion_indices(runs) == (1, 2, 3)
        @test_opt bin(b, runs)

        binned = @inferred bin(b, runs)
        @test isbitstype(eltype(binned))
    end
    @testset "Independent MCMC runs data format" test_run()

    function test_bin()
        b = @inferred Binning(
            0.0:0.5:1.0, 
            0.0:0.04:0.12,
        )
        @test b.partition_sizes == (2, 3)
        point = (0.1, 0.05) 
        vec = zeros(b.n_bins)
        vec[@inferred bin(b, point)] = 1
        @test @inferred vector_to_array(b, vec)[1, 2] == 1

        @test_throws "Value" bin(b, (0.1, 0.13))
    end
    @testset "Bins" test_bin()

    

end