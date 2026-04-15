include("setup.jl")

@testset "Octopodes.jl Tests" begin

    function test_imh()
        binned = bin(b, runs)

        @test_opt run_imh(rng, binned)
        @inferred run_imh(rng, binned)
    end
    @testset "IMH" test_imh()

    function test_run()
        @test max_n_companions(runs) == 3 
        @test first(runs.traces) isa NamedTuple 
        @test b.partition_sizes == (3, 2)
        @test @inferred companion_indices(runs) == (1, 2, 3)
        @test_opt bin(b, runs)

        binned = @inferred bin(b, runs)
        @test isbitstype(eltype(binned.samples))
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

     
    function test_selection()
        binned = bin(b, runs; star_selector = ==("HIP100017"))
        @test binned.star_names == ["HIP100017"]
    end
    @testset "Selection" test_selection()

end