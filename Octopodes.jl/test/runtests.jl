import Octopodes: 
            Binning, bin, vector_to_array, companion_indices,
            IndependentMCMCRuns, max_n_companions, traces, 
            n_companions_prior
using   Test,
        JLD2,
        JET

@testset "Octopodes.jl Tests" begin

    @testset "Independent MCMC runs data format" begin
        dict = JLD2.load("IndependentMCMCRuns_demo.jld2")
        runs = IndependentMCMCRuns(dict) 
        @test max_n_companions(runs) == 3 
        @test first(traces(runs)) isa NamedTuple 
        @inferred traces(runs) 

        b = Binning(runs, n_log_P_yr_intervals = 3, n_log_q_intervals = 2)
        @test b.partition_sizes == (3, 2)

        @test @inferred companion_indices(runs) == (1, 2, 3)

        binned = @inferred bin(b, runs)

        t = traces(runs)[1]
        comp_indices = companion_indices(runs)
        @test_opt bin(b, comp_indices, t)

        @test isbitstype(eltype(binned))
    end

    @testset "Bins" begin
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

    

end