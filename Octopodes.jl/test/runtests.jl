import Octopodes: 
            Binning, bin, vector_to_array, 
            IndependentMCMCRuns, max_n_companions, traces
using   Test,
        JLD2

@testset "Octopodes.jl Tests" begin

    @testset "Independent MCMC runs data format" begin
        dict = JLD2.load("IndependentMCMCRuns_demo.jld2")
        runs = IndependentMCMCRuns(dict) 
        @test max_n_companions(runs) == 3 
        @test first(traces(runs)) isa NamedTuple 

        b = Binning(runs, n_log_P_yr_intervals = 3, n_log_q_intervals = 2)
        @test b.partition_sizes == (3, 2)
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