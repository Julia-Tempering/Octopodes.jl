import Octopodes: Binning, bin, vector_to_array
using Test

@testset "Octopodes.jl Tests" begin

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