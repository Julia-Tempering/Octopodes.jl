using JET # JET currently breaks with Revise, so only use it here, not in setup.jl !

include("setup.jl")

test_dir = @__DIR__ 
project_root_dir = dirname(test_dir)

@testset "Octopodes" begin
    # load all files starting with "test_"
    for test_name in filter(x -> startswith(x, "test_") && endswith(x, ".jl"), readdir(test_dir)) 
        # organize output a little bit
        println() # v otherwise can't tell what is running when it crashes in the middle
        println("### Starting $test_name")
                # + - yes, we need this horror, because we are dealing with a macro
                # v 
        @testset "$test_name" include(test_name)
    end
end