@testset "API" begin
    octopodes(runs)

    octopodes(runs, n_imh_iters = 10)
    octopodes(runs, n_imh_iters = 100000)
end

@testset "Selector" begin
    star_selector(name) = name == "HIP100017" 
    result = octopodes(runs; star_selector)
    n_sys, _ = size(result.imh_output.states_trace)
    @test n_sys == 1
end