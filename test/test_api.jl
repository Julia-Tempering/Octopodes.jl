@testset "API" begin
    binned = bin(b, runs)
    octopodes(runs)

    octopodes(runs, n_imh_iters = 10)
    octopodes(runs, n_imh_iters = 100000)
end