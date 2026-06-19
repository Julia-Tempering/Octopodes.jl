@testset "Population posterior summary" begin
    binned = bin(b, runs)
    result = run_imh(rng, binned)
    n_iters = size(result.psi_trace, 2)

    post = population_posterior(result, b; warmup_frac = 0.2)

    @test post.n_keep == n_iters - max(1, floor(Int, 0.2 * n_iters))
    @test size(post.psi)    == (max_n_companions(runs) + 1, post.n_keep)
    @test size(post.pi)     == (b.n_bins, post.n_keep)
    @test size(post.lambda) == (post.n_keep, b.partition_sizes...)
    @test size(post.P_geq)  == (max_n_companions(runs), post.n_keep)
    @test length(post.E_n)  == post.n_keep

    @test all(0 .≤ post.P_geq .≤ 1)
    # λ = E[n]·π and π sums to one over bins ⇒ summing λ over bins recovers E[n].
    @test vec(sum(post.lambda, dims = (2, 3))) ≈ post.E_n

    @test_throws ArgumentError population_posterior(result, b; warmup_frac = 1.0)
end

@testset "Population posterior plot" begin
    binned = bin(b, runs)
    result = run_imh(rng, binned)
    post = population_posterior(result, b)

    # Edges are read from the binning, so none of these pass them by hand.
    @test population_posterior_plot(post) isa Figure
    @test population_posterior_plot(result, b; warmup_frac = 0.3) isa Figure
    @test population_posterior_plot(post.lambda, post.P_geq[1, :], b) isa Figure

    # Injection-truth overlay path.
    truths = (
        log_P_yr = [0.5], log_q = [-1.0],
        log_P_yr_lo = [0.3], log_P_yr_hi = [0.7],
        n_stars = length(binned.star_names),
    )
    @test population_posterior_plot(post; truths) isa Figure
end

@testset "Poor-sensitivity masking" begin
    binned = bin(b, runs)
    result = run_imh(rng, binned)
    post = population_posterior(result, b)

    sens = relative_sensitivities(binned, 1e-3)
    @test length(sens) == b.n_bins
    thresh = sum(sens) / length(sens)   # mask roughly the most prior-driven half

    @test population_posterior_plot(post; sensitivity = sens,
                                    sensitivity_threshold = thresh) isa Figure
    @test population_posterior_plot(result, b; sensitivity = sens,
                                    sensitivity_threshold = thresh) isa Figure

    # threshold is mandatory when sensitivity is given
    @test_throws ArgumentError population_posterior_plot(post; sensitivity = sens)
    # wrong-length sensitivity vector is rejected
    @test_throws DimensionMismatch population_posterior_plot(post;
        sensitivity = sens[1:end-1], sensitivity_threshold = thresh)
end
