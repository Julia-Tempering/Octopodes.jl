""" 
$SIGNATURES 
"""
function numerical_posterior_plot(posterior::Vector, true_proportion = nothing)
    grid = build_grid(posterior)
    fig, ax, _ = lines(grid, posterior)
    ax.xlabel = L"\psi" 
    ax.ylabel = "posterior density"
    if !isnothing(true_proportion)
        vlines!(ax, [true_proportion], color=:green, linestyle=:dash)
    end
    return fig
end

function joint_detection_prior_sensitivity_synth_plot()
    psi_some_companion_truth = 0.8 
    generated = Octopodes.generate_binary_indep_runs(;
            psi_some_companion_truth, 
            n_systems = 100000,
            n_systems_iters = 10000,
            mcmc_lazy_pr = 0.5)

    # Only the curves corresponding to a no-companion are non-trivial
    s = findfirst(iszero, generated.x_truth)

    fig, ax1, ax2 = _joint_detection_sensitivity_plot()

    bayes_optimal = Octopodes.synthetic_local_posterior(psi_some_companion_truth, false)
    hlines!(ax1, [bayes_optimal[2]], linestyle = :dash, color = :gray, linewidth = 1.5)

    indices = collect(1:n_systems)
    indices[1], indices[s] = indices[s], indices[1]
    series = joint_detection_sensitivity_by_n_systems(generated.runs, indices) 
    len = length(series.posteriors) 
    xs = 2 .^ (1:len)
    lines!(ax1, xs, series.posteriors)
    lines!(ax2, xs, abs.(series.derivatives))
    return fig
end

function joint_detection_prior_sensitivity_real_plot(real_data::BinnedIndepRuns)
    fig, ax1, ax2 = _joint_detection_sensitivity_plot()
    rng = Xoshiro(42)
    for _ in 1:100
        p = randperm(rng, length(real_data.star_names))
        series = joint_detection_sensitivity_by_n_systems(real_data, p) 
        len = length(series.posteriors) 
        xs = 2 .^ (1:len)
        lines!(ax1, xs, series.posteriors)
        lines!(ax2, xs, abs.(series.derivatives))
    end
    return fig
end

function _joint_detection_sensitivity_plot() 
    fig = Figure(size = (600, 450))
    ax1 = Axis(fig[1, 1], xscale = log10,                 ylabel = L"P(n_1 > 0|y_{1:S})")
    ax2 = Axis(fig[2, 1], xscale = log10, yscale = log10, ylabel = L"|\partial_\mu P_\mu(n_1 > 0|y_{1:S})|", xlabel = L"S")
    hidexdecorations!(ax1, grid = false, ticks = false)
    return fig, ax1, ax2
end


## Reproducing the paper's figures

function all_figures(jld2_file::String)
    real_data = IndepRuns(JLD2.load(jld2_file))
    # TODO: save jld2_file + commit + date
    validation_and_interpretation_figure(real_data) 
    full_run(real_data)
end

plots_folder() = mkpath("results/Planet-Demographics/plots")

function prior_sensitivity_figure()
    output_folder = plots_folder()

    synth_plot = joint_detection_prior_sensitivity_synth_plot() 
    save("$output_folder/prior_sensitivity_figure_synth.png", synth_plot)
end

function validation_and_interpretation_figure(real_data::BinnedIndepRuns) 
    output_folder = plots_folder()

    # real data for comparison and setting number of systems, stars
    real_binarized = binarize(real_data)
    real_local_post = standardized_local_posteriors(real_binarized)
    local_fig, local_ax, _ = lines(real_local_post, label = "Real data")
    n_systems, n_systems_iters = size(real_binarized.samples)
    save_latex_key_values("$output_folder/validation_and_interpretation_figure_variables.tex";
        totalnsystems = n_systems,
        niters = n_systems_iters
    )
    real_posterior_plot = numerical_posterior_plot(numerical(real_binarized))
    save("$output_folder/validation_and_interpretation_figure__real.png", real_posterior_plot; size = (200, 200))
    
    # synthetic data on the same plot
    psis = [0.5, 0.9, 1.0] 
    for psi in psis 
        generated = generate_binary_indep_runs(; psi_some_companion_truth = psi, n_systems, n_systems_iters)
        posterior_plot = posterior_recovery_plot(generated)
        save("$output_folder/validation_and_interpretation_figure__psi=$psi.png", posterior_plot; size = (200, 200))
    
        synth_local_post = standardized_local_posteriors(generated.runs)
        lines!(local_ax, synth_local_post, label = L"Synthetic (\psi^* = $psi)", linestyle = :dash)
    end

    save("$output_folder/validation_and_interpretation_figure__locals.png", local_fig)
    return local_fig
end

save_latex_key_values(path::String; kwargs...) = 
    open(path, "w") do io
        for (k, v) in kwargs
            println(io, "\\newcommand{\\$k}{$v}")
        end
    end

function posterior_recovery_plot(generated, eps = default_eps)
    binned = generated.runs 
    true_proportion = generated.psi_some_companion_truth 
    posterior = numerical(binned, eps)
    return numerical_posterior_plot(posterior, true_proportion)
end

function full_run(real_data)
    output_folder = plots_foler()

    # real data for comparison and setting number of systems, stars
    real_binned = bin(Binning(real_data, n_log_P_yr_intervals = 1, n_log_q_intervals = 1), real_data)
    
    rng = Xoshiro(1)
    results = run_imh(rng, real_binned)
    @show mean(results.psi_trace, dims = 2)
    return results
end


## ──────────────────────────────────────────────────────────────────────────
## Population (joint-π) posterior: summary + heatmap
## ──────────────────────────────────────────────────────────────────────────



function _hdi(samples; credible_mass = 0.75)
    s = sort(samples)
    n = length(s)
    n_inc = floor(Int, credible_mass * n)
    widths = [s[i + n_inc] - s[i] for i in 1:(n - n_inc)]
    idx = argmin(widths)
    return (s[idx], s[idx + n_inc])
end

function _bin_counts(values::AbstractVector{<:Real}, edges::AbstractVector{<:Real})
    counts = zeros(Int, length(edges) - 1)
    for v in values
        isfinite(v) || continue
        (v < edges[1] || v > edges[end]) && continue
        i = clamp(searchsortedlast(edges, v), 1, length(counts))
        counts[i] += 1
    end
    return counts
end

function _step_xy(edges::AbstractVector{<:Real}, counts::AbstractVector{<:Real})
    xs = Float64[]; ys = Float64[]
    for i in eachindex(counts)
        push!(xs, edges[i]);     push!(ys, counts[i])
        push!(xs, edges[i + 1]); push!(ys, counts[i])
    end
    return xs, ys
end

# Build a (n_log_P × n_log_q) boolean mask of bins to blank, from a per-bin
# relative-sensitivity vector (as from `relative_sensitivities`) and a threshold.
function _sensitivity_mask(binning::Binning, sensitivity, threshold)
    sensitivity === nothing && return nothing
    threshold === nothing && throw(ArgumentError(
        "`sensitivity_threshold` is required when `sensitivity` is supplied"))
    length(sensitivity) == binning.n_bins || throw(DimensionMismatch(
        "sensitivity has length $(length(sensitivity)), expected n_bins = $(binning.n_bins)"))
    return vector_to_array(binning, collect(sensitivity)) .> threshold
end

"""
$(SIGNATURES)

Plot the joint-π population posterior: a heatmap of the per-bin companion rate
density ``\\lambda \\cdot 100`` (companions per 100 stars per bin) over the
``(\\log_{10} P, \\log_{10} q)`` grid, with the period- and mass-marginal rates
and ``P(\\ge 1)`` shown as boxplots on the margins.

Three methods are provided, all reading the grid edges from the
[`Binning`](@ref) so they never have to be passed by hand:

- `population_posterior_plot(post::PopulationPosterior; ...)` — the usual call.
- `population_posterior_plot(result, binning; warmup_frac = 0.2, ...)` — runs
  [`population_posterior`](@ref) for you, straight from [`run_imh`](@ref) output.
- `population_posterior_plot(lambda_grid, pi_geq1, binning; ...)` — low-level,
  for a pre-computed rate grid `(n_draws × n_log_P × n_log_q)` and `P(≥1)` draws.

Keyword arguments:
- `figsize::Tuple = (900, 600)` — figure size in points.
- `outfile::Union{String, Nothing} = nothing` — if set, save the figure there.
- `colorscale = log10` — heatmap colour scale.
- `truths::Union{NamedTuple, Nothing} = nothing` — optional injection truth for an
  injection/recovery diagnostic. Fields: `log_P_yr`, `log_q` (centres, required);
  optional bounds `log_P_yr_lo`/`log_P_yr_hi`/`log_q_lo`/`log_q_hi`; and `n_stars`
  (host count, used to put truth histograms in the same `λ·100` units). When given,
  cyan crosses/errorbars mark each truth on the heatmap and normalised truth
  histograms are overlaid on the marginals.
- `sensitivity::Union{AbstractVector{<:Real}, Nothing} = nothing` — optional per-bin
  relative prior-sensitivity (length `binning.n_bins`), as returned by
  [`relative_sensitivities`](@ref). Bins whose value exceeds `sensitivity_threshold`
  are masked: blanked (drawn in `masked_color`) on the heatmap and excluded from the
  marginal sums, so the figure shows only data-constrained regions. Typical use:
  `sensitivity = relative_sensitivities(binned, 1e-3)`.
- `sensitivity_threshold::Union{Real, Nothing} = nothing` — required when `sensitivity`
  is given; bins with `sensitivity .> sensitivity_threshold` are masked.
- `masked_color = (:grey, 0.5)` — fill colour for masked (poor-sensitivity) bins.

Returns the `Makie.Figure`.
"""
function population_posterior_plot(post::PopulationPosterior; kwargs...)
    return population_posterior_plot(post.lambda, post.P_geq[1, :], post.binning; kwargs...)
end

function population_posterior_plot(result; warmup_frac::Real = 0.2, kwargs...)
    return population_posterior_plot(population_posterior(result; warmup_frac); kwargs...)
end

function population_posterior_plot(
        lambda_grid::AbstractArray{<:Real, 3},
        pi_geq1::AbstractVector{<:Real},
        binning::Binning;
        figsize::Tuple = (900, 600),
        outfile::Union{String, Nothing} = nothing,
        truths::Union{NamedTuple, Nothing} = nothing,
        colorscale = log10,
        sensitivity::Union{AbstractVector{<:Real}, Nothing} = nothing,
        sensitivity_threshold::Union{Real, Nothing} = nothing,
        masked_color = (:grey, 0.5),
    )
    # Bin edges come straight from the binning, already in log₁₀ space.
    log_per_edges  = collect(binning.log_P_yr_grid)
    log_mass_edges = collect(binning.log_q_grid)

    # Optional poor-sensitivity mask (true ⇒ blank the bin), shaped (n_log_P × n_log_q).
    mask = _sensitivity_mask(binning, sensitivity, sensitivity_threshold)

    n_draws       = size(lambda_grid, 1)
    # For the heatmap, blank masked bins (→ NaN, drawn in `masked_color`).
    quantity_mean = float.(mean(lambda_grid, dims = 1)[1, :, :])
    if mask !== nothing
        quantity_mean[mask] .= NaN
    end
    # For the marginals, exclude masked bins from the per-draw sums (set them to 0).
    quantity = lambda_grid
    if mask !== nothing
        quantity = copy(lambda_grid)
        for s in 1:n_draws
            @view(quantity[s, :, :])[mask] .= 0
        end
    end

    fig = Figure(size = figsize)

    # Main heatmap — λ × 100 (companions per 100 stars per bin).
    ax = Axis(fig[2, 1];
        xlabel = "log₁₀ period [yr]",
        ylabel = "log₁₀ mass-ratio",
        xgridvisible = false, ygridvisible = false,
    )
    h = heatmap!(ax, log_per_edges, log_mass_edges, quantity_mean .* 100;
        colorscale, colormap = :seaborn_rocket_gradient, nan_color = masked_color)
    Colorbar(fig[2, 4], h)
    Label(fig[1, 4], "λ·100", tellheight = false, valign = :bottom)

    per_centres  = (log_per_edges[1:end-1]  .+ log_per_edges[2:end])  ./ 2
    mass_centres = (log_mass_edges[1:end-1] .+ log_mass_edges[2:end]) ./ 2

    # Top marginal: sum over mass.
    ta = Axis(fig[1, 1];
        xticksvisible = false, xticklabelsvisible = false,
        xgridvisible = false,  ygridvisible = false,
        ylabel = "∑λ·100\nover mass",
    )
    boxplot!(ta,
        repeat(per_centres, inner = n_draws),
        sum(quantity, dims = 3)[:] .* 100;
        color = :grey, markersize = 3, width = first(diff(log_per_edges)))

    # Right marginal: sum over period.
    ra = Axis(fig[2, 2:3];
        yticksvisible = false,
        xgridvisible = false, ygridvisible = false,
        xticklabelrotation = pi/4,
        xlabel = "∑λ·100 over period",
        yaxisposition = :right,
        yticks = (mass_centres,
            mapslices(sum(quantity, dims = 2), dims = 1) do s
                lo, hi = _hdi(s)
                @sprintf("[%.0f, %.0f]", lo * 100, hi * 100)
            end[:]),
    )
    boxplot!(ra,
        repeat(mass_centres, inner = n_draws),
        sum(quantity, dims = 2)[:] .* 100;
        color = :grey, orientation = :horizontal, markersize = 3,
        width = first(diff(log_mass_edges)))

    # Top-right: P(≥1) boxplot.
    tra = Axis(fig[1, 2:3];
        xticksvisible = false, xticklabelsvisible = false,
        xgridvisible = false,  ygridvisible = false,
        yaxisposition = :right, ylabel = "P(≥1)",
    )
    ylims!(tra, 0, 1)
    boxplot!(tra, fill(-5.5, length(pi_geq1)), pi_geq1; color = :grey, markersize = 3)

    rowsize!(fig.layout, 2, Auto(4))
    colsize!(fig.layout, 1, Auto(8))
    colsize!(fig.layout, 2, Auto(1))
    colsize!(fig.layout, 3, Auto(1/4))
    rowgap!(fig.layout, 1, 10); colgap!(fig.layout, 1, 10)

    linkxaxes!(ax, ta); linkyaxes!(ax, ra)
    xlims!(ta, extrema(log_per_edges))
    ylims!(ra, extrema(log_mass_edges))

    # ── Truth overlay (injection / recovery diagnostic) ──
    if truths !== nothing
        log_P    = collect(truths.log_P_yr)
        log_q    = collect(truths.log_q)
        log_P_lo = haskey(truths, :log_P_yr_lo) ? collect(truths.log_P_yr_lo) : copy(log_P)
        log_P_hi = haskey(truths, :log_P_yr_hi) ? collect(truths.log_P_yr_hi) : copy(log_P)
        log_q_lo = haskey(truths, :log_q_lo)    ? collect(truths.log_q_lo)    : copy(log_q)
        log_q_hi = haskey(truths, :log_q_hi)    ? collect(truths.log_q_hi)    : copy(log_q)
        n_stars  = haskey(truths, :n_stars)     ? truths.n_stars              : length(log_P)

        cyan = RGBAf(0.0, 0.85, 0.95, 0.95)

        scatter!(ax, log_P, log_q;
            color = cyan, marker = :xcross, markersize = 9, strokewidth = 0)
        errorbars!(ax, log_P, log_q, log_P .- log_P_lo, log_P_hi .- log_P;
            direction = :x, color = cyan, whiskerwidth = 4, linewidth = 1.0)
        errorbars!(ax, log_P, log_q, log_q .- log_q_lo, log_q_hi .- log_q;
            direction = :y, color = cyan, whiskerwidth = 4, linewidth = 1.0)

        per_rate  = _bin_counts(log_P, log_per_edges)  .* (100 / max(n_stars, 1))
        mass_rate = _bin_counts(log_q, log_mass_edges) .* (100 / max(n_stars, 1))
        xs_top, ys_top = _step_xy(log_per_edges, per_rate)
        lines!(ta, xs_top, ys_top; color = cyan, linewidth = 1.6)
        ys_r, xs_r = _step_xy(log_mass_edges, mass_rate)
        lines!(ra, xs_r, ys_r; color = cyan, linewidth = 1.6)
    end

    if outfile !== nothing
        CairoMakie.save(outfile, fig)
        @info "Saved $outfile"
    end
    return fig
end