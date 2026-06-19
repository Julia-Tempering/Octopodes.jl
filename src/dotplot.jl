## ──────────────────────────────────────────────────────────────────────────
## Per-system posterior "dotplots"
##
## Visualize the individual MCMC posteriors that feed the population model: one
## scatter point per companion draw, optionally re-weighted by a per-draw weight
## vector (e.g. an importance weight under a fitted population posterior). The
## engine is units-agnostic — you hand it `x` and `y` already in whatever units
## you want on the axes — so the same code draws the native (log P, log q) plane
## (shared with `population_posterior_plot`, hence overlayable) or physical
## mass-vs-separation views built from a richer samples table.
## ──────────────────────────────────────────────────────────────────────────

""" Default per-companion colours (companion identity ``b, c, d, …``). """
const COMPANION_COLORS = [
    RGBf(0.12, 0.47, 0.71),   # b — blue
    RGBf(0.84, 0.15, 0.16),   # c — red
    RGBf(0.17, 0.63, 0.17),   # d — green
    RGBf(1.00, 0.50, 0.00),   # orange
    RGBf(0.58, 0.40, 0.74),   # purple
]

""" Marker shape encodes the number of companions present in a draw. """
const NCOMP_MARKERS = Dict(1 => :circle, 2 => :utriangle, 3 => :diamond, 4 => :pentagon)

# Flatten the (max_n_companions × n_samples) trace matrices into a tidy list of
# present-companion points, carrying each point's companion slot, the draw's
# companion count (→ marker), and the draw's weight (→ opacity & histograms).
function _dp_points(x::AbstractMatrix, y::AbstractMatrix,
                    n_companions::AbstractVector{<:Integer},
                    weights::Union{Nothing, AbstractVector})
    max_comp, n_samples = size(x)
    size(y) == (max_comp, n_samples) || throw(DimensionMismatch(
        "x is $(size(x)) but y is $(size(y)); both must be (max_n_companions × n_samples)"))
    length(n_companions) == n_samples || throw(DimensionMismatch(
        "n_companions has length $(length(n_companions)), expected n_samples = $n_samples"))
    weights === nothing || length(weights) == n_samples || throw(DimensionMismatch(
        "weights has length $(length(weights)), expected one per draw (n_samples = $n_samples)"))

    xs = Float64[]; ys = Float64[]; ws = Float64[]; slot = Int[]; ncomp = Int[]
    for i in 1:n_samples
        nc = n_companions[i]
        wi = weights === nothing ? 1.0 : Float64(weights[i])
        for c in 1:min(nc, max_comp)
            xi = x[c, i]; yi = y[c, i]
            (isfinite(xi) && isfinite(yi)) || continue
            push!(xs, xi); push!(ys, yi); push!(ws, wi); push!(slot, c); push!(ncomp, nc)
        end
    end
    return (; xs, ys, ws, slot, ncomp)
end

function _dp_weighted_hist(values, weights, edges)
    counts = zeros(Float64, length(edges) - 1)
    for (v, w) in zip(values, weights)
        isfinite(v) || continue
        (v < edges[1] || v > edges[end]) && continue
        i = clamp(searchsortedlast(edges, v), 1, length(counts))
        counts[i] += w
    end
    return counts
end

function _dp_step(edges, counts)
    xs = Float64[]; ys = Float64[]
    for i in eachindex(counts)
        push!(xs, edges[i]);     push!(ys, counts[i])
        push!(xs, edges[i + 1]); push!(ys, counts[i])
    end
    return xs, ys
end

# Weighted quantile via the inverse weighted-CDF.
function _dp_weighted_quantile(values, weights, q)
    isempty(values) && return NaN
    perm = sortperm(values)
    v = values[perm]; w = weights[perm]
    cw = cumsum(w); tot = cw[end]
    tot > 0 || return NaN
    j = searchsortedfirst(cw, q * tot)
    return v[clamp(j, 1, length(v))]
end

_dp_edges(lo, hi, n) = lo < hi ? collect(range(lo, hi; length = n + 1)) :
                                 collect(range(lo - 0.5, hi + 0.5; length = n + 1))

"""
$(SIGNATURES)

Scatter a set of per-companion posterior draws onto an existing `ax::Axis`.

This is the units-agnostic core. `x` and `y` are `(max_n_companions × n_samples)`
matrices — the same layout as an [`IndepRuns`](@ref) trace's `log_P_yr`/`log_q` —
holding whatever quantities you want on the axes. `n_companions` (length
`n_samples`) says how many companions are present in each draw; for draw `i`, the
first `n_companions[i]` rows of column `i` are plotted.

`weights` (length `n_samples`, or `nothing`) is the per-draw re-weighting vector:
every companion in draw `i` is drawn with opacity proportional to `weights[i]`
(normalised to its maximum). Pass `nothing` for the plain, unweighted posterior.
Colour encodes companion identity (`colors`); marker shape encodes the draw's
companion count (`ncomp_markers`).

Returns `ax`.
"""
function posterior_dotplot!(ax::Axis,
        x::AbstractMatrix, y::AbstractMatrix, n_companions::AbstractVector{<:Integer};
        weights::Union{Nothing, AbstractVector} = nothing,
        colors = COMPANION_COLORS,
        ncomp_markers = NCOMP_MARKERS,
        markersize::Real = 4,
        base_alpha::Real = 0.6,
        max_points::Int = 4096,
        rng::AbstractRNG = Xoshiro(1),
    )
    pts = _dp_points(x, y, n_companions, weights)
    npts = length(pts.xs)
    npts == 0 && return ax

    # Subsample points uniformly to keep dense panels legible / files small.
    keep = npts > max_points ? sort!(randperm(rng, npts)[1:max_points]) : collect(1:npts)
    wmax = maximum(@view pts.ws[keep])

    for c in sort(unique(@view pts.slot[keep]))
        sel = [i for i in keep if pts.slot[i] == c]
        isempty(sel) && continue
        col = colors[mod1(c, length(colors))]
        cols = map(sel) do i
            a = wmax > 0 ? base_alpha * (pts.ws[i] / wmax) : base_alpha
            RGBAf(col.r, col.g, col.b, a)
        end
        marks = [get(ncomp_markers, pts.ncomp[i], :star5) for i in sel]
        scatter!(ax, pts.xs[sel], pts.ys[sel];
            color = cols, marker = marks, markersize = markersize, rasterize = 4)
    end
    return ax
end

"""
$(SIGNATURES)

Overlay one [`IndepRuns`](@ref) system `trace` onto `ax`. `view` selects which
trace fields map to the axes:

- `:log_P_q` (default) → `(trace.log_P_yr, trace.log_q)`, the same
  ``(\\log_{10} P, \\log_{10} q)`` plane as [`population_posterior_plot`](@ref),
  so a re-weighted system posterior can be drawn directly on the population
  heatmap (see the `Figure` method).

For physical-unit views (mass vs separation) the trace does not carry the needed
columns; call the `(ax, x, y, n_companions)` method with matrices you build from
a richer samples table instead.
"""
function posterior_dotplot!(ax::Axis, trace::NamedTuple;
                            view::Symbol = :log_P_q, weights = nothing, kwargs...)
    x, y = _trace_xy(trace, view)
    return posterior_dotplot!(ax, x, y, trace.n_planets; weights, kwargs...)
end

function _trace_xy(trace::NamedTuple, view::Symbol)
    view === :log_P_q && return (trace.log_P_yr, trace.log_q)
    throw(ArgumentError(
        "view = :$view is not supported for an IndepRuns trace (it only carries " *
        ":log_P_q). For physical units, pass x and y matrices directly."))
end

"""
$(SIGNATURES)

Overlay a system posterior on top of the heatmap of a figure returned by
[`population_posterior_plot`](@ref) — the second of the two calls in the usual
"plot the population model, then drop a re-weighted system onto it" workflow:

```julia
fig = population_posterior_plot(post)
posterior_dotplot!(fig, runs.traces[s]; weights = w[:, s])
```

The scatter is drawn onto the main heatmap `Axis` (found with
[`population_heatmap_axis`](@ref)); `args`/`kwargs` are forwarded to the
`Axis` methods above. Returns `fig`.
"""
function posterior_dotplot!(fig::Figure, args...; kwargs...)
    posterior_dotplot!(population_heatmap_axis(fig), args...; kwargs...)
    return fig
end

"""
$(SIGNATURES)

Return the main heatmap `Axis` of a [`population_posterior_plot`](@ref) figure
(the block at grid position `[2, 1]`). Errors if no `Axis` is found there.
"""
function population_heatmap_axis(fig::Figure)
    axes = filter(b -> b isa Axis, contents(fig[2, 1]))
    isempty(axes) && error(
        "no Axis at fig[2, 1]; expected a figure from population_posterior_plot")
    return first(axes)
end

"""
$(SIGNATURES)

Standalone per-system dotplot: a scatter of the posterior draws with weighted
period- and mass-marginal step histograms and a weighted upper-envelope line
(the per-`x`-bin `upper_quantile` of `y`).

Units-agnostic like [`posterior_dotplot!`](@ref): `x`, `y` are
`(max_n_companions × n_samples)` and `weights` is per-draw (or `nothing`). Set
the axis dressing with `xlabel`/`ylabel`/`xscale`/`yscale`. Returns the `Figure`.
"""
function posterior_dotplot(
        x::AbstractMatrix, y::AbstractMatrix, n_companions::AbstractVector{<:Integer};
        weights::Union{Nothing, AbstractVector} = nothing,
        xlabel = "", ylabel = "", title = "",
        xscale = identity, yscale = identity,
        figsize::Tuple = (480, 380),
        nbins::Int = 30,
        show_marginals::Bool = true,
        upper_limit::Bool = true,
        upper_quantile::Real = 0.95,
        min_bin_weight::Real = 3,
        kwargs...,
    )
    fig = Figure(size = figsize)
    ax = Axis(fig[2, 1]; xlabel, ylabel, title, xscale, yscale,
              xgridvisible = false, ygridvisible = false)

    posterior_dotplot!(ax, x, y, n_companions; weights, kwargs...)

    pts = _dp_points(x, y, n_companions, weights)
    if !isempty(pts.xs) && (show_marginals || upper_limit)
        x_edges = _dp_edges(minimum(pts.xs), maximum(pts.xs), nbins)
        y_edges = _dp_edges(minimum(pts.ys), maximum(pts.ys), nbins)

        if show_marginals
            ta = Axis(fig[1, 1]; xscale,
                      xticksvisible = false, xticklabelsvisible = false,
                      xgridvisible = false, ygridvisible = false, ylabel = "Σw")
            xs_t, ys_t = _dp_step(x_edges, _dp_weighted_hist(pts.xs, pts.ws, x_edges))
            lines!(ta, xs_t, ys_t; color = :grey20)
            linkxaxes!(ax, ta); ylims!(ta, low = 0)

            ra = Axis(fig[2, 2]; yscale,
                      yticksvisible = false, yticklabelsvisible = false,
                      xgridvisible = false, ygridvisible = false, xlabel = "Σw")
            ys_r, xs_r = _dp_step(y_edges, _dp_weighted_hist(pts.ys, pts.ws, y_edges))
            lines!(ra, xs_r, ys_r; color = :grey20)
            linkyaxes!(ax, ra); xlims!(ra, low = 0)

            colsize!(fig.layout, 1, Auto(6)); rowsize!(fig.layout, 2, Auto(6))
            colgap!(fig.layout, 1, 6); rowgap!(fig.layout, 1, 6)
        end

        if upper_limit
            ul_x = Float64[]; ul_y = Float64[]
            for j in 1:(length(x_edges) - 1)
                lo, hi = x_edges[j], x_edges[j + 1]
                in_bin = [lo <= v < hi for v in pts.xs]
                sum(pts.ws[in_bin]) >= min_bin_weight || continue
                push!(ul_x, (lo + hi) / 2)
                push!(ul_y, _dp_weighted_quantile(pts.ys[in_bin], pts.ws[in_bin], upper_quantile))
            end
            length(ul_x) >= 2 && lines!(ax, ul_x, ul_y;
                color = :black, linewidth = 2, linestyle = :dot)
        end
    end
    return fig
end

"""
$(SIGNATURES)

Standalone dotplot for one [`IndepRuns`](@ref) system `trace` in the chosen
`view` (default `:log_P_q`, with sensible axis labels). See the matrix method
for the keyword arguments.
"""
function posterior_dotplot(trace::NamedTuple; view::Symbol = :log_P_q,
                           weights = nothing,
                           xlabel = view === :log_P_q ? "log₁₀ period [yr]" : "",
                           ylabel = view === :log_P_q ? "log₁₀ mass-ratio" : "",
                           title = hasproperty(trace, :name) ? trace.name : "",
                           kwargs...)
    x, y = _trace_xy(trace, view)
    return posterior_dotplot(x, y, trace.n_planets; weights, xlabel, ylabel, title, kwargs...)
end
