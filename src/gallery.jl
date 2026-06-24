## ──────────────────────────────────────────────────────────────────────────
## Per-system mass–separation "gallery" pages
##
## A faithful, weight-aware port of the multi-companion dotplot panels used in
## the Hipparcos/Gaia companion-demographics papers. Each panel shows one
## system's posterior over up to three companions (b, c, d) in the physical
## mass–separation plane, with marginal histograms, an upper-envelope line, and
## optional overlays of known-companion catalogs (NASA Exoplanet Archive,
## WDS/ORB6 visual orbits, SB9 spectroscopic binaries).
##
## The point of *this* file (vs. the units-agnostic engine in `dotplot.jl`) is
## the joint-population re-weighting: pass a per-draw `weights` vector — e.g. the
## acceptance counts from [`joint_reconstruction_weights`](@ref) — and each draw
## is drawn with opacity ∝ its weight, while the marginal histograms and the
## upper-envelope use the weights too. The result is the system posterior re-
## expressed under the population posterior `p(n_s, x_s | y_{1:S})`. With
## `weights = nothing` the panel is identical to the standalone gallery.
##
## Separation columns are read from `*_sep` / `*_projsep` when present (so an
## enriched input needs no orbit solve) and otherwise reconstructed on the fly
## from the per-draw orbital elements via PlanetOrbits.
## ──────────────────────────────────────────────────────────────────────────

# CairoMakie re-exports most of Makie, but a few names (rich/subscript/Mixed/…)
# are safest fully qualified.
const _MK = CairoMakie.Makie

""" Jupiter masses per solar mass (companion mass = q · M_pri · MJUP_PER_MSUN). """
const MJUP_PER_MSUN = 1047.35

# Marker palette for NEA known-planet overlays (one per planet in a system). Both
# arrays are the same length and indexed with `mod1`, so a system with more
# planets than entries wraps around rather than erroring.
const _GALLERY_NEA_COLORS  = [:red, :blue, :gold, :orange, :purple, :cyan, :black, :pink]
const _GALLERY_NEA_MARKERS = [:circle, :rect, :diamond, :hexagon, :cross, :xcross, :+, :pentagon]

# Posterior scatter colours (companion b/c/d) and the per-draw companion-count
# marker shapes are shared with the units-agnostic dotplot engine — reuse
# `COMPANION_COLORS` / `NCOMP_MARKERS` from dotplot.jl rather than redefining them.

# Rendering caps / envelope tuning, kept as named constants rather than inline
# magic numbers (see `gallery_panel!`).
const _HOST_MASS_DRAW_CAP     = 512      # max per-draw host-mass lines drawn
const _ENVELOPE_MIN_BIN_COUNT = 20       # min draws in a bin to place an envelope point
const _ENVELOPE_QUANTILE      = 0.95     # upper-envelope "compatibility" quantile
const _YAXIS_MASS_MAX         = 5237.86  # mass-axis upper limit [M_jup]

# Convert fractional year (e.g. 2015.5) to Modified Julian Date.
_years2mjd(year::Real) = 51544.5 + (year - 2000.0) * 365.25

# Draw arrows as line segments with triangular tips (log-scale safe). Named with
# a leading underscore to avoid clashing with Makie's own `arrows2d!`.
function _gallery_arrows!(ax, x, y, dx, dy; color = :black, shaftwidth = 1, tipwidth = 5)
    for i in eachindex(x)
        xi  = x  isa Number ? x  : x[i]
        yi  = y  isa Number ? y  : y[i]
        dxi = dx isa Number ? dx : (i <= length(dx) ? dx[i] : dx[end])
        dyi = dy isa Number ? dy : (i <= length(dy) ? dy[i] : dy[end])
        linesegments!(ax, [_MK.Point2f(xi, yi), _MK.Point2f(xi + dxi, yi + dyi)],
                      color = color, linewidth = shaftwidth)
        scatter!(ax, [xi + dxi], [yi + dyi], marker = :utriangle, markersize = tipwidth, color = color)
    end
end

# ESS-aware Silverman bandwidth, matching PairPlots.jl's `default_bandwidth_ess`:
# `0.9 · min(std, IQR/1.34) · n_ess^(-1/5)`, where `n_ess` is the effective
# sample size (MCMCDiagnosticTools.ess) rather than the raw draw count. Because
# MCMC draws are autocorrelated, n_ess ≪ n, which *widens* the kernel — this is
# what gives PairPlots its smooth credible regions (and what a naive
# `default_bandwidth` on the full N would miss). Falls back to n on a degenerate
# ess. The weights don't enter here (PairPlots bandwidths the raw chain too).
function _bandwidth_ess(data::AbstractVector{<:Real}; alpha = 0.9)
    n = length(data)
    n <= 1 && return float(alpha)
    n_ess = try
        e = MCMCDiagnosticTools.ess(data)
        (e <= 1 || !isfinite(e)) ? float(n) : float(e)
    catch
        float(n)
    end
    var_width = std(data)
    q25, q75 = quantile(data, (0.25, 0.75))
    width = min(var_width, (q75 - q25) / 1.34)
    width == 0.0 && (width = var_width == 0.0 ? 1.0 : var_width)
    return alpha * width * n_ess^(-0.2)
end

# Weighted 2-D Gaussian KDE for the :contour encoding, evaluated in the
# already-logged coordinate space (log10 sep/period, log2 mass) where the
# posteriors are roughly elliptical. A smooth *Gaussian* kernel (not a box kernel
# on a coarse histogram) is what removes the boxy, axis-aligned look: at equal
# per-axis bandwidth it is isotropic, so a tilted mass–sep correlation renders as
# a tilted ellipse. The per-axis bandwidth is the PairPlots ESS rule scaled by
# `bw_factor`. The density is evaluated PairPlots-style on an N×N grid spanning
# the data extrema (so the contour math sees the same support PairPlots would).
# Returns grid coords *in linear data units*, the density matrix, and an
# interpolant for evaluating density at the raw draws (the hybrid scatter test).
function _kde_density(logx, logy, w; bw_factor = 1.0, npoints = 100, xbase = 10.0, ybase = 2.0)
    bw = (_bandwidth_ess(logx), _bandwidth_ess(logy)) .* bw_factor
    wts = w === nothing ? Weights(ones(length(logx))) : Weights(Float64.(w))
    k  = KernelDensity.kde((logx, logy); bandwidth = bw, weights = wts)
    ik = KernelDensity.InterpKDE(k)
    xr = range(extrema(logx)...; length = npoints)
    yr = range(extrema(logy)...; length = npoints)
    dens = [KernelDensity.pdf(ik, xi, yi) for xi in xr, yi in yr]
    return xbase .^ collect(xr), ybase .^ collect(yr), dens, ik
end

# Density thresholds for the σ contour bands, matching PairPlots.jl exactly:
# the target fractions are `1 − exp(−½/σ²)`, and each is turned into a density
# threshold by walking the ascending-sorted density cumulatively (so the cell
# area cancels on the uniform grid). Returned ascending, as `contourf!` wants,
# with collisions nudged apart so the band list stays strictly increasing.
function _sigma_levels(density, sigmas)
    fracs = [1 - exp(-0.5 * (1 / s)^2) for s in sigmas]
    d = sort(vec(density))
    sm = cumsum(d); sm[end] > 0 || return Float64[]
    sm = sm ./ sm[end]
    levels = Float64[]
    for f in fracs
        idx = searchsortedlast(sm, f)
        push!(levels, d[clamp(idx, 1, length(d))])
    end
    sort!(levels)
    for i in 2:length(levels)               # enforce strictly increasing
        levels[i] <= levels[i-1] && (levels[i] = nextfloat(levels[i-1]))
    end
    return levels
end

# Format a Bayes factor from a natural-log value, "≥"-capped at 4096.
function _format_bf(lnbf)
    (lnbf === nothing || ismissing(lnbf) || (lnbf isa Real && (isnan(lnbf) || isinf(lnbf)))) && return "—"
    bf = exp(lnbf)
    s = bf > 10 ? @sprintf("%.0f", bf) : bf > 1 ? @sprintf("%.1f", bf) : @sprintf("%.2f", bf)
    return bf >= 4096 ? "≥" * s : s
end

_has(df, c::AbstractString) = c in names(df)
_getfield_or(row, k::Symbol, default) = hasproperty(row, k) ? getproperty(row, k) : default

# Weighted step-histogram counts over `bins`, delegating to the shared
# binary-search histogram in dotplot.jl. `w === nothing` ⇒ unit weights, so the
# unweighted gallery is unchanged. (Step polylines reuse `_dp_step` directly.)
_whist(vals, w, bins) =
    _dp_weighted_hist(vals, w === nothing ? Iterators.repeated(1.0) : w, bins)

"""
$(SIGNATURES)

Expand a per-system posterior `DataFrame` by an integer per-draw `weights`
vector, returning a new `DataFrame` in which draw `i` appears `round(weights[i])`
times. A utility for histogram/quantile code that prefers an explicit multiset;
[`gallery_panel!`](@ref) instead keeps draws distinct and encodes weight as
opacity. `weights === nothing` returns `samples` unchanged.
"""
function reweighted_samples(samples::DataFrame, weights::Union{Nothing, AbstractVector})
    weights === nothing && return samples
    nrow(samples) == length(weights) || throw(DimensionMismatch(
        "weights has length $(length(weights)) but samples has $(nrow(samples)) rows"))
    idx = Int[]
    @inbounds for i in eachindex(weights)
        m = round(Int, weights[i])
        m > 0 && append!(idx, fill(i, m))
    end
    isempty(idx) && return samples
    return samples[idx, :]
end

# Per-companion separation [au] (xaxis_mode=:sep_au), projected sep [mas]
# (:sep_mas) or period [days] (:period) for the active draws of one companion
# slot. Reads precomputed columns when present, else reconstructs from elements.
function _companion_xvals(post_pl::DataFrame, pl::AbstractString, xaxis_mode::Symbol, date)
    sep_col, projsep_col, P_col = "$(pl)_sep", "$(pl)_projsep", "$(pl)_P"
    a_col, e_col, i_col, M_col, tp_col = "$(pl)_a", "$(pl)_e", "$(pl)_i", "$(pl)_M", "$(pl)_tp"
    omega_col, Omega_col = "\$$(pl)_{\\omega}\$", "\$$(pl)_{\\Omega}\$"

    xaxis_mode == :period && return post_pl[!, P_col] .* 365.25
    xaxis_mode == :sep_mas && _has(post_pl, projsep_col) && return Float64.(post_pl[!, projsep_col])
    xaxis_mode == :sep_au  && _has(post_pl, sep_col)     && return Float64.(post_pl[!, sep_col])

    # No precomputed separation column — reconstruct the orbits from the per-draw
    # elements once, then project to the requested observable at a single epoch.
    els = map(eachrow(post_pl)) do r
        orbit(; r.plx, M = r[M_col], a = r[a_col], e = r[e_col],
                i = r[i_col], ω = r[omega_col], Ω = r[Omega_col], tp = r[tp_col])
    end
    date_mjd = mjd(date)
    xaxis_mode == :sep_mas && return projectedseparation.(els, date_mjd)
    return hypot.(posx.(els, date_mjd), posy.(els, date_mjd), posz.(els, date_mjd))
end

"""
$(SIGNATURES)

Render one system's multi-companion mass–separation panel into the grid layout
`gpos` (e.g. `GridLayout(fig[r, c])`).

`samples` is the per-system posterior (one row per MCMC draw, with the
`*_planet_present`, `*_mass`, and either `*_sep` or orbital-element columns).
`name` is the system label; `row` is a summary `NamedTuple` carrying `gaia_id`,
`hip`, `parallax`, and the per-companion log-Bayes-factors `ln_bf`, `ln_bf_c`,
`ln_bf_d`. An optional `row.ipd_frac_multi_peak` (Gaia DR3 percent of IPD windows
with more than one peak) adds a faint top-left "mp<pct>" hint when non-zero.

`weights` (length `nrow(samples)`, or `nothing`) re-weights the draws by the
joint population posterior — each draw is drawn with opacity ∝ weight (robustly
normalised to `weight_quantile`), draws with zero weight are dropped, and the
marginal histograms are weighted accordingly. The black dashed upper-envelope
("compatibility" limit from the non-detected companion slots) deliberately
ignores the weights: it is computed from the system's own unweighted posterior so
it stays a property of the individual system rather than the population. The
companion catalogs `nea`, `wds`, `orb6`, `sb9` are all optional (`nothing` skips them).

Returns the main `Axis`.
"""
function gallery_panel!(gpos, samples::DataFrame, name, row;
        weights::Union{Nothing, AbstractVector} = nothing,
        date = "2015-06-01", hidex::Bool = false, hidey::Bool = false,
        xaxis_mode::Symbol = :sep_au,
        nea = nothing, wds = nothing, orb6 = nothing, sb9 = nothing,
        max_points::Int = 4096, base_alpha::Real = 1.0,
        weight_quantile::Real = 0.9, rng::AbstractRNG = Xoshiro(1),
        weight_encoding::Symbol = :alpha,
        area_ref_quantile::Real = 0.5, area_min_ms::Real = 0.0,
        area_max_ms_factor::Real = 4.0,
        kde_bandwidth::Real = 0.8, contour_sigmas = (1, 2, 3),
        contour_scatter_outliers::Bool = true)

    weights === nothing || length(weights) == nrow(samples) || throw(DimensionMismatch(
        "weights length $(length(weights)) ≠ nrow(samples) $(nrow(samples))"))

    # ── Reduce to the working draw set, carrying weights ──
    # `present_all` is every draw with ≥1 companion (the system's own posterior,
    # used for the system-property upper-envelope below). The scatter/histogram
    # working set additionally drops population-rejected (zero-weight) draws.
    present_all = falses(nrow(samples))
    for pl in ("b", "c", "d")
        col = "$(pl)_planet_present"
        _has(samples, col) && (present_all .|= samples[!, col] .> 0)
    end
    keep  = weights === nothing ? present_all : (present_all .& (weights .> 0))
    work  = samples[keep, :]
    wwork = weights === nothing ? nothing : Float64.(weights[keep])
    if nrow(work) > max_points
        sub = sort(randperm(rng, nrow(work))[1:max_points])
        work = work[sub, :]
        wwork === nothing || (wwork = wwork[sub])
    end

    # Opacity scaling (:alpha encoding): max-weight draw is fully opaque;
    # normalise by a high quantile so a few very-high-multiplicity draws don't
    # wash everything out.
    #
    # Area scaling (:area encoding): instead of fading low-weight draws, encode
    # weight as marker AREA (markersize ∝ √weight), keeping every drawn point
    # fully opaque. The reference weight `area_ref_quantile` (a CENTRAL quantile,
    # default the median) maps to the standalone `base_ms`, so a system the
    # population leaves untouched renders at the same visual density as its
    # standalone panel — only genuinely down-weighted draws shrink away, rather
    # than the whole cloud fading. `wsizeref` is that reference weight.
    wnorm = 1.0
    wsizeref = 1.0
    if wwork !== nothing
        nz = filter(>(0), wwork)
        wnorm    = isempty(nz) ? 1.0 : max(quantile(nz, weight_quantile),  1e-12)
        wsizeref = isempty(nz) ? 1.0 : max(quantile(nz, area_ref_quantile), 1e-12)
    end

    ln_bf    = _getfield_or(row, :ln_bf, NaN)
    gaia_id  = _getfield_or(row, :gaia_id, 0)
    hip      = _getfield_or(row, :hip, 0)
    parallax = _getfield_or(row, :parallax, missing)

    M_pri_vals = _has(work, "M_pri") ? Float64.(work.M_pri) : Float64[]
    M_pri = isempty(M_pri_vals) ? missing : median(M_pri_vals)

    system = (nea !== nothing && nrow(nea) > 0) ?
        nea[findall(h -> !ismissing(h) && h == gaia_id, nea.gaia_dr3_id), :] : nothing

    yticks = (2.0 .^ (-10:12),
        [".001"; ".001"; ".004"; ".008"; ".016"; ".031"; ".063"; ".125"; ".25"; ".5";
         string.(2 .^ (0:12))])

    if xaxis_mode == :sep_au
        xticks = (2.0 .^ (-10:11),
            [".001"; ".001"; ".004"; ".008"; ".016"; ".031"; ".063"; ".125"; ".25"; ".5";
             string.(2 .^ (0:11))])
        xlabel_text = "separation [au] \non $date"
        xlims_range = (1e-2, 2048)
    elseif xaxis_mode == :sep_mas
        xticks = (10.0 .^ (-1:6),
            ["0.1 mas"; "1 mas"; "10 mas"; "100 mas"; "1\""; "10\""; "100\""; "1000\""])
        xlabel_text = "proj. sep.\non $date"
        xlims_range = (0.1, 1e6)
    elseif xaxis_mode == :period
        xticks = (10.0 .^ (-1:8),
            ["0.1"; "1"; "10"; "100"; "1k"; "10k"; "100k"; "1M"; "10M"; "100M"])
        xlabel_text = "period [days]"
        xlims_range = (1e-1, 1e8)
    else
        throw(ArgumentError("unknown xaxis_mode = :$xaxis_mode"))
    end

    a = Axis(gpos[2, 1];
        yscale = log10, xscale = log10,
        xlabel = xlabel_text,
        ylabel = _MK.rich("mass [M", _MK.subscript("jup"), "]"),
        xticklabelrotation = deg2rad(90),
        yticklabelsize = 8, xticklabelsize = 8, xticks, yticks)

    if ismissing(M_pri)
        @warn "Host mass unavailable for $name; panel left blank"
        return a
    end

    # ── Title: display name + nested Bayes factors ──
    bf_b_s = _format_bf(ln_bf)
    bf_c_s = _format_bf(_getfield_or(row, :ln_bf_c, NaN))
    bf_d_s = _format_bf(_getfield_or(row, :ln_bf_d, NaN))

    ta = Axis(gpos[1, 1]; xscale = log10, xticks, titlesize = 8, titlealign = :left)

    display_name = String(name)
    if system !== nothing
        names = filter(!ismissing, system.hostname)
        isempty(names) || (display_name = String(first(names)))
    end
    hip_str = "HIP $(hip)"
    label_text = occursin(string(hip), display_name) ?
        _MK.rich(display_name; font = :bold) :
        _MK.rich(display_name, " ", _MK.rich(hip_str; fontsize = 7, color = :grey); font = :bold)
    Label(gpos[1, 1:2, _MK.Top()], label_text;
          justification = :left, halign = :left, valign = :bottom, fontsize = 10, lineheight = 0.5)
    Label(gpos[1, 1:2, _MK.Top()], _MK.rich("BF ", bf_b_s, "/", bf_c_s, "/", bf_d_s);
          justification = :left, halign = :right, fontsize = 8, lineheight = 0.5)

    ra = Axis(gpos[2, 2]; yscale = log10, yticks, yticklabelsize = 0.1)

    # ── Host-mass band (thin lines per draw) ──
    M_pri_mjup_vals = M_pri_vals .* MJUP_PER_MSUN
    draw_idx = length(M_pri_mjup_vals) > _HOST_MASS_DRAW_CAP ?
        sort(randperm(rng, length(M_pri_mjup_vals))[1:_HOST_MASS_DRAW_CAP]) :
        (1:length(M_pri_mjup_vals))
    # One vectorized hlines! per axis instead of one call per draw.
    host_lines = @view M_pri_mjup_vals[draw_idx]
    hlines!(a,  host_lines, color = (:black, 0.05), linewidth = 0.3)
    hlines!(ra, host_lines, color = (:black, 0.05), linewidth = 0.3)

    # ── Posterior scatter, per companion (opacity ∝ weight) ──
    base_ms = 1.2
    planet_labels = ("b", "c", "d")
    per_planet_xvals  = Dict{String, Vector{Float64}}()
    per_planet_masses = Dict{String, Vector{Float64}}()
    per_planet_w      = Dict{String, Union{Nothing, Vector{Float64}}}()

    for (pl_idx, pl) in enumerate(planet_labels)
        pp_col, mass_col = "$(pl)_planet_present", "$(pl)_mass"
        (_has(work, pp_col) && _has(work, mass_col)) || continue
        mask = work[!, pp_col] .> 0
        count(mask) == 0 && continue
        post_pl = work[mask, :]
        wpl = wwork === nothing ? nothing : wwork[mask]

        xvals  = _companion_xvals(post_pl, pl, xaxis_mode, date)
        masses = Float64.(post_pl[!, mass_col])
        n_pl_vals = Int.(post_pl.n_planets)
        base_c = COMPANION_COLORS[pl_idx]

        for np in sort(unique(n_pl_vals))
            weight_encoding === :contour && break   # density drawn in a later pass
            np_mask = n_pl_vals .== np
            any(np_mask) || continue
            if wpl === nothing
                cols = _MK.RGBAf(base_c.r, base_c.g, base_c.b, base_alpha)
                mss  = base_ms
            elseif weight_encoding === :area
                # Weight → marker AREA: markersize ∝ √(w / wsizeref), full
                # opacity. The √ makes drawn AREA (not radius) linear in weight,
                # so a draw at the reference weight matches `base_ms`. Clamp so a
                # few very-high-multiplicity draws can't balloon into blobs and
                # so near-zero weights collapse to `area_min_ms` (default 0 ⇒
                # they vanish, the area analogue of the alpha→0 fade-out).
                ms_cap = base_ms * area_max_ms_factor
                cols = _MK.RGBAf(base_c.r, base_c.g, base_c.b, base_alpha)
                mss  = [clamp(base_ms * sqrt(w / wsizeref), area_min_ms, ms_cap)
                        for w in wpl[np_mask]]
            else  # :alpha (legacy) — fade opacity by weight, constant size.
                cols = [_MK.RGBAf(base_c.r, base_c.g, base_c.b,
                                  base_alpha * clamp(w / wnorm, 0.0, 1.0)) for w in wpl[np_mask]]
                mss  = base_ms
            end
            scatter!(a, xvals[np_mask], masses[np_mask];
                color = cols, markersize = mss, rasterize = 4,
                marker = get(NCOMP_MARKERS, np, :star5))
        end

        per_planet_xvals[pl]  = collect(Float64, xvals)
        per_planet_masses[pl] = masses
        per_planet_w[pl]      = wpl
    end

    # ── Posterior density contours (:contour weight encoding) ──
    # Instead of one marker per draw, draw filled KDE-density contours per
    # companion, PairPlots.jl-style: a weighted Gaussian KDE (in log space) gives
    # the smooth, correctly-tilted density; bands are placed at enclosed-mass
    # levels (the 2-D 1/2/3-σ credible regions by default) so a confident
    # detection stays a saturated island rather than a faded cloud, and the
    # population update reads as the island moving/shrinking. Draws that fall
    # OUTSIDE the outermost band (long thin tails, outliers) are scattered back in
    # as faint points so they aren't silently dropped — the hybrid contour+scatter
    # used in PairPlots. The standalone panel (`weights = nothing`) gets the same
    # treatment and so is directly comparable.
    #
    # Only DETECTED companion slots (per-companion Bayes factor > 1) are
    # contoured: a non-detected slot's draws smear across the whole prior, and a
    # filled contour would turn that diffuse density into a panel-swamping blob.
    # Those upper-limit slots are instead represented by the black dashed
    # upper-envelope below (built from exactly the BF ≤ 1 companions), so the two
    # are complementary rather than fighting for the same ink.
    if weight_encoding === :contour
        contour_lnbf = Dict("b" => ln_bf,
                            "c" => _getfield_or(row, :ln_bf_c, -Inf),
                            "d" => _getfield_or(row, :ln_bf_d, -Inf))
        for (pl_idx, pl) in enumerate(planet_labels)
            haskey(per_planet_xvals, pl) || continue
            get(contour_lnbf, pl, -Inf) > 0 || continue   # BF > 1 ⇔ ln BF > 0
            xs = per_planet_xvals[pl]; ys = per_planet_masses[pl]; wpl = per_planet_w[pl]
            length(xs) >= _ENVELOPE_MIN_BIN_COUNT || continue
            logx = log10.(max.(xs, 1e-12)); logy = log2.(max.(ys, 1e-12))
            xc, yc, z, ik = _kde_density(logx, logy, wpl;
                                         bw_factor = kde_bandwidth, xbase = 10.0, ybase = 2.0)
            levels = _sigma_levels(z, contour_sigmas)
            isempty(levels) && continue
            base_c = COMPANION_COLORS[pl_idx]
            cmap = [_MK.RGBAf(base_c.r, base_c.g, base_c.b, al)
                    for al in range(0.15, 0.80; length = length(levels))]
            contourf!(a, xc, yc, z; levels, colormap = cmap,
                      extendlow = :transparent, extendhigh = (base_c, 0.85))
            contour!(a, xc, yc, z; levels, color = (base_c, 0.85), linewidth = 0.6)

            # Hybrid: scatter the draws below the outermost band (the tails).
            if contour_scatter_outliers
                lo = first(levels)
                out = [KernelDensity.pdf(ik, logx[i], logy[i]) < lo for i in eachindex(logx)]
                if any(out)
                    if wpl === nothing
                        ocols = _MK.RGBAf(base_c.r, base_c.g, base_c.b, base_alpha)
                    else
                        ocols = [_MK.RGBAf(base_c.r, base_c.g, base_c.b,
                                           base_alpha * clamp(w / wnorm, 0.0, 1.0)) for w in wpl[out]]
                    end
                    scatter!(a, xs[out], ys[out]; color = ocols, markersize = base_ms,
                             rasterize = 4, marker = get(NCOMP_MARKERS, 1, :circle))
                end
            end
        end
    end

    # Separation/period bins, shared by the upper-envelope and the side histograms.
    hist_bins = xaxis_mode == :period  ? 10.0 .^ (-2:8) :
                xaxis_mode == :sep_mas ? 10.0 .^ (-2:7) : 2.0 .^ (-10:20)

    # ── Upper-envelope "compatibility" limit ──
    # The upper-quantile mass-per-bin line from the non-detected companion slots
    # is a property of THIS system's posterior, so it ALWAYS uses the unweighted
    # draws (all present companions, uniform weights) — even when the scatter and
    # histograms are re-weighted. Re-weighting it would make the limit track the
    # population prior rather than the individual system, which is hard to read.
    let
        envdf = samples[present_all, :]
        if nrow(envdf) > max_points
            envdf = envdf[sort(randperm(Xoshiro(1), nrow(envdf))[1:max_points]), :]
        end
        env_x = Dict{String, Vector{Float64}}(); env_m = Dict{String, Vector{Float64}}()
        for pl in planet_labels
            pp_col, mass_col = "$(pl)_planet_present", "$(pl)_mass"
            (_has(envdf, pp_col) && _has(envdf, mass_col)) || continue
            m = envdf[!, pp_col] .> 0
            count(m) == 0 && continue
            sub = envdf[m, :]
            env_x[pl] = collect(Float64, _companion_xvals(sub, pl, xaxis_mode, date))
            env_m[pl] = Float64.(sub[!, mass_col])
        end

        bfs = Dict("b" => exp(ln_bf),
                   "c" => exp(_getfield_or(row, :ln_bf_c, -Inf)),
                   "d" => exp(_getfield_or(row, :ln_bf_d, -Inf)))
        first_nondet = findfirst(pl -> get(bfs, pl, 0.0) <= 1.0, planet_labels)
        if first_nondet !== nothing
            pool_x = Float64[]; pool_m = Float64[]
            for pl in planet_labels[first_nondet:end]
                haskey(env_x, pl) || continue
                append!(pool_x, env_x[pl]); append!(pool_m, env_m[pl])
            end
            if !isempty(pool_x)
                ul_x = Float64[]; ul_y = Float64[]
                for jj in 1:(length(hist_bins) - 1)
                    lo, hi = hist_bins[jj], hist_bins[jj + 1]
                    in_bin = lo .<= pool_x .< hi
                    count(in_bin) == 0 && continue
                    push!(ul_x, sqrt(lo * hi))
                    push!(ul_y, count(in_bin) >= _ENVELOPE_MIN_BIN_COUNT ?
                                quantile(pool_m[in_bin], _ENVELOPE_QUANTILE) : NaN)
                end
                length(ul_x) >= 2 && lines!(a, ul_x, ul_y; color = :black, linewidth = 2, linestyle = :dot)
            end
        end
    end

    # ── NEA known-planet overlay ──
    if system !== nothing && nrow(system) > 0
        for (j, planet_) in enumerate(groupby(system, :pl_name))
            planet_ = copy(planet_)
            nea_color  = _GALLERY_NEA_COLORS[mod1(j, length(_GALLERY_NEA_COLORS))]
            nea_marker = _GALLERY_NEA_MARKERS[mod1(j, length(_GALLERY_NEA_MARKERS))]
            if xaxis_mode == :period
                pl_xvals = copy(planet_.pl_orbper)
            else
                pl_xvals = copy(planet_.pl_orbsmax)
                miss = ismissing.(pl_xvals)
                pl_xvals[miss] .= cbrt.(planet_.st_mass[miss] .* (planet_.pl_orbper[miss] ./ 365.25) .^ 2)
                if xaxis_mode == :sep_mas && !ismissing(parallax) && parallax > 0
                    pl_xvals = pl_xvals .* parallax
                end
            end
            ii  = .!ismissing.(planet_.pl_massj)
            ii2 = .!ismissing.(planet_.pl_msinij)
            jj  = .!ii .&& .!ii2 .&& .!ismissing.(planet_.pl_radj)
            planet_.pl_msinij[jj] .= (r -> 0.3 * r^2.0).(planet_.pl_radj[jj])
            ii2 = .!ismissing.(planet_.pl_msinij)

            if any(ii)
                pp = planet_[ii, :]; xv = identity.(pl_xvals[ii])
                any(ismissing.(xv)) || scatter!(a, xv, identity.(pp.pl_massj);
                    markersize = 14, strokewidth = 0.8, color = :transparent,
                    strokecolor = nea_color, marker = nea_marker)
            end
            if any(ii2)
                pp = planet_[ii2, :]; msini = identity.(pp.pl_msinij); xv = identity.(pl_xvals[ii2])
                if !any(ismissing.(xv))
                    scatter!(a, xv, msini; markersize = 14, strokewidth = 0.8, color = :transparent,
                        strokecolor = nea_color, marker = nea_marker)
                    l = fill(16, length(msini)); l[msini .< 0.01] .= 64
                    _gallery_arrows!(a, xv, msini, [0], l .* msini;
                        color = nea_color, shaftwidth = 1, tipwidth = 5)
                end
            end
            if !any(ii) && !any(ii2) && !any(jj) && !ismissing(pl_xvals[1])
                _gallery_arrows!(a, [pl_xvals[1]], [1e-5], [0], [13];
                    color = nea_color, shaftwidth = 1, tipwidth = 5)
            end
        end
    end

    # ── WDS/ORB6 visual orbits ──
    found_orbit = false
    if orb6 !== nothing && nrow(orb6) > 0
        for wr in eachrow(filter(r -> r.gaia_id == gaia_id, orb6))
            a_au = wr.sma_arcsec * 1000 / wr.parallax
            M_total = a_au^3 / wr.period_years^2
            any(ismissing, [wr.eccentricity, wr.t0_year, wr.argperi_deg, wr.node_deg,
                            wr.inc_deg, wr.parallax, M_total]) && continue
            found_orbit = true
            o = orbit(; e = wr.eccentricity, tp = _years2mjd(wr.t0_year), a = a_au,
                        ω = deg2rad(wr.argperi_deg), Ω = deg2rad(wr.node_deg),
                        i = deg2rad(wr.inc_deg), plx = wr.parallax, M = M_total)
            xv = xaxis_mode == :period  ? period(o) :
                 xaxis_mode == :sep_mas ? projectedseparation(o, mjd(date)) :
                 (sol = orbitsolve(o, mjd(date)); hypot(posx(sol), posy(sol), posz(sol)))
            vlines!(a, [xv]; color = :blue, linewidth = 2.5, linestyle = :dot)
        end
    end

    # ── WDS separation-only entries ──
    if !found_orbit && wds !== nothing && nrow(wds) > 0 && !ismissing(parallax) && parallax > 0
        for wr in eachrow(filter(r -> r.gaia_id == gaia_id, wds))
            (ismissing(wr.n_obs) || wr.n_obs <= 1 || occursin("U", something(wr.notes, ""))) && continue
            (ismissing(wr.sep_last) || isnothing(wr.sep_last)) && continue
            xaxis_mode == :period && continue
            xv = xaxis_mode == :sep_mas ? wr.sep_last * 1000 : wr.sep_last * (1e3 / parallax)
            lw = 1.5
            m1, m2 = wr.mag1, wr.mag2
            (!isnothing(m1) && !ismissing(m1) && !isnothing(m2) && !ismissing(m2)) &&
                (lw = (m2 - m1) < 4 ? 2.5 : 1.5)
            vlines!(a, [xv]; color = :blue, linewidth = lw, linestyle = :dot)
            if !isnothing(wr.epoch_last) && !ismissing(wr.epoch_last)
                text!(a, xv, 512; text = string(wr.epoch_last), fontsize = 8, color = :blue,
                      rotation = pi / 2, justification = :right, align = (:right, :bottom))
            end
        end
    end

    # ── SB9 spectroscopic-binary overlay ──
    if sb9 !== nothing && nrow(sb9) > 0 && !ismissing(M_pri)
        for sr in eachrow(filter(r -> !ismissing(r.HIP) && r.HIP == hip, sb9))
            P_d, K1, ecc = sr.Period_d, sr.K1_kms, sr.ecc
            a_au = cbrt(M_pri * (P_d / 365.25)^2)
            fm = 1.0361e-7 * K1^3 * P_d * (1 - ecc^2)^1.5
            m2sini = cbrt(fm * M_pri^2)
            for _ in 1:10
                m2sini = cbrt(fm * (M_pri + m2sini)^2)
            end
            m2sini_mjup = m2sini * MJUP_PER_MSUN
            xv = xaxis_mode == :period  ? P_d :
                 xaxis_mode == :sep_mas ? (!ismissing(parallax) && parallax > 0 ? a_au * parallax : nothing) :
                 a_au
            isnothing(xv) && continue
            scatter!(a, [xv], [m2sini_mjup]; markersize = 10, strokewidth = 1.2,
                color = :transparent, strokecolor = :blue, marker = :utriangle)
            K2 = sr.K2_kms
            if !ismissing(K2) && !isnothing(K2) && K2 > 0
                # SB2: connect M2 sin i up to the mass-ratio mass M2 = (K1/K2)·M_pri.
                m2_sb2_mjup = (K1 / K2) * M_pri * MJUP_PER_MSUN
                m2_sb2_mjup > m2sini_mjup && linesegments!(a,
                    [_MK.Point2f(xv, m2sini_mjup), _MK.Point2f(xv, m2_sb2_mjup)];
                    color = :blue, linewidth = 1.5, linestyle = :dash)
            else
                # SB1: M2 sin i is a lower limit — draw an upward arrow.
                l = m2sini_mjup < 0.01 ? 64 : 16
                _gallery_arrows!(a, [xv], [m2sini_mjup], [0], [l * m2sini_mjup];
                    color = :blue, shaftwidth = 1, tipwidth = 5)
            end
        end
    end

    # ── Side histograms (per companion, weighted) ──
    # `hist_bins` (separation/period) is shared with the upper-envelope above.
    mass_bins = 2.0 .^ (-10:13)

    for (pl_idx, pl) in enumerate(planet_labels)
        haskey(per_planet_xvals, pl) || continue
        pl_color = COMPANION_COLORS[pl_idx]
        wpl = per_planet_w[pl]
        xs, ys = _dp_step(hist_bins, _whist(per_planet_xvals[pl], wpl, hist_bins))
        lines!(ta, xs, ys; color = pl_color)
        # Right margin is mass-vertical, so swap the step's (edge, count) → (count, mass).
        mass_edge, mass_w = _dp_step(mass_bins, _whist(per_planet_masses[pl], wpl, mass_bins))
        lines!(ra, mass_w, mass_edge; color = pl_color)
    end

    # Host-mass histogram (black) on the right margin (weighted; same axis swap).
    host_edge, host_w = _dp_step(mass_bins, _whist(M_pri_mjup_vals, wwork, mass_bins))
    lines!(ra, host_w, host_edge; color = :black)

    # ── Gaia DR3 IPD multi-peak fraction (very subtle hint when non-zero) ──
    # `ipd_frac_multi_peak` (percent of IPD windows with >1 peak) is a quiet flag
    # for a possibly unresolved/blended companion. When the summary `row` carries
    # it and it is non-zero, drop a faint glyph + "mp<pct>" label in the
    # top-left corner whose opacity grows with the value (clamped to a low,
    # deliberately understated range) so it reads as metadata, not a data point.
    # Drawn in axis-relative space so it ignores the log scales. `row`s without
    # the field (or a zero/missing value) render exactly as before.
    let ipd_mp = _getfield_or(row, :ipd_frac_multi_peak, missing)
        if ipd_mp !== nothing && !ismissing(ipd_mp) && ipd_mp isa Real && ipd_mp > 0
            # ~50% opaque at mp=5, fully opaque by mp=15: a ≳15% multi-peak
            # fraction is effectively a certain second peak (it just varies by
            # scan direction), so it should read at full strength.
            al = clamp(0.25 + 0.05 * ipd_mp, 0.25, 1.0)
            scatter!(a, [_MK.Point2f(0.045, 0.945)]; space = :relative,
                     marker = :diamond, markersize = 6, color = (:grey15, al))
            text!(a, _MK.Point2f(0.085, 0.945); space = :relative,
                  text = @sprintf("mp%d", round(Int, ipd_mp)), align = (:left, :center),
                  fontsize = 6, color = (:grey15, al))
        end
    end

    # ── Limits, linking, decorations ──
    ylims!(a, 1e-2, _YAXIS_MASS_MAX); xlims!(a, xlims_range...)
    xlims!(ta, xlims_range...); ylims!(ta, low = 0)
    ylims!(ra, 1e-2, _YAXIS_MASS_MAX); xlims!(ra, low = 0)
    linkxaxes!(a, ta); linkyaxes!(a, ra)
    hidedecorations!(ra, grid = false); hidedecorations!(ta, grid = false)
    hidex && hidexdecorations!(a, grid = false)
    hidey && hideydecorations!(a, grid = false)

    if gpos isa Figure
        colsize!(gpos.layout, 1, _MK.Auto(8)); rowsize!(gpos.layout, 2, _MK.Auto(8))
        colgap!(gpos.layout, 1, 5); rowgap!(gpos.layout, 1, 5)
    else
        colsize!(gpos, 1, _MK.Auto(8)); rowsize!(gpos, 2, _MK.Auto(8))
        colgap!(gpos, 1, 5); rowgap!(gpos, 1, 5)
    end

    return a
end

# Bottom-of-page legend block, identical to the standalone gallery.
function _gallery_legend!(fig, bottom_row)
    Legend(fig[bottom_row, 1:2],
        [MarkerElement(color = COMPANION_COLORS[1], marker = :circle, markersize = 8),
         MarkerElement(color = COMPANION_COLORS[2], marker = :circle, markersize = 8),
         MarkerElement(color = COMPANION_COLORS[3], marker = :circle, markersize = 8)],
        ["b", "c", "d"], "Planet (posterior)";
        orientation = :horizontal, tellwidth = false, width = _MK.Auto(1.0),
        colgap = 4, patchlabelgap = 4, alignmode = _MK.Mixed(; bottom = 0, top = 0))

    Legend(fig[bottom_row, 3],
        [MarkerElement(color = :gray50, marker = :circle,    markersize = 8),
         MarkerElement(color = :gray50, marker = :utriangle, markersize = 8),
         MarkerElement(color = :gray50, marker = :diamond,   markersize = 8)],
        ["N=1", "N=2", "N=3"], "N_planets";
        orientation = :horizontal, tellwidth = false, width = _MK.Auto(1.0),
        colgap = 4, patchlabelgap = 4, alignmode = _MK.Mixed(; bottom = 0, top = 0))

    Legend(fig[bottom_row, 4:5],
        [LineElement(color = :blue, linewidth = 2, linestyle = :dot),
         LineElement(color = :blue, linewidth = 2, linestyle = :dashdot),
         [MarkerElement(strokecolor = :blue, strokewidth = 1.2, color = :transparent,
                        marker = :utriangle, markersize = 10),
          LineElement(color = :blue, linewidth = 1, linestyle = nothing,
                      points = _MK.Point2f[(0.5, 0), (0.5, 1)])]],
        ["WDS/ORB6 sep.", "CPM sep.", "SB9 min. mass"], "Stellar Companion";
        orientation = :horizontal, nbanks = 2, tellwidth = false, width = _MK.Auto(1.0),
        colgap = 4, patchlabelgap = 4, alignmode = _MK.Mixed(; bottom = 0, top = 0))
    return fig
end

"""
$(SIGNATURES)

Lay out one gallery page (a 5×5 grid of [`gallery_panel!`](@ref)s plus the
legend) from a vector of `entries` and return the `Figure`.

Each entry is a `NamedTuple` with fields `samples` (per-system posterior
`DataFrame`), `name` (system label), `row` (summary NamedTuple) and, optionally,
`weights` (per-draw re-weighting vector; defaults to `nothing`). At most
`ncol × ceil(n/ncol)` entries are drawn; use [`save_gallery`](@ref) to paginate
a longer list. The catalogs `nea`/`wds`/`orb6`/`sb9` and `date`/`xaxis_mode` are
forwarded to every panel.
"""
function gallery_page(entries::AbstractVector;
        date = "2015-06-01", xaxis_mode::Symbol = :sep_au,
        nea = nothing, wds = nothing, orb6 = nothing, sb9 = nothing,
        page_size::Tuple = (850, 1100), ncol::Int = 5, legend::Bool = true,
        rng::AbstractRNG = Xoshiro(1), max_points::Int = 4096, base_alpha::Real = 1.0,
        weight_encoding::Symbol = :alpha,
        area_ref_quantile::Real = 0.5, area_min_ms::Real = 0.0,
        area_max_ms_factor::Real = 4.0,
        kde_bandwidth::Real = 0.8, contour_sigmas = (1, 2, 3),
        contour_scatter_outliers::Bool = true)

    fig = Figure(size = page_size)
    n = length(entries)
    nrows_this = cld(n, ncol)
    row_idx = 1; col_idx = 1
    for e in entries
        gallery_panel!(GridLayout(fig[row_idx, col_idx]), e.samples, e.name, e.row;
            weights = get(e, :weights, nothing),
            date, xaxis_mode, nea, wds, orb6, sb9, max_points, base_alpha, rng,
            weight_encoding, area_ref_quantile, area_min_ms, area_max_ms_factor,
            kde_bandwidth, contour_sigmas, contour_scatter_outliers,
            hidex = row_idx < nrows_this, hidey = col_idx > 1)
        col_idx += 1
        if col_idx > ncol; col_idx = 1; row_idx += 1; end
    end

    if legend
        _gallery_legend!(fig, ncol + 1)
        _MK.rowgap!.((fig.layout,), 1:(ncol - 1), 4)
    end
    return fig
end

"""
$(SIGNATURES)

Paginate `entries` into `stars_per_page`-sized [`gallery_page`](@ref)s, saving
`<outdir>/<prefix>-pNN.png` (and `.pdf` when `pdf = true`) for each. Returns the
vector of saved PNG paths. Entries and the optional catalogs are exactly as for
[`gallery_page`](@ref).
"""
function save_gallery(entries::AbstractVector;
        outdir::AbstractString, prefix::AbstractString = "gallery",
        stars_per_page::Int = 25, pdf::Bool = true, kwargs...)
    mkpath(outdir)
    npages = cld(length(entries), stars_per_page)
    saved = String[]
    for page in 1:npages
        lo = (page - 1) * stars_per_page + 1
        hi = min(page * stars_per_page, length(entries))
        fig = gallery_page(entries[lo:hi]; kwargs...)
        png = joinpath(outdir, @sprintf("%s-p%02d.png", prefix, page))
        CairoMakie.save(png, fig)
        push!(saved, png)
        pdf && CairoMakie.save(joinpath(outdir, @sprintf("%s-p%02d.pdf", prefix, page)), fig)
        @info "Saved gallery page" png stars = hi - lo + 1
    end
    return saved
end

"""
$(SIGNATURES)

Parse the fixed-width WDS summary catalog (`wdsweb_summ2.txt`) into a
`DataFrame`. A convenience so a caller can build the optional `wds` overlay
catalog without re-implementing the column layout.
"""
function load_wds_catalog(filepath::AbstractString)
    lines = readlines(filepath)
    records = NamedTuple[]
    for line in lines[5:end]
        length(line) < 130 && continue
        push!(records, (
            wds_id = strip(line[1:10]), discoverer = strip(line[11:17]),
            component = strip(line[18:22]),
            epoch_first = tryparse(Int, strip(line[24:27])),
            epoch_last  = tryparse(Int, strip(line[29:32])),
            n_obs    = tryparse(Int, strip(line[34:37])),
            pa_first = tryparse(Int, strip(line[39:41])),
            pa_last  = tryparse(Int, strip(line[43:45])),
            sep_first = tryparse(Float64, strip(line[47:51])),
            sep_last  = tryparse(Float64, strip(line[53:57])),
            mag1 = tryparse(Float64, strip(line[59:63])),
            mag2 = tryparse(Float64, strip(line[65:69])),
            spectral = strip(line[71:79]),
            pm_ra1 = tryparse(Int, strip(line[81:84])), pm_dec1 = tryparse(Int, strip(line[85:88])),
            pm_ra2 = tryparse(Int, strip(line[90:93])), pm_dec2 = tryparse(Int, strip(line[94:97])),
            dm_number = strip(line[99:106]), notes = strip(line[108:111]),
            precise_coord = strip(line[113:130])))
    end
    return DataFrame(records)
end
