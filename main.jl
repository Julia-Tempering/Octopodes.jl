using JLD2, CairoMakie, LogExpFunctions, Pigeons, DynamicPPL, Distributions, 
    Turing, Enzyme, LogDensityProblems, LogDensityProblemsAD, ForwardDiff, 
    LinearAlgebra, Random, BenchmarkTools, MCMCChains, Pluto, PlutoUI



## Turing models 

# From William's slack channel message Monday, Oct 27, 12:14 PM
@model function pop_hierarch_reference(q_x_i, q_i, log_Zi, log_Ni)

    K = size(q_x_i, 1)
    n_stars = size(q_x_i, 2)

    # Population parameters
    η ~ Dirichlet(ones(K))
    π ~ Uniform(0, 1)

    # Compute likelihood for each star
    log_contribs = zeros(eltype(η), K+1)
     
    for i in 1:n_stars
        log_contribs .= -Inf
        
        # Null model contribution: (1-π) * Ni
        # log_null_contrib = log(1-π) + log_Ni[i]
        # More numerically stable
        log_null_contrib = log1p(-π) + log_Ni[i]

        # Planet model contributions: π * η[x] * Zi * (q_x_i[x,i] / q_i[x])
        for x in 1:K
            if q_x_i[x,i] > 0 && q_i[x] > 0
                log_contribs[x] = log(π) + log(η[x]) + log_Zi[i] + log(q_x_i[x,i]) - log(q_i[x])
            end
        end

        log_contribs[end] = log_null_contrib
        if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            DynamicPPL.@addlogprob! logsumexp(log_contribs)
        end
    end
end

# Equivalent but faster version:
@model function pop_hierarch(preprocessed)

    # Population parameters
    η ~ Dirichlet(ones(preprocessed.n_bins))
    π ~ Uniform(0, 1)

    # Compute likelihood for each star
    vector = zeros(eltype(η), 1, preprocessed.n_bins + 1)
    for x in 1:preprocessed.n_bins 
        vector[1, x] = π * η[x] 
    end 
    vector[1, end] = (1 - π)
     
    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        DynamicPPL.@addlogprob! sum(log.(vector * preprocessed.coefficients))
    end
end

function preprocess(data)
    n_bins, n_stars = size(data.q_x_i)
    coefficients = zeros(n_bins + 1, n_stars) 
    log_normalization_offset = 0.0 
    @assert sum(data.q_i) ≈ 1 
    for i in 1:n_stars 
        @assert sum(data.q_x_i[:,i]) ≈ 1 
        @assert isfinite(data.log_Zi[i]) 
        @assert isfinite(data.log_Ni[i]) 

        for x in 1:n_bins 
            @assert data.q_x_i[x,i] ≥ 0 
            @assert data.q_i[x] ≥ 0 
            
            coefficients[x, i] = data.q_x_i[x,i] > 0 && data.q_i[x] > 0 ?
                data.log_Zi[i] + log(data.q_x_i[x, i]) - log(data.q_i[x]) :
                -Inf
        end
        coefficients[n_bins + 1, i] = data.log_Ni[i]
        log_normalization_offset += exp_normalize!(@view(coefficients[:, i]))
    end
    return (;
        n_bins, 
        n_stars,
        coefficients,
        log_normalization_offset
    )
end

# check equivalence of the 2
function check_pop_hierarch(data) 
    ldp1 = ldp_with_grad(pop_hierarch_reference(data.q_x_i, data.q_i, data.log_Zi, data.log_Ni)) 
    
    preprocessed = preprocess(data)
    ldp2 = ldp_with_grad(pop_hierarch(preprocessed))  

    dim = LogDensityProblems.dimension(ldp1)
    point = randn(MersenneTwister(1), dim) 

    @assert LogDensityProblems.logdensity_and_gradient(ldp1, point)[2] ≈ 
            LogDensityProblems.logdensity_and_gradient(ldp2, point)[2]

    @assert LogDensityProblems.logdensity(ldp1, point) - LogDensityProblems.logdensity(ldp2, point) ≈ @show preprocessed.log_normalization_offset
end

# compare performance 
function benchmark(data)
    ldp1 = ldp_with_grad(pop_hierarch_reference(data.q_x_i, data.q_i, data.log_Zi, data.log_Ni)) 
    
    preprocessed = preprocess(data)
    ldp2 = ldp_with_grad(pop_hierarch(preprocessed))  

    dim = LogDensityProblems.dimension(ldp1)
    point = randn(MersenneTwister(1), dim) 

    benchmark(ldp1, point, "Reference")  
    benchmark(ldp2, point, "Faster")    
end

function benchmark(ldp, point, test_name)
    println(test_name)
    println(" - primal")
    @btime LogDensityProblems.logdensity($ldp, $point)
    println(" - with gradient (Forward Diff)")
    @btime LogDensityProblems.logdensity_and_gradient($ldp, $point)
    return nothing
end



## Viz

function vector_to_matrix(vector::AbstractVector)
    n_mass_bins = 5 
    n_period_bins = 6 
    @assert length(vector) == n_mass_bins * n_period_bins 
    return reshape(vector, (n_period_bins, n_mass_bins))
end 

matrix_to_vector(matrix::AbstractMatrix) =  vec(matrix) 

function test_vector_matrix_conversion()
    v = collect(1:30) 
    M = vector_to_matrix(v)
    v2 = matrix_to_vector(M)
    M2 = vector_to_matrix(v2)
    @assert v == v2
    @assert M == M2
end

function plot_hist(m)
    matrix = if isa(m, AbstractVector) || (size(m, 1) == 1) || (size(m, 2) == 1)
        vector_to_matrix(vec(m)) 
    else
        @assert size(m) == (6, 5)
        m
    end
    fig, ax, hm = heatmap(matrix)
    ax.xlabel = "Period bin"
    ax.ylabel = "Mass ratio bin"
    Colorbar(fig[1, 2], hm)
    fig
end

specific_model(i, data) = plot_hist(data.q_x_i[:, i])

function indep_models(data)
    averaged = sum(data.q_x_i; dims = 2) / n_systems(data)
    plot_hist(averaged)
end

function just_one_row(row, data)
    _, n_stars = size(data.q_x_i)
    s = zeros(6, 5)
    for i in 1:n_stars
        mtx = vector_to_matrix(data.q_x_i[:, i]) 
        for j in 1:6
            s[j, row] += mtx[j, row] 
        end
    end
    fig, ax, hm = heatmap(s/sum(s))
    ax.xlabel = "Period bin"
    ax.ylabel = "Mass ratio bin"
    Colorbar(fig[1, 2], hm)
    fig
end

function bf_slice(condition, data)
    n_bins, n_stars = size(data.q_x_i)
    sum = zeros(n_bins)
    n = 0
    for i in 1:n_stars 
        log_BF = data.log_Zi[i] - data.log_Ni[i]
        if condition(log_BF)
            sum += data.q_x_i[:, i]
            n += 1
        end
    end
    plot_hist(sum/n)
end

function bf_plot(data, xline = nothing)
    sorted_log_BF = sort(log_BF(data))
    fig, ax, l = lines(-10:0.1:10, x -> searchsortedfirst(sorted_log_BF, x)/length(sorted_log_BF))
    if xline !== nothing
        vlines!(ax, [xline], color=:red, linestyle=:dash)
    end
    return fig
end


function mass_histogram(i, data)
    result = Float64[] 
    # p(y | M_0) 
    push!(result, data.log_Ni[i]) 
    m = vector_to_matrix(data.q_x_i[:, i])
    @assert sum(m) ≈ 1 
    _, n_mass_bins = size(m)
    for mass_index in 1:n_mass_bins 
        # for i > 0,
        # see derivations at bottom of notebook in mass_hist.jl 
        push!(result, data.log_Zi[i] + log(sum(m[:, mass_index])) - log(1/5)) 
    end
    exp_normalize!(result)
    return result
end

function has_mode(list, i)
    if i == 1 
        return list[2] < list[1]
    elseif i == length(list)
        return list[end - 1] < list[end]
    else
        return list[i - 1] < list[i] && list[i + 1] < list[i] 
    end
end

function is_unimodal(list)
    found_mode = false 
    for i in 1:length(list) 
        if has_mode(list, i) 
            if found_mode 
                return false 
            end 
            found_mode = true
        end
    end
    return true
end

function fraction_unimodal_mass_histogram(data)
    n_unimodal = 0 
    _, n_stars = size(data.q_x_i) 
    for i in 1:n_stars 
        if is_unimodal(mass_histogram(i))
            n_unimodal += 1
        end
    end
    return n_unimodal/n_stars
end


## Data loading, preprocessing

root_dir = @__DIR__

function load_data(file_name = "all-data.jld2") 
    data_file = "$root_dir/$file_name"
    if !isfile(data_file)
        @info "Decompressing data" 
        cd(root_dir) do 
            run(`unzip $file_name.zip`)
        end
    end
    dict = load(data_file)
    result = sort_by_bf(dict["d"])
    renormalize(result.q_x_i)
    return result
end

function renormalize(q_x_i)
    n_bins, n_stars = size(q_x_i)
    for star in 1:n_stars 
        old = @view q_x_i[:, star] 
        norm = sum(old)
        q_x_i[:, star]  = old / norm
    end
    return nothing
end

n_systems(data) = size(data.q_x_i)[2]
log_BF(data) = data.log_Zi - data.log_Ni 

function sort_by_bf(data)
    lbfs = log_BF(data)
    perm = sortperm(lbfs)
    return (;
        log_Ni = data.log_Ni[perm],
        log_Zi = data.log_Zi[perm],
        q_i = data.q_i, 
        q_x_i = data.q_x_i[:, perm]
    )
end

real_data = load_data()

synt_data = let
    q_i = fill(1/30, 30)
    q_x_i = zeros(30, 1000)
    q_x_i[1:15,1:end÷2] .= 1/15
    q_x_i[25:30,end÷2+1:end] .= 1/5
    log_Ni = zeros(1000)
    log_Zi = zeros(1000)
    log_Zi[end÷2+1:end] .= log(5)
    renormalize(q_x_i)
    sort_by_bf((;q_i,q_x_i,log_Ni,log_Zi))
end

function subset(indices, data)
    (;  log_Ni = data.log_Ni[indices], 
        log_Zi = data.log_Zi[indices],
        q_i = data.q_i,
        q_x_i = data.q_x_i[:, indices],
    )
end 

function find_selected_indices(vector, subset) 
    as_set = Set(subset)
    return findall(x -> x in as_set, vector)
end

low_mass_sum(matrix::Matrix) = sum(matrix[:, 1]) 

# Symmetrize needs to be rewritten: should adjust also log_Ni to achieve p(y | M_0) = p(y | M_1)

# function symmetrize(matrix::Matrix) 
#     @assert size(matrix) == (6, 5)
#     new_row = ones(6) * low_mass_sum(matrix) / 6 
#     matrix[:, 1] = new_row 
#     return matrix
# end

# function symmetrize(data)
#     new_q = similar(data.q_x_i) 
#     _, n_stars = size(data.q_x_i) 
#     for star in 1:n_stars 
#         m = vector_to_matrix(data.q_x_i[:, star]) 
#         symm = symmetrize(m)
#         new_q[:, star] = matrix_to_vector(symm)
#     end 
#     return (;  
#         log_Ni = data.log_Ni, 
#         log_Zi = data.log_Zi,
#         q_i = data.q_i,
#         q_x_i = new_q
#         )
# end

data_sources() = [:real, :synthetic, :spike_and_slab_nov_7_2025, :dat_44k] 

load_data_source(data_source) = 
    if data_source == :real 
		load_data()
	elseif data_source == :synthetic 
		synt_data 
	elseif data_source == :spike_and_slab_nov_7_2025 
		load_data("spike-and-slab-nov-7-2025.jld2")
    elseif data_source == :dat_44k 
		load_data("dat-44k.jld2")
	else
		error() 
	end


## General utils

function exp_normalize!(log_weights)
    m = maximum(log_weights)
    log_weights .= exp.(log_weights .- m) 
    return m + log(normalize!(log_weights))
end 

function normalize!(weights) 
    s = sum(weights)
    weights .= weights ./ s 
    return s
end

function ldp_with_grad(model)
    ldp = DynamicPPL.LogDensityFunction(model)
    DynamicPPL.link!!(ldp.varinfo, model)
    return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ldp)
end


## Call samplers, postprocessing utilities

run_pigeons(data) = pigeons(;
                        target = TuringLogPotential(pop_hierarch(preprocess(data))),
                        n_chains = 8,
                        multithreaded = true,
                        n_rounds = 10,
                        record = [traces, round_trip]
                    )

function to_chains(pt)
    # temp bug fix
    original_names = sample_names(pt)
    @assert length(original_names) == 31
    fixed_names = original_names[1:29]
    push!(fixed_names, Symbol("η[30]"))
    append!(fixed_names, original_names[30:31])
    array = sample_array(pt)
    @assert sum(array[1, 1:30, 1]) ≈ 1.
    return Chains(array, fixed_names, Dict(:internals => [:log_density]))
end

run_turing(data, n_iters = 1000) = sample(pop_hierarch(preprocess(data)), NUTS(), n_iters)

function trevor_diagnostic(chains)
    n_mcmc_iters, n_params, _ = size(chains) 
    result = Float64[] 
    for i in 1:n_mcmc_iters 
        pi = chains[i, :π, 1] 
        etas = vec(Array(chains[i, 1:30, 1]))
        @assert sum(etas) ≈ 1. 
        sum_undetectable_etas = low_mass_sum(vector_to_matrix(etas))
        ratio = (1 - pi) / (1 - pi + pi * sum_undetectable_etas) 
        push!(result, ratio)
    end
    return result
end 

function run_experiment(use_pigeons::Bool, data)
    chains = use_pigeons ?
                to_chains(run_pigeons(data)) :
                run_turing(data) 
    diagnostic = trevor_diagnostic(chains) 
    return (;
        chains, 
        diagnostic, 
        hist = hist(diagnostic), 
        trace = lines(diagnostic)
    )
end

## Marginal model

planet_probabilities(log_BFs) = 1 ./ (1 .+ exp.(-log_BFs))

function marginal_pi_posterior_density(eps, log_BFs)
    pis = 0.0:eps:1.0 
    result = similar(pis)
    
    for posterior_discretization_index in eachindex(result)
        pi = pis[posterior_discretization_index]
        sum = 0.0 
        for star_index in eachindex(log_BFs) 
            sum += logsumexp(log(pi) + log_BFs[star_index], log1p(-pi))
        end
        result[posterior_discretization_index] = sum
    end
    exp_normalize!(result)
    result = result*length(pis) # make it a density
    return result
end

nothing