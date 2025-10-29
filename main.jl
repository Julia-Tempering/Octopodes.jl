using JLD2, CairoMakie, LogExpFunctions, Pigeons, DynamicPPL, Distributions, 
    Turing, Enzyme, LogDensityProblems, LogDensityProblemsAD, ForwardDiff, 
    LinearAlgebra, Random

root_dir = @__DIR__

function load_data() 
    data_file = "$root/all-data.jld2"
    if !isfile(data_file)
        @info "Decompressing data" 
        cd(root_dir) do 
            run(`unzip all-data.jld2.zip`)
        end
    end
    dict = load("all-data.jld2")
    return dict["d"]
end



n_systems(data) = size(data.q_x_i)[2]

data = load_data()

subset(size) =
    (;  log_Ni = data.log_Ni[1:size], 
        log_Zi = data.log_Zi[1:size],
        q_i = data.q_i,
        q_x_i = data.q_x_i[:, 1:size],
    )

function vector_to_matrix(vector::AbstractVector)
    n_mass_bins = 5 
    n_period_bins = 6 
    @assert length(vector) == n_mass_bins * n_period_bins 
    return reshape(vector, (n_period_bins, n_mass_bins))
end

specific_model(i) = plot_hist(vec(data.q_x_i[:, i]))

function indep_models()
    averaged = sum(data.q_x_i; dims = 2) / n_systems(data)
    plot_hist(vec(averaged))
end

function plot_hist(m)
    @assert isa(m, AbstractVector) || (size(m, 1) == 1) || (size(m, 2) == 1)
    matrix = vector_to_matrix(vec(m)) 
    @show size(matrix)
    fig, ax, hm = heatmap(matrix)
    ax.xlabel = "Period bin"
    ax.ylabel = "Mass ratio bin"
    Colorbar(fig[1, 2], hm)
    fig
end

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

function check_pop_hierarch() 
    #data = subset(100) 
    ldp1 = ldp_with_grad(pop_hierarch_reference(data.q_x_i, data.q_i, data.log_Zi, data.log_Ni)) 
    
    preprocessed = preprocess(data)
    ldp2 = ldp_with_grad(pop_hierarch(preprocessed))  

    dim = LogDensityProblems.dimension(ldp1)
    point = randn(MersenneTwister(1), dim) 
    @show LogDensityProblems.logdensity_and_gradient(ldp1, point)[2] 
    @show LogDensityProblems.logdensity_and_gradient(ldp2, point)[2]
    @show LogDensityProblems.logdensity(ldp1, point) - LogDensityProblems.logdensity(ldp2, point)
    @show preprocessed.log_normalization_offset
end

function ldp_with_grad(model)
    ldp = DynamicPPL.LogDensityFunction(model)
    DynamicPPL.link!!(ldp.varinfo, model)
    return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ldp)
end

function preprocess(data)
    n_bins, n_stars = size(data.q_x_i)
    coefficients = zeros(n_bins + 1, n_stars) 
    log_normalization_offset = 0.0
    for i in 1:n_stars
        for x in 1:n_bins
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


function log_sum_exp(x, y) 
    m = min(x, y)
    M = max(x, y) 
    return M + log1p(exp(m - M))
end

function logd(data, eta, pi)
    K = size(data.q_x_i, 1)
    n_stars = size(data.q_x_i, 2)
    result = 0.0
    for i in 1:n_stars
        current = -Inf 
        for x in 1:K
            current = log_sum_exp(current, log(pi) + log(eta[x]) + data.log_Zi[i] + log(data.q_x_i[x,i]) - log(data.q_i[x]))
        end
        result += current
    end
    return result
end

function mock_manual(vec, mtx)
    n_systems, n_bins = size(mtx)
    result = 0.0
    for system in 1:n_systems 
        inner = 0.0 
        for k in 1:n_bins 
            inner += mtx[system, k] * vec[k]
        end
        result += log(inner)
    end 
    return result

end

function mock_mm(vec, mtx)
    # 0.000383 seconds (4 allocations: 812.594 KiB)
    res = mtx * vec 
    sum(log.(res))
end

function mock_mmt(vec, mtx)
    # 0.000383 seconds (4 allocations: 812.594 KiB)
    res = vec * mtx
    sum(log.(res))
end

function ldp_with_grad(model)
    ldp = DynamicPPL.LogDensityFunction(model)
    DynamicPPL.link!!(ldp.varinfo, model)
    return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ldp)
end

function bench_evals(data)

    dim = 30
    ldp_with_grad1 = ldp_with_grad(pop_hierarch_reference(data.q_x_i, data.q_i, data.log_Zi, data.log_Ni))

    println("DynamicPPL (primal)")
    p = randn(dim)
    for i in 1:5
        @time LogDensityProblems.logdensity(ldp_with_grad1, p)
    end

    println("DynamicPPL2 (primal)")
    ldp_with_grad2 = ldp_with_grad(pop_hierarch(preprocess(data)))
    for i in 1:5
        @time LogDensityProblems.logdensity(ldp_with_grad2, p)
    end

    println("DynamicPPL (FwdDiff)")
    p = randn(dim)
    for i in 1:5
        @time LogDensityProblems.logdensity_and_gradient(ldp_with_grad1, p)
    end

    println("DynamicPPL2 (FwdDiff)")
    ldp_with_grad2 = ldp_with_grad(pop_hierarch(preprocess(data)))
    for i in 1:5
        @time LogDensityProblems.logdensity_and_gradient(ldp_with_grad2, p)
    end


    println("Pure Julia (primal)")
    for i in 1:5 
        @time logd(data, rand(dim), rand())
    end
    println("Pure Julia (Enzyme)")
    for i in 1:5 
        buff = zeros(dim)
        @time Enzyme.autodiff(Enzyme.Reverse, logd, Const(data), Duplicated(rand(dim), buff), Const(rand()))
    end
    v = rand(30) 
    m = rand(52000, 30)
    println("Mock_mm")
    @assert mock_mm(v, m) ≈ mock_manual(v, m)
    for i in 1:5 
        @time x = mock_mm(v, m)
    end
    # println("Mock_mm (Enzyme)")
    # @assert mock_mm(v, m) ≈ mock_manual(v, m)

    vt = rand(30)'
    mt = rand(30, 52000)
    println("Mock_mmt")
    for i in 1:5 
        @time x = mock_mmt(vt, mt)
    end

    # println("Mock_mmt (Enzyme)")
    
    # for i in 1:5 
    #     dv = zeros(30)
    #     @time x = Enzyme.autodiff(Enzyme.Reverse, mock_mm, Duplicated(v, dv), Const(m))
    # end
    println("Mock_manual")
    for i in 1:5 
        @time x = mock_manual(v, m)
    end
    println("Mock_manual (Enzyme)")
    dv = zeros(30)
    for i in 1:5 
        dv = zeros(30)
        @time x = Enzyme.autodiff(Enzyme.Reverse, mock_manual, Duplicated(v, dv), Const(m))
    end
end

run_pigeons(data) = pigeons(;
                        target = TuringLogPotential(pop_hierarch(preprocess(data))),
                        n_chains = 16,
                        multithreaded = false,
                        n_rounds = 10,
                        record = [traces; record_default()]
                    )

function run_turing(data)
    model = pop_hierarch(preprocess(data))
    sample(model, NUTS(), 1000)

end