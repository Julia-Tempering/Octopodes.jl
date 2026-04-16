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

function samples_to_cdf(samples::Vector{Float64})
    n = length(samples)
    @assert n > 0
    xs = sort(samples)
    ys = (1:n) ./ n
    return xs, ys
end

function discrete_pdf_to_cdf(pdf::Vector{Float64})
    n = length(pdf)
    xs = (1:n) ./ n
    ys = cumsum(pdf / (n+1))
    return xs, ys
end

function ks_distance(xs1, ys1, xs2, ys2)
    d1 = DiscreteNonParametric(xs1, diff([0.0; ys1]))
    d2 = DiscreteNonParametric(xs2, diff([0.0; ys2]))

    all_xs = sort(union(support(d1), support(d2)))
    return maximum(x -> abs(Distributions.cdf(d1, x) - Distributions.cdf(d2, x)), all_xs)
end

function ks_p_value(d_stat, n)
    z = sqrt(n) * d_stat
    return 1 - Distributions.cdf(Kolmogorov(), z)
end