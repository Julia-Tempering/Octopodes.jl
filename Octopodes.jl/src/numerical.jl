# TODO: might need to change signature a bit to make sure the AD works properly
function numerical(local_companionship_posteriors::Vector{T}, tilde_psi::Vector, psi_prior::Distribution, eps = 0.001) where {T}
    @assert eps > 0 
    @assert all(0 .<= local_companionship_posteriors .<= 1) 
    @assert length(tilde_psi) == 2 && sum(tilde_psi) ≈ 1 
    @assert iscontinuous(psi_prior) && minimum(psi_prior) == 0 && maximum(psi_prior) == 1

    log_tilde_psi_ratio = log(tilde_psi[1]) - log(tilde_psi[2]) 
    to_logBF(local_companionship_posterior) =  
        log(local_companionship_posterior) -
        log1p(-local_companionship_posterior) + 
        log_tilde_psi_ratio

    log_BFs = to_logBF.(local_companionship_posteriors) 

    grid = eps:eps:(1.0-eps) 
    result = zeros(T, length(grid))

    for posterior_discretization_index in eachindex(result)
        psi = grid[posterior_discretization_index]
        sum = logpdf(psi_prior, psi)
        
        for system_index in eachindex(log_BFs) 
            sum += log_BFs[system_index] == Inf ? log(psi) : logsumexp(log(psi) + log_BFs[system_index], log1p(-psi))
        end
        result[posterior_discretization_index] = sum
    end
    exp_normalize!(result)
    result = result*length(pis)
    @assert sum(result) * eps ≈ 1 
        
    return result
end