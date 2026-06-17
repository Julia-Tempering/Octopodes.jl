"""
$SIGNATURES 

Family of priors used in numerical prior sensitivity analysis.
"""
prior(mu) = Beta(2mu, 2(1 - mu)) 

"""
$(SIGNATURES) 

Assume that the provided `binned` is binary (see [`is_binary`](@ref)).

Compute the mean of the distribution obtain in [`numerical`](@ref). 
Embed the prior assumed there in `Beta(2mu, 2(1-mu))` family prior 
(i.e., considering perturbations of the prior mean). 
Use `ForwardDiff` to compute the derivative of the posterior mean 
with respect to `mu` at `1/2` (i.e., in the neighbourhood of the uniform prior).
"""
function sensitivity(binned::BinnedIndepRuns, eps = default_eps)
    @assert is_binary(binned)
    lcp = local_companionship_posteriors(binned)
    posterior(mu::T) where {T} = 
        numerical_mean(
            numerical(
                lcp,
                binned.tilde_psi,
                prior(mu), 
                eps, 
                T
            )
        )
    return posterior(0.5), ForwardDiff.derivative(posterior, 0.5)
end

function joint_detection_sensitivities(binned::BinnedIndepRuns, eps = default_eps)
    @assert is_binary(binned)
    lcp = local_companionship_posteriors(binned)
    posterior(mu::T) where {T} = 
        numerical_joint_prediction(
            lcp,
            binned.tilde_psi,
            prior(mu), 
            eps, 
            T
        )
    return posterior(0.5), ForwardDiff.derivative(posterior, 0.5) 
end

function joint_detection_sensitivity_by_n_systems(binned::BinnedIndepRuns, system_index::Int, eps = default_eps)
    @assert is_binary(binned)
    full_lcp = local_companionship_posteriors(binned)
    full_lcp[1], full_lcp[system_index] = full_lcp[system_index], full_lcp[1]

    posteriors = Float64[] 
    derivatives = Float64[] 

    current_size = 1 
    while current_size ≤ length(full_lcp)
        lcp = @view full_lcp[1:current_size] 

        posterior(mu::T) where {T} = 
            numerical_joint_prediction(
                lcp,
                binned.tilde_psi,
                prior(mu), 
                eps, 
                T
            )
        push!(posteriors, posterior(0.5)[1])
        push!(derivatives, ForwardDiff.derivative(posterior, 0.5)[1])

        current_size *= 2
    end
    return (; posteriors, derivatives)
end

"""
$(SIGNATURES) 

For each bin, binarize with respect to that bin using 
[`binarize`](@ref), and compute [`sensitivity`](@ref) for that bin.
Return a vector of relative sensivities, one for each bin.

Relative absolute relative_sensitivities are obtained by 
dividing by the primal value and taking abs values.
"""
relative_sensitivities(binned::BinnedIndepRuns, eps) =
    map(1:binned.binning.n_bins) do k 
        binarized = binarize(binned, k) 
        value, deriv = sensitivity(binarized, eps)
        return abs(deriv / value)
    end

