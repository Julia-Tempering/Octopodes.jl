struct JointDetection
    per_system_membership_sums::Matrix{Float64} 
    burnin_fraction::Float64 
    """
    $SIGNATURES
    For each system, keep track of sufficient statistics 
    to compute the detection posterior probability given 
    all the data. 
    """
    function JointDetection(n_systems::Int, max_n_companions::Int, burnin_fraction = 0.5)
        @assert 0 ≤ burnin_fraction < 1 
        return new(zeros(max_n_companions, n_systems), burnin_fraction)
    end
end
(jt::JointDetection)(context) =
    if context.iter / context.n_iters ≥ jt.burnin_fraction
        states = context.states
        for system_index in eachindex(states) 
            n_comp = states[system_index].n_companions
            jt.per_system_membership_sums[n_comp + 1, system_index] += 1.0  
        end
    end

function posterior_detection(jt::JointDetection, n_companion::Int, system_index::Int)
    slice = @view jt.per_system_membership_sums[:, system_index]
    if !(sum(slice) ≈ 1)
        normalize!(slice)
    end
    return slice[n_companion + 1]
end


struct JointReconstuction{T}
    """ Same sizes as IMH proposal matrix: system x iteration """
    states_trace::T
end

"""
$SIGNATURES 

Keep in memory, for each system and IMH iteration, of the 
indicator on the number of companions, and the companion binned states. 
"""
JointReconstuction(binned::BinnedIndepRuns) = JointReconstuction(copy(binned.samples))
function (jr::JointReconstuction)(context)
    jr.states_trace[:, context.iter] = context.states 
end

"""
$SIGNATURES

The `lambda` argument is a function that takes in a 
[`BinnedSample`](@ref) and compute a statistic. 
This function will return apply it to all samples, and 
compute a mean for each system. 
"""
function joint_reconstructions(lambda::Function, jr::JointReconstuction, burnin_fraction = 0.5)
    @assert 0 ≤ burnin_fraction < 1 
    _, n_iters = size(jr.states_trace) 
    first = ceil(Int, n_iters * burnin_fraction) + 1
    burnedin_traces = @view(jr.states_trace[:, first:end]) 
    broadcast_lambda(x) = lambda.(x)
    return mean(broadcast_lambda, eachcol(burnedin_traces))
end

"""
$SIGNATURES
"""
function joint_reconstruction_weights(jr::JointReconstuction) 
    transposed_states_trace = permutedims(jr.states_trace) 

    n_samples, n_systems = size(transposed_states_trace)
    result = zeros(n_samples, n_systems)
    
    for system_index in 1:n_systems
        for iter in 1:n_samples 
            imh_sample = transposed_states_trace[iter, system_index]
            indep_run_sample_index = imh_sample.indep_trace_index 
            result[indep_run_sample_index, system_index] += 1
        end
    end
    return result 
end