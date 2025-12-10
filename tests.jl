using CUDA, BenchmarkTools, KernelAbstractions 

@kernel function log_mul_kernel!(@Const(coeff), @Const(vector), buffer)
    i = @index(Global) # index for this call
    sum = 0.0
    for j in eachindex(vector)
        sum += vector[j] * coeff[j, i]
    end
    buffer[i] = log(sum) 
end

function test(kernel, coeff, vector, buffer, backend)
    n_tasks = length(buffer)
    kernel(coeff, vector, buffer, ndrange = n_tasks)
    KernelAbstractions.synchronize(backend)
    return sum(buffer)  # that's already 65 micro sec
end

#= 

Can get it down to:
julia> include("tests.jl"); ka_test()
  117.416 μs (157 allocations: 3.88 KiB)

 which is only about 3x faster than result on laptop
=#

function ka_test()
    backend = CUDABackend() 
    coeff = copy_to_device(rand(30, 52000), backend)
    vector = copy_to_device(rand(30), backend) 
    buffer = copy_to_device(zeros(52000), backend) 
    kernel = log_mul_kernel!(backend) 
    @btime test($kernel, $coeff, $vector, $buffer, $backend)
end

# function log_mul!(coef, backend)
#     work_array = copy_to_device(array, backend)
#     n_tasks = length(array)
#     kernel = sin_kernel!(backend) 
#     @time begin 
#         kernel(work_array, ndrange = n_tasks) # Note "ndrange" argument: it tells KA how many times to launch sin_kernel!
#         KernelAbstractions.synchronize(backend) # avoid usual benchmarking trap
#     end
#     return work_array
# end

function f(coeff_gpu, vector_buffer_gpu, my_vector_cpu)
    #copyto!(vector_buffer_gpu, my_vector_cpu)
    #result = sum(log.(vector_buffer_gpu * coeff_gpu))
    CUDA.synchronize()
    #return result
end

function cuda_test()
    coeff_gpu = CUDA.rand(30, 52000) 
    vector_buffer_gpu = CUDA.zeros(1, 52000)
    my_vector_cpu = rand(1, 30)
    @btime f($coeff_gpu, $vector_buffer_gpu, $my_vector_cpu)
end