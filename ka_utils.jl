# create a copy to device of an arbitrary array
copy_to_device(array::AbstractArray{E, N}, backend) where {E, N} = 
    copy_to_device(array, backend, E)
function copy_to_device(array::AbstractArray{E, N}, backend, ::Type{F}) where {E, N, F}
    result = KernelAbstractions.zeros(backend, F, size(array)) 
    copyto!(result, array)
    return result
end 