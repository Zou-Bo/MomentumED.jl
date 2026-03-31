"""
This file provides an optional CUDA-backed path for NVIDIA GPUs.
Only names and method signatures are defined here; the actual code is in the extension, which is only loaded if CUDA.jl is available.
"""

global const CUDA_AVAILABLE = Ref(false)

"""
CUDA-backed matrix-free linear map. Constructed by the CUDA extension.
"""
abstract type AbstractCuLinearMap{bits, F <: AbstractFloat} <: AbstractLinearMap{bits, F} end

function _throw_cuda_unavailable()
    if CUDA_AVAILABLE[]
        return nothing
    end
    throw(ArgumentError(
        "CUDA-backed LinearMap requires CUDA.jl and a functional CUDA-capable GPU. " *
        "Install CUDA.jl and verify CUDA.functional()/CUDA.has_cuda_gpu() before using method=:cuda_map or :gpu_map."
    ))
end

# Launch dimension helper — pure arithmetic
@inline function _gpu_launch_dims(n::Int; threads_per_block::Integer=256,
    blocks::Union{Nothing, Integer}=nothing)

    n >= 1 || throw(ArgumentError("GPU LinearMap requires a non-empty Hilbert subspace."))
    threads = clamp(Int(threads_per_block), 1, min(1024, n))
    launch_blocks = isnothing(blocks) ? cld(n, threads) : Int(blocks)
    launch_blocks >= 1 || throw(ArgumentError("GPU block count must be positive."))
    return threads, launch_blocks
end

function create_CuLinearMap end