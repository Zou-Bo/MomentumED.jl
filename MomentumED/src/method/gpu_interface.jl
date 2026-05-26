"""
    gpu_interface.jl

Backend-agnostic GPU interface for matrix-free exact diagonalization.
Defines abstract types, shared kernels (via KernelAbstractions.jl), and fallback
constructors. Backend-specific code lives in extensions:

    MomentumEDCUDAExt    — NVIDIA GPUs  (CUDA.jl)
    MomentumEDoneAPIExt  — Intel GPUs   (oneAPI.jl)
    MomentumEDMetalExt   — Apple GPUs   (Metal.jl)
    MomentumEDNCCLExt    — Multi-NVIDIA (CUDA.jl + NCCL.jl)

Each extension provides:
    1. __init__()                    — set availability flag
    2. create_GPULinearMap(A, ::Val) — concrete struct + CPU→GPU transfer
    3. callable (A)(y, x) and (A)(x)
    4. krylov_map_solve override
    5. basistransform! override     — memory-efficient Krylov restart
    6. shrink! override             — GC after restart
"""


"""Flags of available GPUs"""
const GPU_AVAILABLE = Dict{Symbol, Ref{Bool}}(
    :cuda_map       => Ref(false),
    :oneapi_map     => Ref(false),
    :metal_map      => Ref(false),
    :multi_cuda_map => Ref(false),
)

"""Size of element chunks in the memory-efficient basistransform! restart."""
const GPU_RESTART_CHUNKSIZE = Ref{Int64}(262144)

"""Enable/disable GPU memory status printing after each Krylov restart."""
const GPU_MEMORY_MONITOR = Ref(false)

function _throw_gpu_unavailable(method::Symbol)
    GPU_AVAILABLE[method][] && return nothing
    throw(ArgumentError(
        "method=:$method requires functional GPU(s) of the type.
        Load the required package(s) and verify GPU availability."
    ))
end


abstract type AbstractGPULinearMap{bits, F <: AbstractFloat} <: AbstractLinearMap{bits, F} end
function create_gpu_linearmap end
function create_gpu_matrix end
function release_gpu_memory(level::Int = 2) end


# krylov_map_solve for AbstractGPULinearMap is defined per-extension
# because each backend needs backend-specific random vector generation.
# Fallback:
function krylov_map_solve(H::AbstractGPULinearMap, N_eigen::Int64; kwargs...)
    error("krylov_map_solve not implemented for $(typeof(H)). Is the correct GPU extension loaded?")
end



"""Binary search on a sorted array. Works on any GPU backend."""
@inline function device_binary_search(basis, target::UInt64, N::Int)
    lo = 1
    hi = N
    while lo <= hi
        mid = (lo + hi) >>> 1
        val = @inbounds basis[mid]
        if val < target
            lo = mid + 1
        elseif val > target
            hi = mid - 1
        else
            return mid
        end
    end
    return 0
end

"""Compute Fermi sign occupation count between orbital pairs."""
@inline function device_scat_occ_number(mbs_n::UInt64, i_mask::UInt64)
    mask = zero(UInt64)
    im = i_mask
    while im > UInt64(0)
        tz1 = trailing_zeros(im)
        im &= im - UInt64(1)
        tz2 = trailing_zeros(im)
        im &= im - UInt64(1)
        mask += UInt64(1) << tz2
        mask -= UInt64(1) << (tz1 + 1)
    end
    return count_ones(mbs_n & mask)
end

"""
    ka_linearmap_kernel!

Backend-agnostic GPU kernel for the matrix-free linear map H|v⟩.
Computes y[i] = Σ_j H_{ij} x[j] for basis state i by looping over
all scatter terms and finding connected states via binary search.
"""
@kernel function linearmap_kernel!(y, @Const(x), @Const(basis),
        @Const(scat_amp), @Const(scat_in), @Const(scat_out))
    i = @index(Global)
    N_H = length(basis)
    N_scat = length(scat_amp)

    state_n = @inbounds basis[i]
    acc = zero(eltype(y))

    for s in 1:N_scat
        in_mask  = @inbounds scat_in[s]
        out_mask = @inbounds scat_out[s]
        amp      = @inbounds scat_amp[s]

        (state_n & in_mask != in_mask) && continue

        if in_mask == out_mask
            acc += amp * @inbounds x[i]
            continue
        end

        mid_n = state_n & ~in_mask
        (mid_n & out_mask != UInt64(0)) && continue
        out_n = mid_n | out_mask

        sign_count = _ka_scat_occ_number(mid_n, in_mask) +
                     _ka_scat_occ_number(mid_n, out_mask)
        sign = iseven(sign_count) ? 1 : -1

        j = _ka_binary_search(basis, out_n, N_H)
        if j != 0
            acc += (sign * amp) * @inbounds x[j]
        end
    end

    @inbounds y[i] = acc
end

"""
    ka_adjoint_linearmap_kernel!

Adjoint version: swaps in ↔ out masks and conjugates amplitude.
"""
@kernel function ka_adjoint_linearmap_kernel!(y, @Const(x), @Const(basis),
        @Const(scat_amp), @Const(scat_in), @Const(scat_out))
    i = @index(Global)
    N_H = length(basis)
    N_scat = length(scat_amp)

    state_n = @inbounds basis[i]
    acc = zero(eltype(y))

    for s in 1:N_scat
        in_mask  = @inbounds scat_out[s]
        out_mask = @inbounds scat_in[s]
        amp      = @inbounds conj(scat_amp[s])

        (state_n & in_mask != in_mask) && continue

        if in_mask == out_mask
            acc += amp * @inbounds x[i]
            continue
        end

        mid_n = state_n & ~in_mask
        (mid_n & out_mask != UInt64(0)) && continue
        out_n = mid_n | out_mask

        sign_count = _ka_scat_occ_number(mid_n, in_mask) +
                     _ka_scat_occ_number(mid_n, out_mask)
        sign = iseven(sign_count) ? 1 : -1

        j = _ka_binary_search(basis, out_n, N_H)
        if j != 0
            acc += (sign * amp) * @inbounds x[j]
        end
    end

    @inbounds y[i] = acc
end

"""
    gpu_matvec!(y, x, A::AbstractGPULinearMap)

Launch the linearmap kernel on the appropriate backend.
Works for any backend supported by KernelAbstractions.jl.
"""
function gpu_matvec!(y, x, basis, scat_amp, scat_in, scat_out)
    backend = KernelAbstractions.get_backend(y)
    kernel! = ka_linearmap_kernel!(backend, 256)
    kernel!(y, x, basis, scat_amp, scat_in, scat_out, ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    return y
end

function gpu_adjoint_matvec!(y, x, basis, scat_amp, scat_in, scat_out)
    backend = KernelAbstractions.get_backend(y)
    kernel! = ka_adjoint_linearmap_kernel!(backend, 256)
    kernel!(y, x, basis, scat_amp, scat_in, scat_out, ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    return y
end