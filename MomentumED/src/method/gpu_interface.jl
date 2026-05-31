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
    1. __init__() and activate_device()              — set availability flag and current activated GPU flag
    2. release_gpu_memory()                          — GPU-specific memory cleanup
    3. create_gpu_matrix(H, ::Val)                   — sparse matrix CPU→GPU transfer
    4'. define its LinearMap and callable mul        — GPU LinearMap matvec
    4. create_gpu_linearmap(A, ::Val)                — concrete LinearMap struct + CPU→GPU transfer
    5. krylov_map_solve/krylov_matrix_solve          — Call KrylovKit solver with GPU structs
    6. basistransform! override                      — memory-efficient restart in KrylovKit
    7. shrink! override                              — GC after restart in KrylovKit
"""

"""Flags of available GPUs"""
const GPU_AVAILABLE = Dict{Symbol, Ref{Bool}}(
    :cuda       => Ref(false),
    :oneapi     => Ref(false),
    :metal      => Ref(false),
    :multi_cuda => Ref(false),
)

"""Size of element chunks in the memory-efficient basistransform! restart."""
const GPU_RESTART_CHUNKSIZE = Ref{Int64}(262144)

"""Enable/disable GPU memory status printing after each Krylov restart."""
const GPU_MEMORY_MONITOR = Ref{Bool}(false)

"""Track the last activated GPU for device=:gpu. Updated by loading the extension or explicitly activating a GPU device."""
const DEFAULT_GPU = Ref{Symbol}(:nogpu) # tracking the currently used GPU device (for gpu keyword)
"""Activate a specific GPU device. Use device=:gpu in solve() to use the currently active GPU."""
function activate_gpu_device(device::Symbol; set_default_gpu::Bool = true)
    # fallback for unknown device input
    if !(device ∈ keys(GPU_AVAILABLE))
        throw(ArgumentError("Unknown GPU $device. Valid options are: $(keys(GPU_AVAILABLE))."))
    end
    if device == :oneapi
        activate_oneapi(; set_default_gpu)
    elseif device == :cuda
        activate_cuda(; set_default_gpu)
    elseif device == :metal
        activate_metal(; set_default_gpu)
    elseif device == :multi_cuda
        activate_multi_cuda(; set_default_gpu)
    end
end
# fallbacks
function activate_oneapi end
function activate_cuda end
function activate_metal end
function activate_multi_cuda end

"""Check GPU availability for the specified device and throw an error if not available. If device=:gpu, check the default GPU device."""
function _throw_gpu_unavailable(device::Symbol)

    # first try
    GPU_AVAILABLE[device][] && return nothing

    # second try
    try
        activate_gpu_device(device; set_default_gpu = false)
    catch
    end
    GPU_AVAILABLE[device][] && return nothing

    # throw error if still not available
    throw(ArgumentError(
        "device=:$device requires functional GPU(s) of the type.
        Load the required package(s) and verify GPU availability."
    ))
end

abstract type AbstractGPULinearMap{bits, F <: AbstractFloat} <: AbstractLinearMap{bits, F} end

# fallback constructors and release
function create_gpu_linearmap(A, device::Val) end
function create_gpu_matrix(H, device::Val) end
function release_gpu_memory(device::Val, level::Int64 = 2) end

# krylov_map_solve for AbstractGPULinearMap is defined per-extension
# because each backend needs backend-specific random vector generation.
# Fallback:
function krylov_map_solve(H::AbstractGPULinearMap, N_eigen::Int64; kwargs...)
    error("krylov_map_solve not implemented for $(typeof(H)). Is the correct GPU extension loaded?")
end

# basis is the gpu Vector of UInt64 basis states, sorted in ascending order.
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
@inline function device_scat_parity(n_mid::UInt64, scat_parity::UInt64)::Int
    return count_ones(n_mid & scat_parity) & 1
end

using KernelAbstractions
"""
    ka_linearmap_kernel!

Backend-agnostic GPU kernel for the matrix-free linear map H|v⟩.
Computes y[i] = Σ_j H_{ij} x[j] for basis state i by looping over
all scatter terms and finding connected states via binary search.
"""
@kernel function ka_linearmap_kernel!(y, @Const(x), @Const(basis),
        @Const(scat_amp), @Const(scat_in), @Const(scat_out), @Const(scat_parity))
    i = @index(Global)
    N_H = length(basis)
    N_scat = length(scat_amp)

    state_n = @inbounds basis[i]
    acc = zero(eltype(y))

    for s in 1:N_scat
        in_mask  = @inbounds scat_in[s]
        out_mask = @inbounds scat_out[s]
        amp      = @inbounds scat_amp[s]
        parity_mask = @inbounds scat_parity[s]

        (state_n & in_mask != in_mask) && continue

        if in_mask == out_mask
            acc += amp * @inbounds x[i]
            continue
        end

        mid_n = state_n & ~in_mask
        (mid_n & out_mask != UInt64(0)) && continue
        out_n = mid_n | out_mask

        sign = 1 - 2 * device_scat_parity(mid_n, parity_mask) # combine parity and occupation count for total sign

        j = device_binary_search(basis, out_n, N_H)
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
        @Const(scat_amp), @Const(scat_in), @Const(scat_out), @Const(scat_parity))
    i = @index(Global)
    N_H = length(basis)
    N_scat = length(scat_amp)

    state_n = @inbounds basis[i]
    acc = zero(eltype(y))

    for s in 1:N_scat
        in_mask  = @inbounds scat_out[s]
        out_mask = @inbounds scat_in[s]
        amp      = @inbounds conj(scat_amp[s])
        parity_mask = @inbounds scat_parity[s]

        (state_n & in_mask != in_mask) && continue

        if in_mask == out_mask
            acc += amp * @inbounds x[i]
            continue
        end

        mid_n = state_n & ~in_mask
        (mid_n & out_mask != UInt64(0)) && continue
        out_n = mid_n | out_mask

        sign = 1 - 2 * device_scat_parity(mid_n, parity_mask) # combine parity and occupation count for total sign

        j = device_binary_search(basis, out_n, N_H)
        if j != 0
            acc += (sign * amp) * @inbounds x[j]
        end
    end

    @inbounds y[i] = acc
end

"""
    gpu_matvec!(y, x, basis, scat_amp, scat_in, scat_out)

Launch the linearmap kernel on the appropriate backend.
Auto-detects backend from the array type of `y`.
Works for any backend supported by KernelAbstractions.jl.
"""
function gpu_matvec!(y, x, basis, scat_amp, scat_in, scat_out, scat_parity)
    backend = KernelAbstractions.get_backend(y)
    kernel! = ka_linearmap_kernel!(backend, 256)
    kernel!(y, x, basis, scat_amp, scat_in, scat_out, scat_parity, ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    return y
end

function gpu_adjoint_matvec!(y, x, basis, scat_amp, scat_in, scat_out, scat_parity)
    backend = KernelAbstractions.get_backend(y)
    kernel! = ka_adjoint_linearmap_kernel!(backend, 256)
    kernel!(y, x, basis, scat_amp, scat_in, scat_out, scat_parity, ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    return y
end