"""
This file provides an optional CUDA-backed path for NVIDIA GPUs.
"""

const _MOMENTUMED_HAS_CUDA = try
    @eval import CUDA
    true
catch
    false
end

"""
CUDA-backed matrix-free linear map.

This stores the scatter list and Hilbert basis list on the active CUDA device while
keeping the host-side HilbertSubspace for dimensions and final CPU-side wrapping.
"""
mutable struct CuLinearMap{bits, F <: AbstractFloat, AS, AT} <: AbstractLinearMap{bits, F}
    scat_list::AS
    space_list::AT
    space::HilbertSubspace{bits}
    threads::Int
    blocks::Int
end

"""
Adjoint view of a CUDA-backed matrix-free linear map.
"""
mutable struct CuAdjointLinearMap{bits, F <: AbstractFloat, AS, AT} <: AbstractLinearMap{bits, F}
    scat_list::AS
    space_list::AT
    space::HilbertSubspace{bits}
    threads::Int
    blocks::Int
end

function adjoint!(A_adj::CuAdjointLinearMap{bits, F, AS, AT})::CuLinearMap{bits, F, AS, AT} where {bits, F <: AbstractFloat, AS, AT}
    CuLinearMap{bits, F, AS, AT}(A_adj.scat_list, A_adj.space_list, A_adj.space, A_adj.threads, A_adj.blocks)
end
function adjoint!(A::CuLinearMap{bits, F, AS, AT})::CuAdjointLinearMap{bits, F, AS, AT} where {bits, F <: AbstractFloat, AS, AT}
    CuAdjointLinearMap{bits, F, AS, AT}(A.scat_list, A.space_list, A.space, A.threads, A.blocks)
end

function adjoint(A_adj::CuAdjointLinearMap{bits, F, AS, AT})::CuLinearMap{bits, F, AS, AT} where {bits, F <: AbstractFloat, AS, AT}
    CuLinearMap{bits, F, AS, AT}(A_adj.scat_list, A_adj.space_list, A_adj.space, A_adj.threads, A_adj.blocks)
end
function adjoint(A::CuLinearMap{bits, F, AS, AT})::CuAdjointLinearMap{bits, F, AS, AT} where {bits, F <: AbstractFloat, AS, AT}
    CuAdjointLinearMap{bits, F, AS, AT}(A.scat_list, A.space_list, A.space, A.threads, A.blocks)
end



@inline function _gpu_launch_dims(n::Int; threads_per_block::Integer = 256,
    blocks::Union{Nothing, Integer} = nothing)

    n >= 1 || throw(ArgumentError("GPU LinearMap requires a non-empty Hilbert subspace."))
    threads = clamp(Int(threads_per_block), 1, min(1024, n))
    launch_blocks = isnothing(blocks) ? cld(n, threads) : Int(blocks)
    launch_blocks >= 1 || throw(ArgumentError("GPU block count must be positive."))
    return threads, launch_blocks
end

function cuda_available()::Bool
    _MOMENTUMED_HAS_CUDA || return false
    return CUDA.functional(false) && CUDA.has_cuda_gpu()
end

function _throw_cuda_unavailable()
    throw(ArgumentError(
        "CUDA-backed LinearMap requires CUDA.jl and a functional CUDA-capable GPU. " *
        "Install CUDA.jl and verify CUDA.functional()/CUDA.has_cuda_gpu() before using method=:cuda_map or :gpu_map."
    ))
end

if _MOMENTUMED_HAS_CUDA
    @eval begin
        @inline function _device_state_index(space_list, target::MBS64{bits}) where {bits}
            lo = 1
            hi = length(space_list)
            target_n = target.n
            while lo <= hi
                mid = (lo + hi) ÷ 2
                mid_n = @inbounds space_list[mid].n
                if mid_n < target_n
                    lo = mid + 1
                elseif mid_n > target_n
                    hi = mid - 1
                else
                    return mid
                end
            end
            return 0
        end

        function _cuda_linearmap_kernel!(y, x, space_list, scat_list)
            i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
            stride = CUDA.blockDim().x * CUDA.gridDim().x
            n = length(space_list)

            while i <= n
                mbs_out = @inbounds space_list[i]
                acc = zero(eltype(y))
                for s in 1:length(scat_list)
                    scat = @inbounds scat_list[s]
                    amp, mbs_in = mbs_out * scat
                    if !iszero(amp)
                        j = _device_state_index(space_list, mbs_in)
                        if j != 0
                            acc += amp * @inbounds x[j]
                        end
                    end
                end
                @inbounds y[i] = acc
                i += stride
            end
            return nothing
        end

        function _cuda_adjoint_linearmap_kernel!(y, x, space_list, scat_list)
            i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
            stride = CUDA.blockDim().x * CUDA.gridDim().x
            n = length(space_list)

            while i <= n
                mbs_out = @inbounds space_list[i]
                acc = zero(eltype(y))
                for s in 1:length(scat_list)
                    scat = @inbounds scat_list[s]
                    amp, mbs_in = scat * mbs_out
                    if !iszero(amp)
                        j = _device_state_index(space_list, mbs_in)
                        if j != 0
                            acc += conj(amp) * @inbounds x[j]
                        end
                    end
                end
                @inbounds y[i] = acc
                i += stride
            end
            return nothing
        end

        function _cuda_random_complex(::Type{F}, m::Int) where {F <: AbstractFloat}
            complex.(CUDA.rand(F, m), CUDA.rand(F, m))
        end

        function CuLinearMap(A::LinearMap{bits, F};
            device_id::Union{Nothing, Integer} = nothing,
            threads_per_block::Integer = 256,
            blocks::Union{Nothing, Integer} = nothing,
        ) where {bits, F <: AbstractFloat}

            cuda_available() || _throw_cuda_unavailable()
            isnothing(device_id) || CUDA.device!(device_id)
            threads, launch_blocks = _gpu_launch_dims(length(A.space);
                threads_per_block = threads_per_block,
                blocks = blocks,
            )
            gpu_scats = CUDA.CuArray(A.scat_list)
            gpu_space = CUDA.CuArray(A.space.list)
            return CuLinearMap{bits, F, typeof(gpu_scats), typeof(gpu_space)}(
                gpu_scats, gpu_space, A.space, threads, launch_blocks,
            )
        end

        function CuLinearMap(op::MBOperator{Complex{F}, MBS64{bits}},
            space::HilbertSubspace{bits}; kwargs...
        ) where {bits, F <: AbstractFloat}
            CuLinearMap(LinearMap(op, space); kwargs...)
        end

        function (A::CuLinearMap{bits, F})(y::CUDA.CuArray{Complex{F}, 1}, x::CUDA.CuArray{Complex{F}, 1}) where {bits, F}
            n = length(A.space)
            _check_linearmap_dims(y, x, n)
            CUDA.@cuda threads=A.threads blocks=A.blocks _cuda_linearmap_kernel!(y, x, A.space_list, A.scat_list)
            return y
        end

        function (A::CuAdjointLinearMap{bits, F})(y::CUDA.CuArray{Complex{F}, 1}, x::CUDA.CuArray{Complex{F}, 1}) where {bits, F}
            n = length(A.space)
            _check_linearmap_dims(y, x, n)
            CUDA.@cuda threads=A.threads blocks=A.blocks _cuda_adjoint_linearmap_kernel!(y, x, A.space_list, A.scat_list)
            return y
        end

        function (A::CuLinearMap{bits, F})(x::CUDA.CuArray{Complex{F}, 1}) where {bits, F}
            y = similar(x)
            A(y, x)
            return y
        end

        function (A::CuAdjointLinearMap{bits, F})(x::CUDA.CuArray{Complex{F}, 1}) where {bits, F}
            y = similar(x)
            A(y, x)
            return y
        end

        function krylov_map_solve(
            H::CuLinearMap{bits, eltype}, N_eigen::Int64;
            ishermitian::Bool = true,
            vec0::Union{Nothing, AbstractVector{Complex{eltype}}} = nothing,
            krylovkit_kwargs...
        ) where {bits, eltype <: AbstractFloat}

            m = length(H.space)
            if isnothing(vec0)
                vec0 = _cuda_random_complex(eltype, m)
            elseif !(vec0 isa CUDA.CuArray{Complex{eltype}, 1})
                vec0 = CUDA.CuArray(vec0)
            end
            N_eigen = min(N_eigen, m)
            eigsolve(H, vec0, N_eigen, :SR; ishermitian, krylovkit_kwargs...)
        end
    end
else
    function CuLinearMap(::LinearMap{bits, F}; kwargs...) where {bits, F <: AbstractFloat}
        _throw_cuda_unavailable()
    end

    function CuLinearMap(::MBOperator{Complex{F}, MBS64{bits}}, ::HilbertSubspace{bits}; kwargs...) where {bits, F <: AbstractFloat}
        _throw_cuda_unavailable()
    end
end