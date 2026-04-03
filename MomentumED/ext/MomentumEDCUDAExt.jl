module MomentumEDCUDAExt

    using MomentumED
    using CUDA
    using KrylovKit
    using LinearAlgebra

    function __init__()
        if CUDA.functional()
            MomentumED.Methods.CUDA_AVAILABLE[] = true
        else
            @warn "CUDA.jl is loaded but no functional GPU detected. GPU methods disabled."
        end
    end

    using MomentumED.Methods: _check_linearmap_dims, _gpu_launch_dims, _throw_cuda_unavailable

    # CuLinearMap and CuAdjointLinearMap
    using MomentumED.Methods: LinearMap, AbstractCuLinearMap
    mutable struct CuLinearMap{bits, F <: AbstractFloat} <: AbstractCuLinearMap{bits, F}
        scat_amp::CuVector{Complex{F}}
        scat_in::CuVector{UInt64}
        scat_out::CuVector{UInt64}

        space_list::CuVector{UInt64}
        space::HilbertSubspace{bits}

        threads::Int64
        blocks::Int64
    end
    mutable struct CuAdjointLinearMap{bits, F <: AbstractFloat} <: AbstractCuLinearMap{bits, F}
        scat_amp::CuVector{Complex{F}}
        scat_in::CuVector{UInt64}
        scat_out::CuVector{UInt64}

        space_list::CuVector{UInt64}
        space::HilbertSubspace{bits}
        
        threads::Int64
        blocks::Int64

        function CuAdjointLinearMap(A::CuLinearMap{bits, F}) where {bits, F <: AbstractFloat}
            new{bits, F}(A.scat_amp, A.scat_in, A.scat_out, A.space_list, A.space, A.threads, A.blocks)
        end
    end

    import Base: adjoint
    """
    (docstring needed)
    """
    function adjoint(A_adj::CuAdjointLinearMap{bits, F})::CuLinearMap{bits, F} where {bits, F <: AbstractFloat}
        reinterpret(CuLinearMap{bits, F}, A_adj)
    end
    function adjoint(A::CuLinearMap{bits, F})::CuAdjointLinearMap{bits, F} where {bits, F <: AbstractFloat}
        reinterpret(CuAdjointLinearMap{bits, F}, A)
    end

    # multiplication kernels on GPU
    @inline function _device_binary_search(basis::CuDeviceVector{UInt64}, target::UInt64, N::Int)
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
    @inline function _device_scat_occ_number(mbs_n::UInt64, i_mask::UInt64)
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
    function _cuda_linearmap_kernel!(y, x, basis, scat_amp, scat_in, scat_out)
        N_H = length(basis)
        N_scat = length(scat_amp)
        
        i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        i > N_H && return nothing

        state_n = @inbounds basis[i]
        acc = zero(eltype(y))

        for s in 1:N_scat
            in_mask  = @inbounds scat_in[s]
            out_mask = @inbounds scat_out[s]
            amp      = @inbounds scat_amp[s]

            # Are all "in" orbitals occupied?
            (state_n & in_mask != in_mask) && continue

            if in_mask == out_mask
                # Diagonal scatter
                acc += amp * @inbounds x[i]
                continue
            end

            # Annihilate, then check target orbitals are empty
            mid_n = state_n & ~in_mask
            (mid_n & out_mask != UInt64(0)) && continue

            # Create
            out_n = mid_n | out_mask

            # Fermi sign
            sign_count = _device_scat_occ_number(mid_n, in_mask) +
                        _device_scat_occ_number(mid_n, out_mask)
            sign = iseven(sign_count) ? 1 : -1

            j = _device_binary_search(basis, out_n, N_H)
            if j != 0
                acc += (sign * amp) * @inbounds x[j]
            end
        end

        @inbounds y[i] = acc
        return nothing
    end
    function _cuda_adjoint_linearmap_kernel!(y, x, basis, scat_amp, scat_in, scat_out)
        N_H = length(basis)
        N_scat = length(scat_amp)

        i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        i > N_H && return nothing

        state_n = @inbounds basis[i]
        acc = zero(eltype(y))

        for s in 1:N_scat
            # Adjoint: swap in ↔ out, conjugate amplitude
            in_mask  = @inbounds scat_out[s]   # swapped
            out_mask = @inbounds scat_in[s]    # swapped
            amp      = @inbounds conj(scat_amp[s])

            (state_n & in_mask != in_mask) && continue

            if in_mask == out_mask
                acc += amp * @inbounds x[i]
                continue
            end

            mid_n = state_n & ~in_mask
            (mid_n & out_mask != UInt64(0)) && continue
            out_n = mid_n | out_mask

            sign_count = _device_scat_occ_number(mid_n, in_mask) +
                        _device_scat_occ_number(mid_n, out_mask)
            sign = iseven(sign_count) ? 1 : -1

            j = _device_binary_search(basis, out_n, N_H)
            if j != 0
                acc += (sign * amp) * @inbounds x[j]
            end
        end

        @inbounds y[i] = acc
        return nothing
    end

    # Constructors, create from CPU linear map or MBOperator
    import MomentumED.Methods: create_CuLinearMap
    function create_CuLinearMap(A::LinearMap{bits, F};
        device_id::Union{Nothing, Integer} = nothing,
        threads_per_block::Integer = 256,
        blocks::Union{Nothing, Integer} = nothing) where {bits, F}

        isnothing(device_id) || CUDA.device!(device_id)
        threads, launch_blocks = _gpu_launch_dims(
            length(A.space); threads_per_block, blocks
        )

        # Flatten scatter list into struct-of-arrays on GPU
        h_amp = Complex{F}[s.Amp for s in A.scat_list]
        h_in  = UInt64[s.in.n for s in A.scat_list]
        h_out = UInt64[s.out.n for s in A.scat_list]

        # Basis as raw UInt64 sorted array
        h_basis = UInt64[mbs.n for mbs in A.space.list]

        return CuLinearMap{bits, F}(
            CuArray(h_amp), CuArray(h_in), CuArray(h_out), CuArray(h_basis),
            A.space, threads, launch_blocks
        )
    end
    # function CuLinearMap(op::MBOperator{Complex{F}, MBS64{bits}},
    #         space::HilbertSubspace{bits}; kwargs...) where {bits, F}
    #     CuLinearMap(LinearMap(op, space); kwargs...)
    # end

    # ── Callable (launch kernels) ──
    function (A::CuLinearMap{bits, F})(y::CuVector{Complex{F}}, x::CuVector{Complex{F}}) where {bits, F}
        @cuda threads=A.threads blocks=A.blocks _cuda_linearmap_kernel!(
            y, x, A.space_list, A.scat_amp, A.scat_in, A.scat_out)
        return y
    end

    function (A::CuAdjointLinearMap{bits, F})(y::CuVector{Complex{F}}, x::CuVector{Complex{F}}) where {bits, F}
        @cuda threads=A.threads blocks=A.blocks _cuda_adjoint_linearmap_kernel!(
            y, x, A.space_list, A.scat_amp, A.scat_in, A.scat_out)
        return y
    end

    function (A::CuLinearMap{bits, F})(x::CuVector{Complex{F}}) where {bits, F}
        y = similar(x)
        A(y, x)
        return y
    end

    function (A::CuAdjointLinearMap{bits, F})(x::CuVector{Complex{F}}) where {bits, F}
        y = similar(x)
        A(y, x)
        return y
    end

    import MomentumED.Methods: krylov_map_solve
    function krylov_map_solve(
        H::AbstractCuLinearMap{bits, F}, N_eigen::Int64;
        ishermitian::Bool=true,
        vec0::Union{Nothing, AbstractVector{Complex{F}}}=nothing,
        krylovkit_kwargs...) where {bits, F}

        m = length(H.space)
        if isnothing(vec0)
            vec0 = complex.(CUDA.rand(F, m), CUDA.rand(F, m))
        elseif !(vec0 isa CuVector)
            vec0 = CuArray(vec0)
        end
        N_eigen = min(N_eigen, m)

        previous_threads = KrylovKit.get_num_threads()
        KrylovKit.set_num_threads(1) 
        results = eigsolve(H, vec0, N_eigen, :SR; ishermitian, krylovkit_kwargs...)
        KrylovKit.set_num_threads(previous_threads)
        return results
    end


    # gpu memory release
    import MomentumED.Methods: release_cuda_after_eigsolve!
    function release_cuda_after_eigsolve!(level::Int = 1)
        level >= 3 && CUDA.synchronize()
        level >= 1 && GC.gc()
        level >= 2 && CUDA.reclaim()
    end



    # in-place restart in krylov eigsolve
    import KrylovKit: basistransform!, OrthonormalBasis
    import LinearAlgebra: mul!
    function basistransform!(b::OrthonormalBasis{CuArray{T, 1, CUDA.DeviceMemory}}, U::AbstractMatrix) where {T}
        m, n = size(U)
        m == length(b) || throw(DimensionMismatch())
        
        N = length(b[1])
        chunk = min(N, MomentumED.Methods.CUDA_KRYLOV_INPLACE_RESTART_CHUNKSIZE[])
        
        buf_in  = CuArray{T}(undef, chunk, m)
        buf_out = CuArray{T}(undef, chunk, n)
        U_gpu   = CuArray(T.(U))
        
        for start in 1:chunk:N
            len = min(chunk, N - start + 1)
            stop = start + len - 1
            
            for j in 1:m
                copyto!(view(buf_in, 1:len, j), view(b[j], start:stop))
            end
            
            mul!(view(buf_out, 1:len, :), view(buf_in, 1:len, :), view(U_gpu, :, 1:n))
            
            for j in 1:n
                copyto!(view(b[j], start:stop), view(buf_out, 1:len, j))
            end
        end
        
        CUDA.unsafe_free!(buf_in)
        CUDA.unsafe_free!(buf_out)
        CUDA.unsafe_free!(U_gpu)

        CUDA.memory_status()

        
        # resize!(b, n)
        return b
    end


    # Gabbage collection after shrinking and restart
    # import KrylovKit: shrink!, LanczosFactorization
    # function shrink!(state::LanczosFactorization{CuVector{T}, S}, k; verbosity::Int = KrylovDefaults.verbosity[])
    #     length(state) == length(state.V) ||
    #         error("we cannot shrink LanczosFactorization without keeping Lanczos vectors")
    #     length(state) <= k && return state
    #     V = state.V
    #     while length(V) > k + 1
    #         pop!(V)
    #     end
    #     r = pop!(V)
    #     resize!(state.αs, k)
    #     resize!(state.βs, k)
    #     state.k = k
    #     β = KrylovKit.normres(state)
    #     if verbosity > KrylovKit.EACHITERATION_LEVEL
    #         @info "Lanczos reduction to dimension $k: subspace normres = $(KrylovKit.normres2string(β))"
    #     end
    #     state.r = KrylovKit.scale!!(r, β)

    #     # add GPU memory cleanup here
    #     GC.gc()
    #     CUDA.reclaim()
    #     CUDA.memory_status()
    #     return state
    # end







end