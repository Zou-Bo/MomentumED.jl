module MomentumEDoneAPIExt

    using MomentumED
    using oneAPI
    using KrylovKit
    using LinearAlgebra
    using SparseArrays

    # global flags and parameters from main module
    using MomentumED.Methods: GPU_AVAILABLE, DEFAULT_GPU
    using MomentumED.Methods: GPU_RESTART_CHUNKSIZE, GPU_MEMORY_MONITOR

    function __init__()
        println("Checking oneAPI GPU availability...")
        if oneAPI.functional()
            GPU_AVAILABLE[:oneapi][] = true
            if DEFAULT_GPU[] == :nogpu
                DEFAULT_GPU[] = :oneapi
            end
            @info "MomentumED oneAPI extension loaded."
        else
            GPU_AVAILABLE[:oneapi][] = false
            @warn "oneAPI.jl is loaded but no functional Intel GPU detected. GPU methods disabled."
        end
    end
    
    import MomentumED.Methods: activate_oneapi
    function activate_oneapi(; set_default_gpu::Bool = true)
        println("Activating oneAPI GPU device...")
        if oneAPI.functional()
            GPU_AVAILABLE[:oneapi][] = true
            @info "MomentumED oneAPI extension loaded."
            if set_default_gpu 
                DEFAULT_GPU[] = :oneapi
                @info "oneAPI device activated as the default GPU for device=:gpu."
            end
        else
            GPU_AVAILABLE[:oneapi][] = false
            @warn "oneAPI.jl is loaded but no functional Intel GPU detected. GPU methods disabled."
        end
    end

    # gpu memory release
    import MomentumED.Methods: release_gpu_memory
    function release_gpu_memory(::Val{:oneapi}, level::Int64 = 2)
        level >= 1 && GC.gc()
        # oneAPI.jl does not have synchronize() or reclaim() equivalents
        # GC.gc() is sufficient to free unreachable oneArray objects
    end

    # ══════════════════════════════════════════════════════════════
    #  oneAPI LinearMap type definitions
    # ══════════════════════════════════════════════════════════════
    
    using MomentumED.Methods: LinearMap, AbstractGPULinearMap
    const oneVec{T} = oneArray{T, 1} # oneAPI array type alias
    mutable struct oneLinearMap{bits, F <: AbstractFloat} <: AbstractGPULinearMap{bits, F}
        scat_amp::oneVec{Complex{F}}
        scat_in::oneVec{UInt64}
        scat_out::oneVec{UInt64}
        scat_parity::oneVec{UInt64}

        space_list::oneVec{UInt64}
        space::HilbertSubspace{bits}
    end
    mutable struct oneAdjointLinearMap{bits, F <: AbstractFloat} <: AbstractGPULinearMap{bits, F}
        scat_amp::oneVec{Complex{F}}
        scat_in::oneVec{UInt64}
        scat_out::oneVec{UInt64}
        scat_parity::oneVec{UInt64}

        space_list::oneVec{UInt64}
        space::HilbertSubspace{bits}

        function oneAdjointLinearMap(A::oneLinearMap{bits, F}) where {bits, F <: AbstractFloat}
            new{bits, F}(A.scat_amp, A.scat_in, A.scat_out, A.scat_parity, A.space_list, A.space)
        end
    end

    # ── Interface methods ──

    import Base: adjoint, size, eltype
    size(A::Union{oneLinearMap, oneAdjointLinearMap}) = (length(A.space), length(A.space))
    eltype(::Union{oneLinearMap{bits,F}, oneAdjointLinearMap{bits,F}}) where {bits,F} = Complex{F}

    function adjoint(A::oneLinearMap{bits, F}) where {bits, F <: AbstractFloat}
        oneAdjointLinearMap(A)
    end
    function adjoint(A::oneAdjointLinearMap{bits, F}) where {bits, F <: AbstractFloat}
        # Reconstruct the original — shares the same GPU arrays
        oneLinearMap{bits, F}(A.scat_amp, A.scat_in, A.scat_out, A.scat_parity, A.space_list, A.space)
    end
    
    #  Constructor — override the fallback in main module

    import MomentumED.Methods: create_gpu_linearmap
    function create_gpu_linearmap(A::LinearMap{bits, F}, ::Val{:oneapi}) where {bits, F}
        
        # Flatten scatter list into struct-of-arrays on GPU
        h_amp = Complex{F}[s.Amp for s in A.scat_list]
        h_in  = UInt64[s.in.n for s in A.scat_list]
        h_out = UInt64[s.out.n for s in A.scat_list]
        h_parity = UInt64[s.parity_mask for s in A.scat_list]

        # Basis as raw UInt64 sorted array
        h_basis = UInt64[mbs.n for mbs in A.space.list]

        return oneLinearMap{bits, F}(
            oneArray(h_amp),
            oneArray(h_in),
            oneArray(h_out),
            oneArray(h_parity),
            oneArray(h_basis),
            A.space
        )
    end

    #  Callable — uses shared KernelAbstractions kernels

    using MomentumED.Methods: gpu_matvec!, gpu_adjoint_matvec!

    function (A::oneLinearMap{bits, F})(y::oneVec{Complex{F}}, x::oneVec{Complex{F}}) where {bits, F}
        gpu_matvec!(y, x, A.space_list, A.scat_amp, A.scat_in, A.scat_out, A.scat_parity)
        return y
    end

    function (A::oneAdjointLinearMap{bits, F})(y::oneVec{Complex{F}}, x::oneVec{Complex{F}}) where {bits, F}
        gpu_adjoint_matvec!(y, x, A.space_list, A.scat_amp, A.scat_in, A.scat_out, A.scat_parity)
        return y
    end

    function (A::oneLinearMap{bits, F})(x::oneVec{Complex{F}}) where {bits, F}
        y = similar(x)
        A(y, x)
        return y
    end

    function (A::oneAdjointLinearMap{bits, F})(x::oneVec{Complex{F}}) where {bits, F}
        y = similar(x)
        A(y, x)
        return y
    end

    # ══════════════════════════════════════════════════════════════
    #  oneAPI sparse matrix method
    # ══════════════════════════════════════════════════════════════

    using oneAPI.oneMKL: oneSparseMatrixCSR

    import MomentumED.Methods: create_gpu_matrix
    function create_gpu_matrix(H::SparseMatrixCSC, ::Val{:oneapi})
        return oneSparseMatrixCSR(H)
    end
    function create_gpu_matrix(H::Hermitian{C, SparseMatrixCSC{C}}, ::Val{:oneapi}) where {C <: Complex}
        return oneSparseMatrixCSR(sparse(H))
    end

    # ══════════════════════════════════════════════════════════════
    #  KrylovKit solver
    # ══════════════════════════════════════════════════════════════

    import MomentumED.Methods: krylov_map_solve, krylov_matrix_solve
    function krylov_map_solve(
        H::Union{oneLinearMap{bits, F}, oneAdjointLinearMap{bits, F}},
        N_eigen::Int64;
        ishermitian::Bool=true,
        vec0::Union{Nothing, AbstractVector{Complex{F}}}=nothing,
        krylovkit_kwargs...) where {bits, F}

        m = length(H.space)
        if isnothing(vec0)
            vec0 = oneArray(complex.(rand(F, m), rand(F, m)))
        elseif !(vec0 isa oneVec)
            vec0 = oneArray(vec0)
        end
        N_eigen = min(N_eigen, m)

        previous_threads = KrylovKit.get_num_threads()
        KrylovKit.set_num_threads(1)
        results = eigsolve(H, vec0, N_eigen, :SR; ishermitian, krylovkit_kwargs...)
        KrylovKit.set_num_threads(previous_threads)
        return results
    end
    function krylov_matrix_solve(
        H::oneSparseMatrixCSR{Complex{F}, idtype},
        N_eigen::Int64;
        ishermitian::Bool = true,
        vec0::Union{Nothing, AbstractVector{Complex{F}}}=nothing,
        krylovkit_kwargs...) where {F <: AbstractFloat, idtype <: Integer}

        m = size(H, 2)
        if isnothing(vec0)
            vec0 = oneArray(complex.(rand(F, m), rand(F, m)))
        elseif !(vec0 isa oneArray)
            vec0 = oneArray(vec0)
        end
        N_eigen = min(N_eigen, m)

        previous_threads = KrylovKit.get_num_threads()
        KrylovKit.set_num_threads(1)
        results = eigsolve(H, vec0, N_eigen, :SR; ishermitian, krylovkit_kwargs...)
        KrylovKit.set_num_threads(previous_threads)
        return results
    end

    # ══════════════════════════════════════════════════════════════
    #  KrylovKit memory management overrides
    # ══════════════════════════════════════════════════════════════

    # Determine the fully expanded oneArray type for dispatch
    # oneAPI.jl's oneArray may have different type parameters than CUDA's CuArray.
    # Use the actual type from an instance to ensure correct dispatch.

    import KrylovKit: basistransform!, OrthonormalBasis
    import LinearAlgebra: mul!

    # In-place basis transform using chunked matrix multiply
    function basistransform!(b::OrthonormalBasis{<:oneArray{T, 1}}, U::AbstractMatrix) where {T}

        # test
        # println("overriding basistransform! for oneAPI is called with basis length = $(length(b)) and U size = $(size(U))")

        m, n = size(U)
        m == length(b) || throw(DimensionMismatch())

        N = length(b[1])
        chunk = min(N, GPU_RESTART_CHUNKSIZE[])

        buf_in  = oneArray{T}(undef, chunk, m)
        buf_out = oneArray{T}(undef, chunk, n)
        U_gpu   = oneArray(T.(U))

        for start in 1:chunk:N  #  the last chunk if it's smaller than the full size.
            # len = min(chunk, N - start + 1)
            # stop = start + len - 1

            # for j in 1:m
            #     copyto!(view(buf_in, 1:len, j), view(b[j], start:stop))
            # end

            # mul!(view(buf_out, 1:len, :), view(buf_in, 1:len, :), U_gpu)

            # for j in 1:n
            #     copyto!(view(b[j], start:stop), view(buf_out, 1:len, j))
            # end

            if N - start + 1 >= chunk
                len = chunk
                stop = start + len - 1

                for j in 1:m
                    copyto!(view(buf_in, 1:len, j), view(b[j], start:stop))
                end

                mul!(buf_out, buf_in, U_gpu)

                for j in 1:n
                    copyto!(view(b[j], start:stop), view(buf_out, 1:len, j))
                end
            else
                len = N - start + 1
                stop = N

                buf_in  = oneArray{T}(undef, len, m)
                buf_out = oneArray{T}(undef, len, n)

                for j in 1:m
                    copyto!(view(buf_in, 1:len, j), view(b[j], start:stop))
                end

                mul!(buf_out, buf_in, U_gpu)

                for j in 1:n
                    copyto!(view(b[j], start:stop), view(buf_out, 1:len, j))
                end

            end
        end

        # GPU memory monitoring
        GPU_MEMORY_MONITOR[] && println("oneAPI has no real-time memory status printing function.")

        # Free temporary buffers
        buf_in  = nothing
        buf_out = nothing
        U_gpu   = nothing
        GC.gc()

        return b
    end

    # GC after Krylov restart
    import KrylovKit: shrink!, LanczosFactorization
    function shrink!(state::LanczosFactorization{<:oneArray{T, 1}, S}, k;
        verbosity::Int = KrylovDefaults.verbosity[]) where {T <: Complex, S <: Real}

        # test
        # println("overriding shrink! for oneAPI is called with k = $k and current length = $(length(state))")
        
        length(state) == length(state.V) ||
            error("we cannot shrink LanczosFactorization without keeping Lanczos vectors")
        length(state) <= k && return state
        V = state.V
        while length(V) > k + 1
            pop!(V)
        end
        r = pop!(V)
        resize!(state.αs, k)
        resize!(state.βs, k)
        state.k = k
        β = KrylovKit.normres(state)
        if verbosity > KrylovKit.EACHITERATION_LEVEL
            @info "Lanczos reduction to dimension $k: subspace normres = $(KrylovKit.normres2string(β))"
        end
        state.r = KrylovKit.scale!!(r, β)

        # GPU memory cleanup
        GC.gc()

        return state
    end

end
