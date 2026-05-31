module MomentumEDMetalExt

    using MomentumED
    using Metal
    using KrylovKit
    using LinearAlgebra
    # using SparseArrays

    # global flags and parameters from main module
    using MomentumED.Methods: GPU_AVAILABLE, DEFAULT_GPU
    using MomentumED.Methods: GPU_RESTART_CHUNKSIZE, GPU_MEMORY_MONITOR

    function __init__()
        if Metal.functional()
            GPU_AVAILABLE[:metal][] = true
            if DEFAULT_GPU[] == :nogpu
                DEFAULT_GPU[] = :metal
            end
            @info "MomentumED Metal extension loaded."
        else
            GPU_AVAILABLE[:metal][] = false
            @warn "Metal.jl is loaded but no functional Apple GPU detected. GPU methods disabled."
        end
    end
    import MomentumED.Methods: activate_metal
    function activate_metal(; set_default_gpu::Bool = true)
        println("Activating Metal GPU device...")
        if Metal.functional()
            GPU_AVAILABLE[:metal][] = true
            @info "MomentumED Metal extension loaded."
            if set_default_gpu
                DEFAULT_GPU[] = :metal
                @info "Metal device activated as the default GPU for device=:gpu."
            end
        else
            GPU_AVAILABLE[:metal][] = false
            @warn "Metal.jl is loaded but no functional Apple GPU detected. GPU methods disabled."
        end
    end

    # gpu memory release
    import MomentumED.Methods: release_gpu_memory
    function release_gpu_memory(::Val{:metal}, level::Int64 = 2)
        level >= 1 && GC.gc()
        # Metal.jl frees unreachable MtlArray objects via GC.
        # Metal has no explicit reclaim(); GC.gc() is sufficient.
    end

    # ══════════════════════════════════════════════════════════════
    #  Metal LinearMap type definitions
    # ══════════════════════════════════════════════════════════════

    using MomentumED.Methods: LinearMap, AbstractGPULinearMap
    const MtlVec{T} = MtlArray{T, 1}  # Metal array type alias
    mutable struct MtlLinearMap{bits, F <: AbstractFloat} <: AbstractGPULinearMap{bits, F}
        scat_amp::MtlVec{Complex{F}}
        scat_in::MtlVec{UInt64}
        scat_out::MtlVec{UInt64}
        scat_parity::MtlVec{UInt64}

        space_list::MtlVec{UInt64}
        space::HilbertSubspace{bits}
    end
    mutable struct MtlAdjointLinearMap{bits, F <: AbstractFloat} <: AbstractGPULinearMap{bits, F}
        scat_amp::MtlVec{Complex{F}}
        scat_in::MtlVec{UInt64}
        scat_out::MtlVec{UInt64}
        scat_parity::MtlVec{UInt64}

        space_list::MtlVec{UInt64}
        space::HilbertSubspace{bits}

        function MtlAdjointLinearMap(A::MtlLinearMap{bits, F}) where {bits, F <: AbstractFloat}
            new{bits, F}(A.scat_amp, A.scat_in, A.scat_out, A.scat_parity, A.space_list, A.space)
        end
    end

    # ── Interface methods ──

    import Base: adjoint, size, eltype
    size(A::Union{MtlLinearMap, MtlAdjointLinearMap}) = (length(A.space), length(A.space))
    eltype(::Union{MtlLinearMap{bits,F}, MtlAdjointLinearMap{bits,F}}) where {bits,F} = Complex{F}

    function adjoint(A::MtlLinearMap{bits, F}) where {bits, F <: AbstractFloat}
        MtlAdjointLinearMap(A)
    end
    function adjoint(A::MtlAdjointLinearMap{bits, F}) where {bits, F <: AbstractFloat}
        MtlLinearMap{bits, F}(A.scat_amp, A.scat_in, A.scat_out, A.scat_parity, A.space_list, A.space)
    end

    #  Constructor — override the fallback in main module

    import MomentumED.Methods: create_gpu_linearmap
    function create_gpu_linearmap(A::LinearMap{bits, F}, ::Val{:metal}) where {bits, F}

        # Flatten scatter list into struct-of-arrays on GPU
        h_amp = Complex{F}[s.Amp for s in A.scat_list]
        h_in  = UInt64[s.in.n for s in A.scat_list]
        h_out = UInt64[s.out.n for s in A.scat_list]
        h_parity = UInt64[s.parity_mask for s in A.scat_list]

        # Basis as raw UInt64 sorted array
        h_basis = UInt64[mbs.n for mbs in A.space.list]

        return MtlLinearMap{bits, F}(
            MtlArray(h_amp),
            MtlArray(h_in),
            MtlArray(h_out),
            MtlArray(h_parity),
            MtlArray(h_basis),
            A.space
        )
    end

    #  Callable — uses shared KernelAbstractions kernels

    using MomentumED.Methods: gpu_matvec!, gpu_adjoint_matvec!

    function (A::MtlLinearMap{bits, F})(y::MtlVec{Complex{F}}, x::MtlVec{Complex{F}}) where {bits, F}
        gpu_matvec!(y, x, A.space_list, A.scat_amp, A.scat_in, A.scat_out, A.scat_parity)
        return y
    end

    function (A::MtlAdjointLinearMap{bits, F})(y::MtlVec{Complex{F}}, x::MtlVec{Complex{F}}) where {bits, F}
        gpu_adjoint_matvec!(y, x, A.space_list, A.scat_amp, A.scat_in, A.scat_out, A.scat_parity)
        return y
    end

    function (A::MtlLinearMap{bits, F})(x::MtlVec{Complex{F}}) where {bits, F}
        y = similar(x)
        A(y, x)
        return y
    end

    function (A::MtlAdjointLinearMap{bits, F})(x::MtlVec{Complex{F}}) where {bits, F}
        y = similar(x)
        A(y, x)
        return y
    end

    # ══════════════════════════════════════════════════════════════
    #  Metal sparse matrix method
    #  NOTE: Metal.jl does NOT currently provide sparse matrix support.
    #  create_gpu_matrix and krylov_matrix_solve are intentionally omitted.
    #  Use device=:metal only with method=:map (matrix-free).
    #  For method=:sparse on Metal, EDsolve should fall back to CPU.
    # ══════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════
    #  KrylovKit solver
    # ══════════════════════════════════════════════════════════════

    import MomentumED.Methods: krylov_map_solve
    function krylov_map_solve(
        H::Union{MtlLinearMap{bits, F}, MtlAdjointLinearMap{bits, F}},
        N_eigen::Int64;
        ishermitian::Bool=true,
        vec0::Union{Nothing, AbstractVector{Complex{F}}}=nothing,
        krylovkit_kwargs...) where {bits, F}

        m = length(H.space)
        if isnothing(vec0)
            vec0 = MtlArray(complex.(rand(F, m), rand(F, m)))
        elseif !(vec0 isa MtlVec)
            vec0 = MtlArray(vec0)
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

    import KrylovKit: basistransform!, OrthonormalBasis
    import LinearAlgebra: mul!

    # In-place basis transform using chunked matrix multiply
    function basistransform!(b::OrthonormalBasis{<:MtlArray{T, 1}}, U::AbstractMatrix) where {T}

        m, n = size(U)
        m == length(b) || throw(DimensionMismatch())

        N = length(b[1])
        chunk = min(N, GPU_RESTART_CHUNKSIZE[])

        buf_in  = MtlArray{T}(undef, chunk, m)
        buf_out = MtlArray{T}(undef, chunk, n)
        U_gpu   = MtlArray(T.(U))

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

        GPU_MEMORY_MONITOR[] && println("Metal memory: $(Metal.current_allocated_memory() / 1e9) GB allocated")

        buf_in  = nothing
        buf_out = nothing
        U_gpu   = nothing
        GC.gc()

        return b
    end

    # GC after Krylov restart
    import KrylovKit: shrink!, LanczosFactorization
    function shrink!(state::LanczosFactorization{<:MtlArray{T, 1}, S}, k;
        verbosity::Int = KrylovDefaults.verbosity[]) where {T <: Complex, S <: Real}

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

        GC.gc()

        return state
    end

end