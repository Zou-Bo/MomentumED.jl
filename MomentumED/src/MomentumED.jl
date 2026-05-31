"""
This module gives general methods for 2D momentum-block-diagonalized ED calculations.
Sectors of other quantum Numbers should be handled outside this module.
This module only sets sectors of total (crystal) momentum, also called blocks.

# export list
```julia
# submodules
export EDCore
public Preparation, Methods, Analysis

# main solving function
export EDsolve

# from EDCore
export MBS64, MBOperator
export HilbertSubspace, MBS64Vector, Scatter
public get_bits, get_body, index_fit
public isphysical, isupper, isdiagonal
export ED_bracket, ED_bracket_threaded
public ColexMBS64, ColexMBS64Mask

# preparation
export EDPara
export MBS_totalmomentum, ED_momentum_subspaces
export ED_scatterlist_onebody, ED_scatterlist_twobody

# methods
public SparseHmltMatrix, LinearMap
public GPU_AVAILABLE, DEFAULT_GPU, GPU_MEMORY_MONITOR
public activate_gpu_device, release_gpu_memory

# analysis - reduced density matrix for entanglement spectrum
export PES_1rdm, PES_MomtBlocks, PES_MomtBlock_rdm
export OES_NumMomtBlocks, OES_NumMomtBlock_spectrum

# analysis - many-body connection
export ED_connection_step, ED_connection_gaugefixing!
export ED_step_inner_prod

# environment variables
public PRINT_RECURSIVE_MOMENTUM_DIVISION
public PRINT_TWOBODY_SCATTER_PAIRS
```
"""
module MomentumED

    using EDCore
    using LinearAlgebra
    using SparseArrays
    using KrylovKit

    # submodules
    export EDCore
    public Preparation, Methods, Analysis

    # main solving function
    export EDsolve

    # from EDCore
    export MBS64, MBOperator
    export HilbertSubspace, MBS64Vector, Scatter
    public get_bits, get_body, index_fit
    public isphysical, isupper, isdiagonal
    export ED_bracket, ED_bracket_threaded
    public ColexMBS64, ColexMBS64Mask

    # preparation
    export EDPara
    export MBS_totalmomentum, ED_momentum_subspaces
    export ED_scatterlist_onebody, ED_scatterlist_twobody

    # methods
    public SparseHmltMatrix, LinearMap
    public GPU_AVAILABLE, DEFAULT_GPU, GPU_MEMORY_MONITOR
    public activate_gpu_device, release_gpu_memory

    # analysis - reduced density matrix for entanglement spectrum
    export PES_1rdm, PES_MomtBlocks, PES_MomtBlock_rdm
    export OES_NumMomtBlocks, OES_NumMomtBlock_spectrum

    # analysis - many-body connection
    export ED_connection_step, ED_connection_gaugefixing!
    export ED_step_inner_prod

    # environment variables
    public PRINT_RECURSIVE_MOMENTUM_DIVISION
    public PRINT_TWOBODY_SCATTER_PAIRS

    global PRINT_RECURSIVE_MOMENTUM_DIVISION::Bool = false
    global PRINT_TWOBODY_SCATTER_PAIRS::Bool = false

    """
    This module provides methods of defining a system for calculation.
    Many functions assume the system has at most two-body symmetric interaction,
    Hermitian Hamiltonian, and momentum conservation.
    """
    module Preparation
        export EDPara
        export MBS_totalmomentum, ED_momentum_subspaces
        export ED_scatterlist_onebody, ED_scatterlist_twobody

        import ..MomentumED.PRINT_RECURSIVE_MOMENTUM_DIVISION
        import ..MomentumED.PRINT_TWOBODY_SCATTER_PAIRS

        using EDCore

        include("preparation/init_parameter.jl")
        include("preparation/momentum_decomposition.jl")
        include("preparation/scat_list.jl")
    end

    """
    This module provides methods of generating Hamiltonian sparse matrix or Hamiltonian linear map,
    and use KrylovKit to solve them.
    """
    module Methods
        export SparseHmltMatrix, LinearMap
        export krylov_map_solve, krylov_matrix_solve

        # CUDA-specific flags, parameters, and functions
        # export release_cuda
        # export CUDA_AVAILABLE
        # export CUDA_KRYLOV_INPLACE_RESTART_CHUNKSIZE
        # export CUDA_MEMORY_MONITOR

        export GPU_AVAILABLE, DEFAULT_GPU, GPU_MEMORY_MONITOR
        export activate_gpu_device, release_gpu_memory

        using EDCore
        using ..Preparation
        using SparseArrays
        using LinearAlgebra
        using KrylovKit
        using KernelAbstractions
        
        include("method/sparse_matrix.jl")
        include("method/linear_map.jl")
        include("method/gpu_interface.jl")
    end

    """
    This module provides methods to analysis the ED eigenwavefunctions,
    including generating particle(hole)/orbital reduced density matrix for entanglement spectrum analysis, 
    and many-body connection analysis with momentum shifts
    """
    module Analysis
        export PES_1rdm, PES_MomtBlocks, PES_MomtBlock_rdm
        export OES_NumMomtBlocks, OES_NumMomtBlock_coef, OES_NumMomtBlock_spectrum
        export ED_connection_step, ED_connection_gaugefixing!
        export ED_step_inner_prod

        using EDCore
        using ..Preparation

        include("analysis/particle_reduced_density_matrix.jl")
        include("analysis/orbital_reduced_density_matrix.jl")
        include("analysis/manybody_quantum_geometry.jl")
    end

    using .Preparation
    using .Methods
    using .Analysis

    """
        EDsolve(subspace::HilbertSubspace, hamiltonian; kwargs...) 
            -> energies::Vector, vectors::Vector{MBS64Vector}

    Main exact diagonalization solver for momentum-conserved quantum systems.

    This function finds the lowest eigenvalues and eigenvectors of a Hamiltonian within a given momentum subspace. It supports multiple methods for diagonalization and can accept the Hamiltonian in two formats.

    # Arguments
    - `subspace::HilbertSubspace`: The Hilbert subspace for a specific momentum block, containing the basis states.
    - `hamiltonian`: The Hamiltonian to be diagonalized. It can be provided in two forms:
        1. As a series of sorted `Vector{<:Scatter}` arguments (e.g., `EDsolve(subspace, scat1, scat2)`). This form is used for matrix-based methods.
        2. As a single `MBOperator` object (e.g., `EDsolve(subspace, H_operator)`). This form is required for the matrix-free `:map` method.

    # Keyword Arguments
    - `N::Int64 = 6`: The number of eigenvalues/eigenvectors to compute.
    - `method::Symbol = :sparse`: The diagonalization method. Options are:
        - `:sparse`: (Default) Constructs the Hamiltonian as a sparse matrix. Good for most cases.
        - `:dense`: Constructs a dense matrix. Can be faster for very small systems.
        - `:map`: Uses a CPU matrix-free `LinearMap` approach. This is the most memory-efficient method for very large systems and requires the `hamiltonian` to be an `MBOperator`.
        - `:cuda_map` / `:gpu_map`: Uses a CUDA-backed matrix-free `CuLinearMap` on the active NVIDIA GPU. This also requires the `hamiltonian` to be an `MBOperator`.
    - `element_type::Type = Float64`: The element type for the Hamiltonian matrix (for `:sparse`/:`dense`).
    - `index_type::Type = Int64`: The integer type for the sparse matrix indices (for `:sparse`).
    - `min_sparse_dim::Int64 = 100`: If `method` is `:sparse` but the dimension is smaller than this, it will automatically switch to `:dense`.
    - `max_dense_dim::Int64 = 200`: If `method` is `:dense` but the dimension is larger than this, it will automatically switch to `:sparse`.
    - `ishermitian::Bool = true`: Specifies if the Hamiltonian is Hermitian. This is passed to the eigensolver for optimization.
    - `showtime::Bool = false`: If `true`, prints the time taken for matrix construction and diagonalization.
    - `krylovkit_kwargs...`: Additional keyword arguments passed directly to `KrylovKit.eigsolve`.

    # Returns
    - `energies::Vector`: A vector containing the `N` lowest eigenvalues.
    - `vectors::Vector{MBS64Vector}`: A vector of the corresponding eigenvectors, wrapped in the `MBS64Vector` type.

    # Examples

    **1. Using Scatter Lists (Sparse Matrix Method):**
    ```julia
    subspaces, _, _ = ED_momentum_subspaces(para, (1,1))
    scat1 = ED_scatterlist_onebody(para)
    scat2 = ED_scatterlist_twobody(para)

    # Find the 2 lowest energy states
    energies, vecs = EDsolve(subspaces[1], scat1, scat2; N=2, method=:sparse)
    ```

    **2. Using MBOperator (Sparse Matrix or Linear Map Method):**
    ```julia
    H_op = MBOperator(scat1, scat2)
    # Find the 2 lowest energy states using the matrix-free approach
    energies, vecs = EDsolve(subspaces[1], H_op; N=2, method=:map)
    ```
    """
    function EDsolve(subspace::HilbertSubspace{bits}, scat_lists::Vector{<: Scatter}...;
        N::Int64, method::Symbol = :sparse, device::Symbol = :cpu,
        showtime::Bool = false, ishermitian::Bool = true,
        element_type::Type = Float64, index_type::Type = Int32, 
        min_sparse_dim::Int64 = 100, max_dense_dim::Int64 = 500, map_warning_dim::Int64 = 20000, method_info::Bool = true, 
        krylovkit_kwargs... ) where {bits}

        @assert N >= 1 "N should be at least 1."
        @assert length(subspace) <= typemax(index_type) "Hilbert space dimension $(length(subspace)) exceeds limit of index_type=$index_type."
        @assert method ∈ (:sparse, :dense, :map) "Unknown method: $method. Use :sparse, :dense, :map."
        @assert device ∈ (:cpu, :gpu, :cuda, :oneapi, :metal, :multi_cuda) "Unknown device: $device. Use :cpu, :gpu, :cuda, :oneapi, :metal, :multi_cuda."

        if method == :map
            error("Linear map methods are only supported when input Hamitonian is MBOperator instead of Vector{Scatter}.")
        end

        if min_sparse_dim > max_dense_dim
            method_info && @info "Trying to set min_sparse_dim as $min_sparse_dim, larger than max_dense_dim(=$max_dense_dim). Reset to $max_dense_dim automatically."
            min_sparse_dim = max_dense_dim
        end
        if method == :sparse && length(subspace) < min_sparse_dim
            method_info && @info "Hilbert space dimension < $min_sparse_dim; switch to method=:dense automatically."
            method = :dense
        end
        if method == :dense && length(subspace) > max_dense_dim
            method_info && @info "Hilbert space dimension > $max_dense_dim; switch to method=:sparse automatically."
            method = :sparse
        end

        @assert ishermitian "Current Hamiltonian matrix construction assumes it being Hermitian."

        # Construct sparse Hamiltonian matrix from Scatter terms
        if showtime
            @time H = SparseHmltMatrix(subspace, vcat(scat_lists...);
                element_type = element_type, index_type = index_type
            )
        else
            H = SparseHmltMatrix(subspace, vcat(scat_lists...);
                element_type = element_type, index_type = index_type
            )
        end

        if method == :sparse

            dim = size(H, 1)
            N > dim && (N = dim)

            if device == :cpu

                # Solve the eigenvalue problem
                if showtime
                    @time vals, vecs, _ = krylov_matrix_solve(H, N; ishermitian, krylovkit_kwargs...)
                else
                    vals, vecs, _ = krylov_matrix_solve(H, N; ishermitian, krylovkit_kwargs...)
                end

                if length(vals) < N
                    error("Krylov method fails. Cannot find $N eigenvectors.")
                end
                energies = vals[1:N]
                vectors = [MBS64Vector(vecs[i], subspace) for i in 1:N]

                return energies, vectors

            # elseif device == :cuda # keep the old method only for :cuda before new methods are tested working

            #     Methods._throw_cuda_unavailable()
            #     H_gpu = Methods.create_gpu_matrix(H)

            #     # Solve the eigenvalue problem with GPU-accelerated sparse matrix
            #     if showtime
            #         @time vals, vecs_gpu, _ = krylov_matrix_solve(H_gpu, N; ishermitian, krylovkit_kwargs...)
            #     else
            #         vals, vecs_gpu, _ = krylov_matrix_solve(H_gpu, N; ishermitian, krylovkit_kwargs...)
            #     end

            #     if length(vals) < N
            #         error("Krylov method fails. Cannot find $N eigenvectors.")
            #     end
            #     energies = vals[1:N]
            #     vectors = [MBS64Vector(Array(vecs_gpu[i]), subspace) for i in 1:N]

            #     # free GPU memory
            #     H_gpu = nothing; vecs_gpu = nothing
            #     release_cuda(2)

            #     return energies, vectors

            elseif device ∈ (:cuda, :oneapi, :metal, :multi_cuda, :gpu)

                if device == :gpu
                    device = DEFAULT_GPU[]
                    if device == :nogpu
                        throw(ArgumentError(
                            "No GPU available. Please load a supported GPU extension and verify availability."
                        ))
                    end
                end

                Methods._throw_gpu_unavailable(device)
                H_gpu = Methods.create_gpu_matrix(H, Val(device))

                # Solve the eigenvalue problem with GPU-accelerated sparse matrix
                if showtime
                    @time vals, vecs_gpu, _ = krylov_matrix_solve(H_gpu, N; ishermitian, krylovkit_kwargs...)
                else
                    vals, vecs_gpu, _ = krylov_matrix_solve(H_gpu, N; ishermitian, krylovkit_kwargs...)
                end

                if length(vals) < N
                    error("Krylov method fails. Cannot find $N eigenvectors.")
                end
                energies = vals[1:N]
                vectors = [MBS64Vector(Array(vecs_gpu[i]), subspace) for i in 1:N]

                # free GPU memory
                H_gpu = nothing; vecs_gpu = nothing
                release_gpu_memory(Val(device), 2)

                return energies, vectors

            end

        elseif method == :dense

            dim = size(H, 1)
            if dim > 1000
                method_info && @warn "Dense diagonalization may be slow for dim=$dim. Consider using :sparse method."
            end
            N > dim && (N = dim)

            if device != :cpu
                @warn "Dense method does not support GPU acceleration. Ignoring device=$device and using CPU."
            end

            # Convert H to a dense matrix and solve
            if ishermitian
                if showtime
                    @time vals, vecs = eigen(Hermitian(Matrix(H)), 1:N)
                else
                    vals, vecs = eigen(Hermitian(Matrix(H)), 1:N)
                end
            else
                if showtime
                    @time vals, vecs = eigen(Matrix(H))
                else
                    vals, vecs = eigen(Matrix(H))
                end
            end

            energies = vals[1:N]
            vectors = [MBS64Vector(vecs[:, i], subspace) for i in 1:N] # Convert to vector of vectors

            return energies, vectors

        end

    end
    function EDsolve(subspace::HilbertSubspace{bits}, Hamiltonian::MBOperator;
        N::Int64, method::Symbol = :sparse, device::Symbol = :cpu,
        showtime::Bool = false, ishermitian::Bool = true,
        element_type::Type = Float64, index_type::Type = Int32,
        min_sparse_dim::Int64 = 100, max_dense_dim::Int64 = 200, map_warning_dim::Int64 = 20000, method_info::Bool = true, 
        krylovkit_kwargs... ) where{bits}

        @assert N >= 1
        @assert length(subspace) <= typemax(index_type) "Hilbert space dimension $(length(subspace)) exceeds limit of index_type=$index_type."
        @assert method ∈ (:sparse, :dense, :map) "Unknown method: $method. Use :sparse, :dense, :map."
        @assert device ∈ (:cpu, :gpu, :cuda, :oneapi, :metal, :multi_cuda) "Unknown device: $device. Use :cpu, :gpu, :cuda, :oneapi, :metal, :multi_cuda."


        if ishermitian
            @assert isupper(Hamiltonian) "Use upper_hermitian form of Hamiltonian operator when ishermitian = true."
        end

        if method == :sparse || method == :dense
            return EDsolve(subspace, Hamiltonian.scats; N, method, device, showtime, ishermitian,
                min_sparse_dim, max_dense_dim, map_warning_dim, method_info,
                element_type, index_type, krylovkit_kwargs...
            )
        end

        dim = length(subspace)
        if dim < map_warning_dim && method_info
            @warn "Linear map may be slow for dim=$dim. Consider using :sparse method."
        end

        if device == :cpu

            H_map = LinearMap(Hamiltonian, subspace)

            # Solve the eigenvalue problem
            if showtime
                @time vals, vecs, _ = krylov_map_solve(H_map, N; ishermitian, krylovkit_kwargs...)
            else
                vals, vecs, _ = krylov_map_solve(H_map, N; ishermitian, krylovkit_kwargs...)
            end

            length(vals) < N && error("Krylov method fails. Cannot find $N eigenvectors.")
            energies = vals[1:N]
            vectors = [MBS64Vector(vecs[i], subspace) for i in 1:N]

            return energies, vectors

        # elseif device == :cuda # keep the old method only for :cuda before new methods are tested working
            
        #     Methods._throw_cuda_unavailable()
        #     H_map = LinearMap(Hamiltonian, subspace)
        #     H_gpu = Methods.create_CuLinearMap(H_map)

        #     # Solve the eigenvalue problem with GPU-accelerated linear map
        #     if showtime
        #         @time vals, vecs_gpu, _ = krylov_map_solve(H_gpu, N; ishermitian, krylovkit_kwargs...)
        #     else
        #         vals, vecs_gpu, _ = krylov_map_solve(H_gpu, N; ishermitian, krylovkit_kwargs...)
        #     end

        #     length(vals) < N && error("Krylov method fails. Cannot find $N eigenvectors.")
        #     energies = vals[1:N]
        #     vectors = [MBS64Vector(Array(vecs_gpu[i]), subspace) for i in 1:N]

        #     # free GPU memory
        #     H_gpu = nothing; vecs_gpu = nothing
        #     release_cuda(2)
            
        #     return energies, vectors

        elseif device ∈ (:cuda, :oneapi, :metal, :multi_cuda, :gpu)
            
            if device == :gpu
                device = DEFAULT_GPU[]
                if device == :nogpu
                    throw(ArgumentError(
                        "No GPU available. Please load a supported GPU extension and verify availability."
                    ))
                end
            end

            Methods._throw_gpu_unavailable(device)
            H_map = LinearMap(Hamiltonian, subspace)
            H_gpu = Methods.create_gpu_linearmap(H_map, Val(device))

            # Solve the eigenvalue problem with GPU-accelerated sparse matrix
            if showtime
                @time vals, vecs_gpu, _ = krylov_map_solve(H_gpu, N; ishermitian, krylovkit_kwargs...)
            else
                vals, vecs_gpu, _ = krylov_map_solve(H_gpu, N; ishermitian, krylovkit_kwargs...)
            end

            if length(vals) < N
                error("Krylov method fails. Cannot find $N eigenvectors.")
            end
            energies = vals[1:N]
            vectors = [MBS64Vector(Array(vecs_gpu[i]), subspace) for i in 1:N]

            # free GPU memory
            H_gpu = nothing; vecs_gpu = nothing
            release_gpu_memory(Val(device), 2)

            return energies, vectors

        end

    end

end