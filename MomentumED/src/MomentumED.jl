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
export MBS64, NormalScatter, MBOperator
export HilbertSubspace, MBS64Vector, Scatter
public get_bits, get_body, make_dict!, delete_dict!
public isphysical, isupper, isnormal, isnormalupper, isdiagonal
export ED_bracket, ED_bracket_threaded
public ColexMBS64, ColexMBS64Mask

# preparation
export EDPara, ED_momentum_subspaces
export ED_sortedScatterList_onebody
export ED_sortedScatterList_twobody

# methods
public ED_HamiltonianMatrix_threaded, LinearMap

# analysis - reduced density matrix for entanglement spectrum
export PES_1rdm, PES_MomtBlocks, PES_MomtBlock_rdm
export OES_NumMomtBlocks, OES_NumMomtBlock_coef

# analysis - many-body connection
export ED_connection_step, ED_connection_gaugefixing!

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
export MBS64, NormalScatter, MBOperator
export HilbertSubspace, MBS64Vector, Scatter
public get_bits, get_body, make_dict!, delete_dict!
public isphysical, isupper, isnormal, isnormalupper, isdiagonal
export ED_bracket, ED_bracket_threaded
public ColexMBS64, ColexMBS64Mask

# preparation
export EDPara, ED_momentum_subspaces
export ED_sortedScatterList_onebody
export ED_sortedScatterList_twobody

# methods
public ED_HamiltonianMatrix_threaded, LinearMap

# analysis - reduced density matrix for entanglement spectrum
export PES_1rdm, PES_MomtBlocks, PES_MomtBlock_rdm
export OES_NumMomtBlocks, OES_NumMomtBlock_coef

# analysis - many-body connection
export ED_connection_step, ED_connection_gaugefixing!

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
    export EDPara, ED_momentum_subspaces
    export ED_sortedScatterList_onebody
    export ED_sortedScatterList_twobody

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
    export ED_HamiltonianMatrix_threaded, LinearMap
    export krylov_map_solve, krylov_matrix_solve

    using EDCore
    using ..Preparation
    using SparseArrays
    using LinearAlgebra
    using KrylovKit
    
    include("method/sparse_matrix.jl")
    include("method/linear_map.jl")
end

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
    - `:map`: Uses a matrix-free `LinearMap` approach. This is the most memory-efficient method for very large systems and requires the `hamiltonian` to be an `MBOperator`.
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
scat1 = ED_sortedScatterList_onebody(para)
scat2 = ED_sortedScatterList_twobody(para)

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
function EDsolve(subspace::HilbertSubspace{bits}, sorted_scat_lists::Vector{<: Scatter}...;
    N::Int64 = 6, showtime::Bool = false, method::Symbol = :sparse,
    element_type::Type = Float64, index_type::Type = Int64, 
    min_sparse_dim::Int64 = 100, max_dense_dim::Int64 = 200,
    ishermitian::Bool = true, krylovkit_kwargs...
    ) where {bits}

    @assert N >= 1

    if method == :map

        error("Linear map method is only supported when input Hamitonian is MBOperator instead of Vector{Scatter}.")

    elseif method == :sparse || method == :dense

        @assert max_dense_dim > min_sparse_dim
        if method == :sparse && length(subspace) < min_sparse_dim
            @info "Hilbert space dimension < $min_sparse_dim; switch to method=:dense automatically."
            method = :dense
        end
        if method == :dense && length(subspace) > max_dense_dim
            @info "Hilbert space dimension > $max_dense_dim; switch to method=:sparse automatically."
            method = :sparse
        end

        @assert ishermitian "Current Hamiltonian matrix construction assumes it being Hermitian."

        # Construct sparse Hamiltonian matrix from Scatter terms
        if showtime
            @time H = ED_HamiltonianMatrix_threaded(subspace, sorted_scat_lists...;
                element_type = element_type, index_type = index_type
            )
        else
            H = ED_HamiltonianMatrix_threaded(subspace, sorted_scat_lists...;
                element_type = element_type, index_type = index_type
            )
        end

        if method == :sparse

            @assert N <= length(subspace)

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

        elseif method == :dense

            dim = size(H, 1)
            if dim > 1000
                @warn "Dense diagonalization may be slow for dim=$dim. Consider using :sparse method."
            end
            N > dim && (N = dim)

            # Convert to dense matrix and solve
            if ishermitian
                if showtime
                    @time vals, vecs = eigen(Hermitian(Matrix(H)))
                else
                    vals, vecs = eigen(Hermitian(Matrix(H)))
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

        end

    else
        error("Unknown method: $method. Use :sparse, :dense, or :map.")
    end

    return energies, vectors
end
function EDsolve(subspace::HilbertSubspace{bits}, Hamiltonian::MBOperator;
    N::Int64 = 6, showtime::Bool = false, method::Symbol = :sparse,
    element_type::Type = Float64, index_type::Type = Int64, 
    min_sparse_dim::Int64 = 100, max_dense_dim::Int64 = 200,
    ishermitian::Bool = true, krylovkit_kwargs...
    ) where{bits}

    if ishermitian
        @assert isupper(Hamiltonian) "Use upper_hermitian form of Hamiltonian operator when ishermitian = true."
    end

    if method == :map

        dim = length(subspace)
        if dim < 20000
            @warn "Linear map may be slow for dim=$dim. Consider using :sparse method."
        end

        H_map = LinearMap(Hamiltonian, subspace, element_type)

        # Solve the eigenvalue problem
        if showtime
            @time vals, vecs, _ = krylov_map_solve(H_map, N; ishermitian, krylovkit_kwargs...)
        else
            vals, vecs, _ = krylov_map_solve(H_map, N; ishermitian, krylovkit_kwargs...)
        end

        energies = vals[1:N]
        vectors = [MBS64Vector(vecs[i], subspace) for i in 1:N]

        return energies, vectors

    else
        return EDsolve(subspace, Hamiltonian.scats...; N, showtime, method, 
            element_type, index_type, min_sparse_dim, max_dense_dim, 
            ishermitian, krylovkit_kwargs...
        )
    end

end

"""
This module provides methods to analysis the ED eigenwavefunctions,
including generating particle(hole)/orbital reduced density matrix for entanglement spectrum analysis, 
and many-body connection analysis with momentum shifts
"""
module Analysis
    export PES_1rdm, PES_MomtBlocks, PES_MomtBlock_rdm
    export OES_NumMomtBlocks, OES_NumMomtBlock_coef
    export ED_connection_step, ED_connection_gaugefixing!

    using EDCore
    using ..Preparation

    include("analysis/particle_reduced_density_matrix.jl")
    include("analysis/orbital_reduced_density_matrix.jl")
    include("analysis/manybody_connection.jl")
end

using .Preparation
using .Methods
using .Analysis

end