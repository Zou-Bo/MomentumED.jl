"""
This module gives general methods for 2D momentum-block-diagonalized ED calculations.
Sectors of other quantum Numbers should be handled outside this module.
This module only sets sectors of total (crystal) momentum, also called blocks.
"""
module MomentumED

using EDCore
using LinearAlgebra
using SparseArrays
using KrylovKit

# types from MomentumEDCore
export MBS64, HilbertSubspace, MBS64Vector, Scatter, MBOperator
public get_bits, get_body, make_dict!, delete_dict!
public isphysical, isupper, isnormal, isnormalupper, isdiagonal
export ED_bracket, ED_bracket_threaded

# Include utilities
include("preparation/init_parameter.jl")
include("preparation/momentum_decomposition.jl")
include("preparation/scat_list.jl")
include("method/sparse_matrix.jl")
include("method/linear_map.jl")
include("analysis/particle_reduced_density_matrix.jl")
include("analysis/orbital_reduced_density_matrix.jl")
include("analysis/manybody_connection.jl")

# preparation
public mbslist_onecomponent
export EDPara, ED_momentum_subspaces
export ED_sortedScatterList_onebody
export ED_sortedScatterList_twobody

# methods
public ED_HamiltonianMatrix_threaded, LinearMap

# main solving function
export EDsolve

# analysis - reduced density matrix for entanglement spectrum
export PES_1rdm, PES_MomtBlocks, PES_MomtBlock_rdm
export OES_NumMomtBlocks, OES_NumMomtBlock_coef

# analysis - many-body connection
export ED_connection_step, ED_connection_gaugefixing!

# environment variables
public PRINT_RECURSIVE_MOMENTUM_DIVISION
public PRINT_TWOBODY_SCATTER_PAIRS



"""
    EDsolve(subspace::HilbertSubspace, hamiltonian; 
        kwargs...) -> (energies::Vector, vectors::Vector{MBS64Vector})

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
% - `element_type::Type = Float64`: The element type for the Hamiltonian matrix (for `:sparse`/:`dense`).
% - `index_type::Type = Int64`: The integer type for the sparse matrix indices (for `:sparse`).
- `min_sparse_dim::Int64 = 100`: If `method` is `:sparse` but the dimension is smaller than this, it will automatically switch to `:dense`.
- `max_dense_dim::Int64 = 200`: If `method` is `:dense` but the dimension is larger than this, it will automatically switch to `:sparse`.
- `ishermitian::Bool = true`: Specifies if the Hamiltonian is Hermitian. This is passed to the eigensolver for optimization.
- `showtime::Bool = false`: If `true`, prints the time taken for matrix construction and diagonalization.
- `krylovkit_kwargs...`: Additional keyword arguments passed directly to `KrylovKit.eigsolve`.

# Returns
- `energies::Vector`: A vector containing the `N` lowest eigenvalues.
- `vectors::Vector{MBS64Vector}`: A vector of the corresponding eigenvectors, wrapped in the `MBS64Vector` type.

# Examples

**1. Using Scatter Lists (Sparse Method):**
```julia
subspaces, _, _ = ED_momentum_subspaces(para, (1,1))
scat1 = ED_sortedScatterList_onebody(para)
scat2 = ED_sortedScatterList_twobody(para)

# Find the 2 lowest energy states
energies, vecs = EDsolve(subspaces[1], scat1, scat2; N=2, method=:sparse)
```

**2. Using MBOperator (Linear Map Method):**
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


    if method == :map

        error("Linear map method is only used when input Hamitonian is MBOperator instead of Vector{Scatter}.")

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

            # Solve the eigenvalue problem
            if showtime
                @time vals, vecs, _ = krylov_matrix_solve(H, N; ishermitian, krylovkit_kwargs...)
            else
                vals, vecs, _ = krylov_matrix_solve(H, N; ishermitian, krylovkit_kwargs...)
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



end

#=
"""
Calculate the reduced density matrix for a subsystem
"""
function reduced_density_matrix(psi::Vector{ComplexF64}, 
                               block_MBSList::Vector{MBS{bits}},
                               nA::Vector{Int64},
                               iA::Vector{Int64};
                               cutoff::Float64=1E-7) where {bits}
    
    bits_total = bits
    sorted_state_num_list = getfield.(block_MBSList, :n)
    
    # Filter by cutoff
    index_list = findall(x -> abs2(x) > cutoff, psi)
    psi_filtered = psi[index_list]
    num_list = sorted_state_num_list[index_list]
    
    Amask = sum(1 << i for i in iA; init=0)
    Bmask = (1 << bits_total) - 1 - Amask
    
    # Sort by B subsystem
    myless_fine(n1, n2) = n1 & Bmask < n2 & Bmask || n1 & Bmask == n2 & Bmask && n1 < n2
    myless_coarse(n1, n2) = n1 & Bmask < n2 & Bmask
    
    perm = sortperm(num_list; lt=myless_fine)
    psi_sorted = psi_filtered[perm]
    num_sorted = num_list[perm]
    
    # Find B chunks
    Bchunks_lastindices = Int64[0]
    let i=0
        while i < length(num_sorted)
            i = searchsortedlast(num_sorted, num_sorted[i+1]; lt=myless_coarse)
            push!(Bchunks_lastindices, i)
        end
    end
    
    NA = length(nA)
    RDM_threads = zeros(ComplexF64, NA, NA, Threads.nthreads())
    
    Threads.@threads for nchunk in 1:length(Bchunks_lastindices)-1
        id = Threads.threadid()
        chunkpiece = Bchunks_lastindices[nchunk]+1:Bchunks_lastindices[nchunk+1]
        numB = num_sorted[chunkpiece[1]] & Bmask
        
        for i in 1:NA
            numA = nA[i]
            num = numB + numA
            index = my_searchsortedfirst(num_sorted[chunkpiece], num)
            index == 0 && continue
            
            RDM_threads[i, i, id] += abs2(psi_sorted[chunkpiece[index]])
            
            for i′ in i+1:NA
                numA′ = nA[i′]
                num′ = numB + numA′
                index′ = my_searchsortedfirst(num_sorted[chunkpiece], num′)
                index′ == 0 && continue
                
                rhoii′ = conj(psi_sorted[chunkpiece[index′]]) * psi_sorted[chunkpiece[index]]
                RDM_threads[i, i′, id] += rhoii′
                RDM_threads[i′, i, id] += conj(rhoii′)
            end
        end
    end
    
    return sum(RDM_threads; dims=3)[:, :, 1]
end

"""
Calculate entanglement entropy from reduced density matrix
"""
function entanglement_entropy(RDM_A::Matrix{ComplexF64}; cutoff::Float64=1E-6)
    vals = eigvals(Hermitian(RDM_A))
    return sum(vals) do x
        if abs(x) < cutoff || x < 0
            return 0.0
        end
        -x * log2(x)
    end
end

"""
Calculate one-body reduced density matrix
"""
function one_body_reduced_density_matrix(psi::Vector{ComplexF64},
                                        block_MBSList::Vector{MBS{bits}};
                                        cutoff::Float64=1E-7) where {bits}
    
    bits_total = bits
    sorted_state_num_list = getfield.(block_MBSList, :n)
    
    # Filter by cutoff
    index_list = findall(x -> abs2(x) > cutoff, psi)
    psi_filtered = psi[index_list]
    num_list = sorted_state_num_list[index_list]
    
    # One-body RDM: ρ[i,j] = <c†_j c_i>
    rdm = zeros(ComplexF64, bits_total, bits_total)
    
    Threads.@threads for i in 0:bits_total-1
        for j in 0:bits_total-1
            if i == j
                # Diagonal: <n_i>
                for (idx, num) in enumerate(num_list)
                    if isodd(num >>> i)
                        rdm[i+1, j+1] += abs2(psi_filtered[idx])
                    end
                end
            else
                # Off-diagonal: <c†_j c_i>
                for (idx, num) in enumerate(num_list)
                    if isodd(num >>> i) && iseven(num >>> j)
                        new_num = num - (1 << i) + (1 << j)
                        new_idx = my_searchsortedfirst(num_list, new_num)
                        new_idx == 0 && continue
                        
                        sign_flip = sum(k -> (num >>> k) % 2, min(i,j)+1:max(i,j)-1; init=0)
                        sign = isodd(sign_flip) ? -1 : 1
                        
                        rho_ij = sign * conj(psi_filtered[new_idx]) * psi_filtered[idx]
                        rdm[i+1, j+1] += rho_ij
                        rdm[j+1, i+1] += conj(rho_ij)
                    end
                end
            end
        end
    end
    
    return rdm
end



=#
