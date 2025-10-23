"""
This module gives general methods for 2D momentum-block-diagonalized ED calculations.
Sectors of other quantum Numbers should be handled outside this module.
This module only sets sectors of total (crystal) momentum, also called blocks.
"""
module MomentumED

using MomentumEDCore
using LinearAlgebra
using SparseArrays
using KrylovKit

# types from MomentumEDCore
export MBS64, HilbertSubspace, MBS64Vector, Scatter, MBOperator
public get_bits, get_body, make_dict!, delete_dict!
public isphysical, isupper, isnormal, isnormalupper
export ED_bracket, ED_bracket_threaded # , multiplication_threaded

# Include utilities
include("preparation/init_parameter.jl")
include("preparation/momentum_decomposition.jl")
include("preparation/scat_list.jl")
include("method/sparse_matrix.jl")
include("method/linear_map.jl")
include("analysis/onebody_reduced_density_matrix.jl")
include("analysis/orbital_reduced_density_matrix.jl")
# include("analysis/entanglement_entropy.jl")
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

# analysis - reduced density matrix
export RDM_OneBody
export RDM_NumberBlocks, RDM_MomentumCoefficients

# analysis - many-body connection
export ED_connection_step, ED_connection_gaugefixing!

# environment variables
public PRINT_RECURSIVE_MOMENTUM_DIVISION
public PRINT_TWOBODY_SCATTER_PAIRS



"""
    EDsolve(
        subspace::HilbertSubspace, 
        sorted_scat_lists::Vector{<: Scatter}...;
        N::Int64 = 6, showtime = false, method = :sparse,
        element_type::Type = Float64, index_type::Type = Int64, 
        min_sparse_dim::Int64 = 100, max_dense_dim::Int64 = 200,
        krylovkit_kwargs...) -> (vals, vecs)

Main interface function for exact diagonalization of momentum-conserved quantum systems.
Constructs the sparse Hamiltonian matrix from Scatter lists and diagonalizes it.

# Arguments
- `subspace::HilbertSubspace`: Sorted list of many-body states in the momentum block
- `sorted_scat_list::Vector{<: Scatter}`: Sorted Scatter terms (one-body, two-body, etc.)
- `N_eigen::Int64=6`: Number of eigenvalues/eigenvectors to compute (default: 6)

# Keywords
- `showtime::Bool=false`: Whether to print timing information for matrix construction and diagonalization
- `krylovkit_kwargs...`: Additional keyword arguments to pass to KrylovKit.eigsolve


# Returns
- `vals::Vector{Float64}`: Eigenvalues (energies) in ascending order
- `vecs::Vector{Vector{ComplexF64}}`: Corresponding eigenvectors

# Examples
```julia
# Create basis and Scatter lists for a 2-site system
basis = ED_mbslist(para, (2,))
blocks, _, _, _ = ED_momentum_block_division(para, basis)
Scatter1 = ED_sortedScatterList_onebody(para)
Scatter2 = ED_sortedScatterList_twobody(para)

# Solve for ground state and first excited state
energies, wavefunctions = EDsolve(blocks[1], Scatter1, Scatter2; N=1)
println("Ground state energy: ", energies[1])
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
            @warn "Hilbert space dimension < $min_sparse_dim; switch to method=:dense automatically."
            method = :dense
        end
        if method == :dense && length(subspace) > max_dense_dim
            @warn "Hilbert space dimension > $max_dense_dim; switch to method=:sparse automatically."
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
