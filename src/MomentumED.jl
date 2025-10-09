"""
This module gives general methods for 2D momentum-block-diagonalized ED calculations.
Sectors of other quantum Numbers should be handled outside this module.
This module only sets sectors of total (crystal) momentum, also called blocks.
"""
module MomentumED

# type
export MBS64, MBS64Vector, Scattering, MBSOperator
public get_bits, create_state_mapping
export ED_bracket, ED_bracket_threaded, multiplication_threaded

# preparation
public ED_mbslist_onecomponent
export EDPara, ED_mbslist, ED_momentum_block_division
export ED_sortedScatteringList_onebody
export ED_sortedScatteringList_twobody

# methods
public ED_HamiltonianMatrix_threaded

# main solving function
export EDsolve

# analysis
export ED_onebody_rdm
# export ED_entanglement_entropy
export ED_connection_step, ED_connection_gaugefixing!

using LinearAlgebra
using SparseArrays
using KrylovKit


# Include utilities
include("type/manybodystate.jl")
include("type/scattering.jl")
include("type/operator_on_state.jl")
include("preparation/init_parameter.jl")
include("preparation/momentum_decomposition.jl")
include("preparation/scat_list.jl")
include("method/sparse_matrix.jl")
# include("method/linear_map.jl") # in development





"""
    matrix_solve(H::SparseMatrixCSC{ComplexF64, Int64}, N_eigen::Int64=6; 
        converge_warning::Bool=false, krylovkit_kwargs...) -> (vals, vecs)

Solve the sparse Hamiltonian matrix using KrylovKit's eigsolve function for the lowest n eigenvalues and eigenvectors.

# Arguments
- `H::SparseMatrixCSC{Complex{eltype}, idtype}`: Sparse Hamiltonian matrix to diagonalize
- `N_eigen::Int64=6`: Number of eigenvalues/eigenvectors to compute (default: 6)

# Keywords
- `vec0::Vector{Complex{eltype}}=rand(Complex{eltype}, H.m)`: Initial guess vector for Krylov iteration
- `ishermitian::Bool=true`: Whether the matrix is Hermitian (default: true)
- `krylovkit_kwargs...`: Additional keyword arguments to pass to KrylovKit.eigsolve

# Returns
- `vals::Vector{eltype}`: Eigenvalues (energies) in ascending order
- `vecs::Vector{Vector{Complex{eltype}}}`: Corresponding eigenvectors
- `info`: Convergence information from KrylovKit

# Examples
```julia
# Solve for 3 lowest eigenstates
vals, vecs, info = matrix_solve(H_matrix, 3)
println("Ground state energy: ", vals[1])
```

# Notes
- Uses KrylovKit's eigsolve with :SR (smallest real) eigenvalue selection
- Assumes Hermitian matrix (standard for quantum Hamiltonians)
- Random initial vector ensures good convergence properties
- Automatically handles convergence warnings from KrylovKit
- For better control over convergence, consider using KrylovKit directly
"""
function krylov_matrix_solve(
    H::SparseMatrixCSC{Complex{eltype}}, N_eigen::Int64;
    ishermitian = true, krylovkit_kwargs...
)::Tuple{Vector{eltype}, Vector{Vector{Complex{eltype}}}, Any} where {eltype<:AbstractFloat}

    vec0 = rand(Complex{eltype}, H.m)

    N_eigen = min(N_eigen, H.m)
    vals, vecs, info = eigsolve(H, vec0, N_eigen, :SR; ishermitian, krylovkit_kwargs...)
    return view(vals, 1:N_eigen), view(vecs, 1:N_eigen), info
end


"""
    EDsolve(
        sorted_mbs_block_list::Vector{<: MBS64}, 
        sorted_scat_lists::Vector{<: Scattering}...;
        N::Int64 = 6, showtime = false, method = :sparse,
        element_type::Type = Float64, index_type::Type = Int64, 
        min_sparse_dim::Int64 = 100, max_dense_dim::Int64 = 200,
        krylovkit_kwargs...) -> (vals, vecs)

Main interface function for exact diagonalization of momentum-conserved quantum systems.
Constructs the sparse Hamiltonian matrix from scattering lists and diagonalizes it.

# Arguments
- `sorted_mbs_block_list::Vector{<: MBS64}`: Sorted list of many-body states in the momentum block
- `sorted_scat_list::Vector{<: Scattering}`: Sorted scattering terms (one-body, two-body, etc.)
- `N_eigen::Int64=6`: Number of eigenvalues/eigenvectors to compute (default: 6)

# Keywords
- `showtime::Bool=false`: Whether to print timing information for matrix construction and diagonalization
- `krylovkit_kwargs...`: Additional keyword arguments to pass to KrylovKit.eigsolve


# Returns
- `vals::Vector{Float64}`: Eigenvalues (energies) in ascending order
- `vecs::Vector{Vector{ComplexF64}}`: Corresponding eigenvectors

# Examples
```julia
# Create basis and scattering lists for a 2-site system
basis = ED_mbslist(para, (2,))
blocks, _, _, _ = ED_momentum_block_division(para, basis)
scattering1 = ED_sortedScatteringList_onebody(para)
scattering2 = ED_sortedScatteringList_twobody(para)

# Solve for ground state and first excited state
energies, wavefunctions = EDsolve(blocks[1], scattering1, scattering2; N=1)
println("Ground state energy: ", energies[1])
```
"""
function EDsolve(
    sorted_mbs_block_list::Vector{<: MBS64}, sorted_scat_lists::Vector{<: Scattering}...;
    N::Int64 = 6, showtime = false, method = :sparse,
    element_type::Type = Float64, index_type::Type = Int64, 
    min_sparse_dim::Int64 = 100, max_dense_dim::Int64 = 200,
    krylovkit_kwargs...)



    if method == :map

        error("Linear map method is under development. Please use :sparse or :dense method.")

    elseif method == :sparse || method == :dense

        @assert max_dense_dim > min_sparse_dim
        if method == :sparse && length(sorted_mbs_block_list) < min_sparse_dim
            @warn "Hilbert space dimension < $min_sparse_dim; switch to method=:dense automatically."
            method = :dense
        end
        if method == :dense && length(sorted_mbs_block_list) > max_dense_dim
            @warn "Hilbert space dimension > $max_dense_dim; switch to method=:sparse automatically."
            method = :sparse
        end

        # Construct sparse Hamiltonian matrix from scattering terms
        if showtime
            @time H = ED_HamiltonianMatrix_threaded(sorted_mbs_block_list, sorted_scat_lists...;
                element_type = element_type, index_type = index_type
            )
        else
            H = ED_HamiltonianMatrix_threaded(sorted_mbs_block_list, sorted_scat_lists...;
                element_type = element_type, index_type = index_type
            )
        end

        if method == :sparse

            # Solve the eigenvalue problem
            if showtime
                @time vals, vecs, _ = krylov_matrix_solve(H, N; krylovkit_kwargs...)
            else
                vals, vecs, _ = krylov_matrix_solve(H, N; krylovkit_kwargs...)
            end

        elseif method == :dense

            dim = size(H, 1)
            if dim > 1000
                @warn "Dense diagonalization may be slow for dim=$dim. Consider using :sparse method."
            end
            N > dim && (N = dim)

            # Convert to dense matrix and solve
            if showtime
                @time vals, vecs = eigen(Hermitian(Matrix(H)))
            else
                vals, vecs = eigen(Hermitian(Matrix(H)))
            end
            vals = vals[1:N]
            vecs = vecs[:, 1:N]
            vecs = [vecs[:, i] for i in 1:N]  # Convert to vector of vectors

        end

    else
        error("Unknown method: $method. Use :sparse, :dense, or :map.")
    end

    return vals, vecs
end

function EDsolve(
    HilbertSubspace::Dict{MBS64{bits}, idtype}, HamiltonianOperator::MBSOperator{eltype};
    N::Int64 = 6, showtime = false, method = :sparse, 
    min_sparse_dim::Int64 = 100, max_dense_dim::Int64 = 200,
    krylovkit_kwargs...) where{bits, eltype, idtype}

end



include("analysis/onebody_reduced_density_matrix.jl")
include("analysis/entanglement_entropy.jl")
include("analysis/manybody_connection.jl")





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
