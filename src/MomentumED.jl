"""
This module gives general methods for 2D momentum-block-diagonalized ED calculations.
Sectors of other quantum Numbers should be handled outside this module.
This module only sets sectors of total (crystal) momentum, also called blocks.
"""
module MomentumED

# type
public MBS64, Scattering
# preparation
public ED_mbslist_onecomponent
export EDPara, ED_mbslist, ED_momentum_block_division
export ED_sortedScatteringList_onebody
export ED_sortedScatteringList_twobody
# main solving function
export EDsolve
# analysis
export ED_etg_entropy, ED_connection_integral

using LinearAlgebra
using SparseArrays
using KrylovKit
using Base.Threads


# Include utilities
include("type/manybodystate.jl")
include("type/scattering.jl")
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
- `H::SparseMatrixCSC{ComplexF64, Int64}`: Sparse Hamiltonian matrix to diagonalize
- `N_eigen::Int64=6`: Number of eigenvalues/eigenvectors to compute (default: 6)

# Keywords
- `converge_warning::Bool=false`: Whether to show a warning if the solver does not converge
- `krylovkit_kwargs...`: Additional keyword arguments to pass to KrylovKit.eigsolve

# Returns
- `vals::Vector{Float64}`: Eigenvalues (energies) in ascending order
- `vecs::Vector{Vector{ComplexF64}}`: Corresponding eigenvectors

# Examples
```julia
# Solve for 3 lowest eigenstates
vals, vecs = matrix_solve(H_matrix, 3)
println("Ground state energy: ", vals[1])
```

# Notes
- Uses KrylovKit's eigsolve with :SR (smallest real) eigenvalue selection
- Assumes Hermitian matrix (standard for quantum Hamiltonians)
- Random initial vector ensures good convergence properties
- Automatically handles convergence warnings from KrylovKit
- For better control over convergence, consider using KrylovKit directly
"""
function matrix_solve(
    H::SparseMatrixCSC{ComplexF64, Int64},
    N_eigen::Int64=6;
    converge_warning::Bool=false, krylovkit_kwargs...
)::Tuple{Vector{Float64}, Vector{Vector{ComplexF64}}}
    dim = H.m
    vec0 = rand(ComplexF64, dim)
    N_eigen > dim && (N_eigen = dim)
    vals, vecs, info = eigsolve(H, vec0, N_eigen, :SR; ishermitian=true, krylovkit_kwargs...)
    # Handle convergence information from KrylovKit
    if !(info.converged == true || info.converged == 1)
        if converge_warning
            @warn "Eigensolver did not converge. Residual norm: $(info.normres)"
        end
    end

    return vals[1:N_eigen], vecs[1:N_eigen]
end


"""
    EDsolve(sorted_mbs_block_list::Vector{MBS64{bits}}, 
           sorted_onebody_scat_list::Vector{Scattering{1}},
           sorted_twobody_scat_list::Vector{Scattering{2}},
           N_eigen::Int64=6; showtime::Bool = false, converge_warning::Bool=false,
           krylovkit_kwargs...) -> (vals, vecs)

Main interface function for exact diagonalization of momentum-conserved quantum systems.
Constructs the sparse Hamiltonian matrix from scattering lists and diagonalizes it.

# Arguments
- `sorted_mbs_block_list::Vector{MBS64{bits}}`: Sorted list of many-body states in the momentum block
- `sorted_onebody_scat_list::Vector{Scattering{1}}`: Sorted one-body scattering terms
- `sorted_twobody_scat_list::Vector{Scattering{2}}`: Sorted two-body scattering terms  
- `N_eigen::Int64=6`: Number of eigenvalues/eigenvectors to compute (default: 6)

# Keywords
- `showtime::Bool=false`: Whether to print timing information for matrix construction and diagonalization
- `converge_warning::Bool=false`: Whether to show a warning if the eigensolver does not converge
- `krylovkit_kwargs...`: Additional keyword arguments to pass to KrylovKit.eigsolve

# Type Parameters
- `bits`: Number of bits in MBS64 type (determines system size)

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
energies, wavefunctions = EDsolve(blocks[1], scattering1, scattering2, 2, 1)
println("Ground state energy: ", energies[1])
```

# Physics Notes
- Conserves total momentum through block diagonalization
- Uses scattering formalism for efficient Hamiltonian construction
- Suitable for both real and complex Hamiltonian matrices
- Automatically handles Hermitian symmetry optimization

# Performance
- Memory efficient: Uses sparse matrix storage
- Computationally efficient: Krylov subspace methods for large sparse systems
- Typical use case: Systems with 10-20 single-particle orbitals
"""
function EDsolve(sorted_mbs_block_list::Vector{MBS64{bits}}, 
    sorted_onebody_scat_list::Vector{Scattering{1}},
    sorted_twobody_scat_list::Vector{Scattering{2}}, 
    N_eigen::Int64=6; showtime = false, converge_warning::Bool=false,
    krylovkit_kwargs...) where {bits}
    # Construct sparse Hamiltonian matrix from scattering terms
    if showtime
        @time H = HmltMatrix_threaded(sorted_mbs_block_list, 
            sorted_onebody_scat_list, sorted_twobody_scat_list;
        )
    else
        H = HmltMatrix_threaded(sorted_mbs_block_list,
            sorted_onebody_scat_list, sorted_twobody_scat_list;
        )
    end

    # Solve the eigenvalue problem
    if showtime
        @time vals, vecs = matrix_solve(H, N_eigen; converge_warning = converge_warning, krylovkit_kwargs...)
    else
        vals, vecs = matrix_solve(H, N_eigen; converge_warning = converge_warning, krylovkit_kwargs...)
    end

    return vals, vecs
end




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