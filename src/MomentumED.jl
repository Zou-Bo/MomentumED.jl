"""
This module gives general methods for 2D momentum-block-diagonalized ED calculations.
Sectors of other quantum Numbers should be handled outside this module.
This module only sets sectors of total (crystal) momentum, also called blocks.
"""
module MomentumED

public MBS64, Scattering, ED_mbslist_onecomponent
export EDPara, ED_mbslist, ED_momentum_block_division
export ED_sortedScatteringList_onebody
export ED_sortedScatteringList_twobody
export EDsolve, ED_etg_entropy, ED_connection_integral

using LinearAlgebra, SparseArrays

# Include utilities
include("init_parameter.jl")
include("MBS.jl")
include("momentum_decomposition.jl")
include("scattering.jl")
include("sparse_matrix.jl")


"""
    ED_sortedScatteringList_onebody(para::EDPara) -> Vector{Scattering{1}}

Generate sorted lists of one-body scattering terms from the parameters.

Extracts one-body terms from EDpara.H_onebody for multi-component systems and converts
them to scattering terms with proper normal ordering.

# Arguments
- `para::EDPara`: Parameter structure containing system configuration

# Returns
- `Vector{Scattering{1}}`: Sorted list of one-body scattering terms

# Details
- Maps component indices to global orbital indices using: `global_index = k + Nk * (ch - 1) + Nk * Nch * (cc - 1)`
- Applies normal ordering to avoid double-counting
- Uses `sortMergeScatteringList` to eliminate duplicates and sort terms
- Only includes non-zero amplitude terms

# Example
```julia
para = EDPara(k_list=k_list, Gk=(3, 5), V_int=V_int)
scattering1 = ED_sortedScatteringList_onebody(para)
```
"""
function ED_sortedScatteringList_onebody(para::EDPara)
    sct_list1 = Vector{Scattering{1}}()
    Nk = para.Nk
    Nch = para.Nc_hopping
    Ncc = para.Nc_conserve

    # Extract one-body terms from H1[ch1, ch2, cc, k]
    for ch1 in 1:Nch, ch2 in 1:Nch, cc in 1:Ncc, k in 1:Nk
        V = para.H_onebody[ch1, ch2, cc, k]
        if !iszero(V)
            # Map component indices to global orbital indices
            i_ot = k + Nk * (ch1 - 1) + Nk * Nch * (cc - 1)  # output orbital
            i_in = k + Nk * (ch2 - 1) + Nk * Nch * (cc - 1)  # input orbital

            # Create scattering term with normal ordering
            i_in >= i_ot && push!(sct_list1, NormalScattering(V, i_ot, i_in))
        end
    end
    
    return sortMergeScatteringList(sct_list1)
end



"""
    group_momentum_pairs(para::EDPara) -> Dict{Tuple{Int64,Int64}, Vector{Tuple{Int64,Int64}}}

Generate grouped momentum pairs by their total momentum.

Creates a dictionary mapping total momentum quantum numbers to lists of 
momentum index pairs that conserve that total momentum.

# Arguments
- `para::EDPara`: Parameter structure containing system configuration

# Returns
- `Dict{Tuple{Int64,Int64}, Vector{Tuple{Int64,Int64}}}`: Dictionary where:
  - Keys are total momentum tuples `(K1, K2)`
  - Values are vectors of momentum index pairs `[(i,j), ...]` with that total momentum

# Details
- Generates all possible pairs `(i,j)` with `i >= j` to avoid duplicates
- Uses `MBS64_totalmomentum(para, i, j)` to compute total momentum for each pair
- Essential for efficient two-body scattering term generation
- Enables momentum conservation enforcement in Hamiltonian construction

# Example
```julia
para = EDPara(k_list=k_list, Gk=(3, 5), V_int=V_int)
groups = group_momentum_pairs(para)
# Access all pairs with total momentum (0, 0)
pairs_with_zero_momentum = groups[(0, 0)]
```
"""
function group_momentum_pairs(para::EDPara)
    
    # Dictionary to store momentum groups
    momentum_groups = Dict{Tuple{Int64,Int64}, Vector{Tuple{Int64,Int64}}}()
    
    # Generate all possible pairs (including identical pairs)
    for i in 1:para.Nk, j in 1:i  # i >= j to avoid duplicates
        # Calculate total momentum using existing function
        K_total = MBS64_totalmomentum(para, i, j)
        pair_indices = (i, j)
        
        # Add to appropriate group
        if haskey(momentum_groups, K_total)
            push!(momentum_groups[K_total], pair_indices)
        else
            momentum_groups[K_total] = [pair_indices]
        end
    end
    
    return momentum_groups
end

"""
    scat_pair_group(pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara;
                   kshift::Tuple{Float64, Float64} = (0.0, 0.0), output::Bool = false) -> Vector{Scattering{2}}

Generate all scattering terms between momentum pairs with the same total momentum.

Creates two-body scattering terms for all possible transitions between momentum pairs
that conserve total momentum, including all component index combinations.

# Arguments
- `pair_group::Vector{Tuple{Int64,Int64}}`: List of momentum index pairs with same total momentum
- `para::EDPara`: Parameter structure containing system configuration
- `kshift::Tuple{Float64, Float64}=(0.0, 0.0)`: Momentum shift for twisted boundary conditions
- `output::Bool=false`: Whether to print debugging information

# Returns
- `Vector{Scattering{2}}`: List of two-body scattering terms for this momentum group

# Details
- Iterates over all input/output momentum pair combinations within the group
- Generates all component index combinations for each momentum pair
- Maps momentum and component indices to global orbital indices
- Applies normal ordering: `minmax(i1, i2) >= minmax(f1, f2)`
- Calculates scattering amplitudes using `int_amp` function with momentum shift
- Includes both direct and exchange contributions
- Handles identical orbital pairs with proper exclusion

# Physics
The scattering amplitude includes:
- Direct term: `int_amp(i1, i2, f1, f2, para; kshift=kshift)`
- Exchange term: `int_amp(i2, i1, f1, f2, para; kshift=kshift)`
- Total amplitude: `amp = amp_direct - amp_exchange`

# Example
```julia
# Get momentum pairs with total momentum (0, 0)
groups = group_momentum_pairs(para)
zero_momentum_pairs = groups[(0, 0)]
# Generate all scattering terms for this momentum group
scattering_terms = scat_pair_group(zero_momentum_pairs, para)
```
"""
function scat_pair_group(pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara;
    kshift::Tuple{Float64, Float64} = (0.0, 0.0), output::Bool = false)::Vector{Scattering{2}}
    
    scattering_list = Vector{Scattering{2}}()
    Nc = para.Nc
    Nk = para.Nk
    
    # Iterate over all input and output pairs
    for (ki1, ki2) in pair_group, (kf1, kf2) in pair_group
        output && println()
        output && println("ki1, ki2, kf1, kf2 = ($ki1, $ki2), ($kf1, $kf2)")
        # Generate all component index combinations
        for ci1 in 1:Nc, ci2 in 1:Nc, cf1 in 1:Nc, cf2 in 1:Nc
            
            # Map to global orbital indices
            # Global index = momentum_index + Nk * (component_index - 1)
            f1 = kf1 + Nk * (cf1 - 1)
            f2 = kf2 + Nk * (cf2 - 1)
            i1 = ki1 + Nk * (ci1 - 1)
            i2 = ki2 + Nk * (ci2 - 1)

            # no duplicate indices
            if i1 == i2 || f1 == f2
                continue
            end

            if ki1 == ki2 && i1 < i2
                continue
            end

            if kf1 == kf2 && f1 < f2
                continue
            end

            # inverse scattering only need to count onece, as the Hamiltonian is generated with upper half Hermitian()
            if minmax(i1, i2) >= minmax(f1, f2)

                # Calculate the direct and exchange amplitudes
                output && print(fldmod1(i1, Nk), fldmod1(i2, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk),"        ")
                amp_direct = int_amp(i1, i2, f1, f2, para; kshift=kshift)
                output && print(fldmod1(i2, Nk), fldmod1(i1, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk))
                amp_exchange = int_amp(i2, i1, f1, f2, para; kshift=kshift)
                amp = amp_direct - amp_exchange
                iszero(amp) || push!(scattering_list, NormalScattering(amp, f1, f2, i2, i1))
                output && println()
            end
        
        end
    end
    
    return scattering_list
end

"""
    ED_sortedScatteringList_twobody(para::EDPara; kshift::Tuple{Float64, Float64} = (0.0, 0.0)) -> Vector{Scattering{2}}

Generate sorted lists of two-body scattering terms from the parameters.

Uses the interaction function from EDPara.V_int to calculate scattering amplitudes
for all possible two-body processes, grouped by total momentum conservation.

# Arguments
- `para::EDPara`: Parameter structure containing system configuration
- `kshift::Tuple{Float64, Float64}=(0.0, 0.0)`: Momentum shift for twisted boundary conditions

# Returns
- `Vector{Scattering{2}}`: Sorted list of two-body scattering terms

# Details
- Groups momentum pairs by total momentum for efficiency
- Generates all component index combinations for each momentum pair
- Applies normal ordering: `minmax(i1, i2) >= minmax(f1, f2)`
- Includes both direct and exchange contributions: `amp = amp_direct - amp_exchange`
- Uses momentum shift in interaction calculations: `(k_list .+ kshift) ./ Gk`
- Applies `sortMergeScatteringList` to eliminate duplicates and sort terms

# Example
```julia
para = EDPara(k_list=k_list, Gk=(3, 5), V_int=V_int)
# Without momentum shift
scattering2 = ED_sortedScatteringList_twobody(para)
# With twisted boundary conditions
scattering2_shifted = ED_sortedScatteringList_twobody(para; kshift=(0.1, 0.1))
```
"""
function ED_sortedScatteringList_twobody(para::EDPara; kshift::Tuple{Float64, Float64} = (0.0, 0.0))

    sct_list2 = Vector{Scattering{2}}()
    
    momentum_groups = group_momentum_pairs(para)
    
    for (K_total, pairs) in momentum_groups
        append!(sct_list2, scat_pair_group(pairs, para; kshift=kshift))
    end
    
    return sortMergeScatteringList(sct_list2)
end







"""
    matrix_solve(H::SparseMatrixCSC{ComplexF64,Int64}, N_eigen::Int64=6; 
        converge_warning::Bool=false, krylovkit_kwargs...) -> (vals, vecs)

Solve the sparse Hamiltonian matrix using KrylovKit's eigsolve function for the lowest n eigenvalues and eigenvectors.

# Arguments
- `H::SparseMatrixCSC{ComplexF64,Int64}`: Sparse Hamiltonian matrix to diagonalize
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
    H::SparseMatrixCSC{ComplexF64,Int64},
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




include("entanglement_entropy.jl")
include("manybody_connection.jl")





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