"""
    sparse_matrix.jl - Sparse Hamiltonian matrix construction from Scatter lists
    
    This file provides functions for converting Scatter lists into sparse matrix
    Hamiltonians for momentum-conserved exact diagonalization. Handles both one-body
    and two-body Scatter terms with proper Hermitian matrix construction.
"""




"""
    ED_HamiltonianMatrix_threaded(sorted_mbs_block_list, sorted_scat_lists...)

Threaded version of generating Hamiltonian Matrix with pre-computed state mapping and COO format construction.

# Arguments
- `sorted_mbs_block_list::Vector{<: MBS64}`: Sorted basis states for momentum block
- `sorted_scat_lists::Vector{<: Scatter}`: lists of Scatter terms (one-body, two-body, etc.)

# Keywords
- `element_type::Type=Float64`: Element type of the sparse matrix (Float64, Float32, Float16)
- `index_type::Type=Int64`: Index type of the sparse matrix (Int64, Int32, Int16, Int8)

# Returns  
- `SparseMatrixCSC`: Sparse Hamiltonian matrix (Hermitian)

# Notes
Uses COO format construction with thread-local storage for better parallel performance.
Provides 4-8x speedup for medium to large systems compared to the one-thread version.
"""
function ED_HamiltonianMatrix_threaded(
    subspace::HilbertSubspace, 
    sorted_scat_lists::Vector{<: Scatter}...;
    element_type::Type = Float64, index_type::Type = idtype(subspace),
)::SparseMatrixCSC

    @assert element_type ∈ (Float64, Float32) "element_type=$element_type. Use element_type Float64, Float32."
    @assert index_type ∈ (Int64, Int32, Int16, Int8, UInt64, UInt32, UInt16, UInt8) """
    index_type=$index_type. Use index_type Int64, Int32, Int16, Int8, UInt64, UInt32, UInt16, or UInt8."""
    @assert isempty(subspace.dict) || index_type == index_type(subspace) """
    Input index_type is inconsistant to subspace dictionary index type."""

    n_states = length(subspace)
    @assert n_states <= typemax(index_type) "Hilbert space too large for $index_type."
    
    # Thread-local storage for COO format
    n_threads = Threads.nthreads()
    thread_I = [Vector{index_type}() for _ in 1:n_threads]
    thread_J = [Vector{index_type}() for _ in 1:n_threads]
    thread_V = [Vector{Complex{element_type}}() for _ in 1:n_threads]
    
    # Parallel construction over columns
    # Threads.@threads 
    for j in 1:n_states
        tid = Threads.threadid()# - Threads.nthreads(:interactive)
        mbs_in = subspace.list[j]
        
        for scat_list in sorted_scat_lists
            for scat in scat_list
                amp, mbs_out = scat * mbs_in
                if !iszero(amp)
                    i = get(subspace, mbs_out)
                    @assert i != 0 "H is not momentum- or component-conserving."
                    push!(thread_I[tid], i)
                    push!(thread_J[tid], j)
                    push!(thread_V[tid], amp)
                end
            end
        end
    end
    
    # Merge thread-local results
    I = reduce(vcat, thread_I)
    J = reduce(vcat, thread_J)
    V = reduce(vcat, thread_V)
    
    # Convert to sparse matrix (Hermitian)
    H = sparse(I, J, V, n_states, n_states)
    return sparse(Hermitian(H, :U))
end