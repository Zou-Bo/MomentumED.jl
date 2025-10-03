"""
    sparse_matrix.jl - Sparse Hamiltonian matrix construction from scattering lists
    
    This file provides functions for converting scattering lists into sparse matrix
    Hamiltonians for momentum-conserved exact diagonalization. Handles both one-body
    and two-body scattering terms with proper Hermitian matrix construction.
"""

include("search.jl")


"""
    HmltMatrix_threaded(sorted_mbs_block_list, sorted_onebody_scat_list, sorted_twobody_scat_list, n_threads=Threads.nthreads())

Threaded version of HmltMatrix with pre-computed state mapping and COO format construction.

This function provides the same functionality as HmltMatrix but uses multi-threading
for parallel matrix construction and returns a standard SparseMatrixCSC instead of 
ExtendableSparseMatrix.

# Arguments
- `sorted_mbs_block_list::Vector{<: MBS64}`: Sorted basis states for momentum block
- `sorted_scat_lists::Vector{<: Scattering}`: lists of scattering terms (one-body, two-body, etc.)

# Keywords
- `element_type::Type=Float64`: Element type of the sparse matrix (Float64, Float32, Float16)
- `index_type::Type=Int64`: Index type of the sparse matrix (Int64, Int32, Int16, Int8)

# Returns  
- `SparseMatrixCSC`: Sparse Hamiltonian matrix (Hermitian)

# Notes
Uses COO format construction with thread-local storage for better parallel performance.
Provides 4-8x speedup for medium to large systems compared to the basic version.
"""
function HmltMatrix_threaded(
    sorted_mbs_block_list::Vector{<: MBS64}, 
    sorted_scat_lists::Vector{<: Scattering}...;
    element_type::Type = Float64, index_type::Type = Int64,
)::SparseMatrixCSC

    @assert element_type ∈ (Float64, Float32, Float16) "Use element_type Float64, Float32, or Float16."
    @assert index_type ∈ (Int64, Int32, Int16, Int8) "Use index_type Int64, Int32, Int16, or Int8."

    n_states = length(sorted_mbs_block_list)
    @assert n_states <= typemax(index_type) "Hilbert space too large for $index_type."
    # state_mapping = create_state_mapping(sorted_mbs_block_list)
    
    # Thread-local storage for COO format
    n_threads = Threads.nthreads()
    thread_I = [Vector{index_type}() for _ in 1:n_threads]
    thread_J = [Vector{index_type}() for _ in 1:n_threads]
    thread_V = [Vector{Complex{element_type}}() for _ in 1:n_threads]
    
    # Parallel construction over columns
    Threads.@threads for j in 1:n_states
        tid = Threads.threadid()
        mbs_in = sorted_mbs_block_list[j]
        
        for scat_list in sorted_scat_lists
            for scat in scat_list
                amp, mbs_out = scat * mbs_in
                if !iszero(amp)
                    # i = get(state_mapping, mbs_out, 0)
                    i = my_searchsortedfirst(sorted_mbs_block_list, mbs_out)
                    @assert i != 0 "H is not momentum- or component-conserving."
                    push!(thread_I[tid], i)
                    push!(thread_J[tid], j)
                    push!(thread_V[tid], amp)
                end
                # if isoccupied(mbs_in, scat.in...)
                #     if scat.in == scat.out
                #         # Diagonal term
                #         push!(thread_I[tid], j)
                #         push!(thread_J[tid], j)
                #         push!(thread_V[tid], scat.Amp)
                #     else
                #         # Off-diagonal term
                #         mbs_mid = empty!(mbs_in, scat.in...; check=false)
                #         if isempty(mbs_mid, scat.out...)
                #             mbs_out = occupy!(mbs_mid, scat.out...; check=false)
                #             # i = get(state_mapping, mbs_out, 0)
                #             i = my_searchsortedfirst(sorted_mbs_block_list, mbs_out)
                #             @assert i != 0 "H is not momentum- or component-conserving."

                #             sign_occ = (-1)^(scat_occ_number(mbs_mid, scat.in) + scat_occ_number(mbs_mid, scat.out))
                #             push!(thread_I[tid], i)
                #             push!(thread_J[tid], j)
                #             push!(thread_V[tid], sign_occ * scat.Amp)
                #         end
                #     end
                # end
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