"""
    sparse_matrix.jl - Sparse Hamiltonian matrix construction from Scatter lists
    
    This file provides functions for converting Scatter lists into sparse matrix
    Hamiltonians for momentum-conserved exact diagonalization. Handles both one-body
    and two-body Scatter terms with proper Hermitian matrix construction.
"""

# depreciated, only complete_lower always false
function ED_HamiltonianMatrix_threaded_COO2CSC(
    subspace::HilbertSubspace{bits}, 
    sorted_scat_lists::Vector{<: Scatter}...; 
    isupper::Bool = true, complete_lower::Bool = false,
    element_type::Type{ET} = Float64, index_type::Type{IT} = Int64,
) where {bits, ET <: AbstractFloat, IT <: Integer}

    @assert element_type ∈ (Float64, Float32) "element_type=$element_type. Use element_type Float64, Float32."
    @assert index_type ∈ (Int128, Int64, Int32, UInt128, UInt64, UInt32) """
    index_type=$index_type. Use index_type Int128, Int64, Int32, Int128, UInt64, or UInt32."""

    n_states = length(subspace)
    @assert n_states <= typemax(index_type) "Hilbert space too large for $index_type."
    
    # Thread-local storage for COO format
    n_threads = Threads.nthreads()
    thread_I = [Vector{index_type}() for _ in 1:n_threads]
    thread_J = [Vector{index_type}() for _ in 1:n_threads]
    thread_V = [Vector{Complex{element_type}}() for _ in 1:n_threads]
    
    # Parallel construction over columns
    Threads.@threads for j in 1:n_states
        tid = Threads.threadid() - Threads.nthreads(:interactive)
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
    if isupper
        return sparse(Hermitian(H, :U))
    else
        return H
    end
end

function ED_HamiltonianMatrix_threaded_CSCdirect(
    subspace::HilbertSubspace{bits}, 
    sorted_scat_lists::Vector{<: Scatter}...; 
    isupper::Bool = true, complete_lower::Bool = true,
    element_type::Type{ET} = Float64, index_type::Type{IT} = Int64,
) where {bits, ET <: AbstractFloat, IT <: Integer}

    @assert element_type ∈ (Float64, Float32) "element_type=$element_type. Use element_type Float64, Float32."
    @assert index_type ∈ (Int128, Int64, Int32, UInt128, UInt64, UInt32) """
    index_type=$index_type. Use index_type Int128, Int64, Int32, Int128, UInt64, or UInt32."""

    n_states = length(subspace)
    @assert n_states <= typemax(index_type) "Hilbert space too large for $index_type."

    # --- 1. Partition Columns for Multithreading ---
    n_chunks = Threads.nthreads()
    cols_per_chunk = div(n_states, n_chunks)
    
    # Storage for the vertical strips of the matrix
    strips = Vector{SparseMatrixCSC{Complex{ET}, IT}}(undef, n_chunks)

    # --- 2. Parallel Construction ---
    Threads.@threads for t in 1:n_chunks
        # --- ALLOCATE BUFFERS ONCE PER THREAD ---
        # These will grow to the size of the largest column and stay there.
        col_buf_I = Vector{IT}()
        col_buf_V = Vector{Complex{ET}}()
        col_buf_P = Vector{Int}()

        # Determine the column range for this thread
        start_col = (t - 1) * cols_per_chunk + 1
        end_col = (t == n_chunks) ? n_states : t * cols_per_chunk
        
        # Pre-allocate CSC vectors for this strip
        # strip_colptr must start with 1.
        strip_colptr = Vector{IT}(undef, 1)
        strip_colptr[1] = 1
        
        # We start with empty data vectors and append to them dynamically.
        # Julia's push! is amortized O(1), so this is efficient.
        strip_rowval = Vector{IT}()
        strip_nzval = Vector{Complex{ET}}()

        # Buffers reused for all the j
        # Iterate strictly over the columns THIS thread owns
        for j in start_col:end_col

            mbs_in = subspace.list[j]
            empty!(col_buf_I)
            empty!(col_buf_V)

            if isupper && complete_lower
                for scat_list in sorted_scat_lists
                    for scat in scat_list
                        amp, mbs_out = scat * mbs_in
                        if !iszero(amp)
                            i = get(subspace, mbs_out)
                            @assert i != 0 "H is not momentum- or component-conserving."
                            push!(col_buf_I, i)
                            push!(col_buf_V, amp)
                        end
                        if !isdiagonal(scat)
                            amp, mbs_out = mbs_in * scat # inversely scatting
                            if !iszero(amp)
                                i = get(subspace, mbs_out)
                                @assert i != 0 "H is not momentum- or component-conserving."
                                push!(col_buf_I, i)
                                push!(col_buf_V, conj(amp))
                            end
                        end
                    end
                end
            else
                for scat_list in sorted_scat_lists
                    for scat in scat_list
                        amp, mbs_out = scat * mbs_in
                        if !iszero(amp)
                            i = get(subspace, mbs_out)
                            @assert i != 0 "H is not momentum- or component-conserving."
                            push!(col_buf_I, i)
                            push!(col_buf_V, amp)
                        end
                    end
                end
            end

            if !isempty(col_buf_I)
                m = length(col_buf_I)
                resize!(col_buf_P, m)
                # In-place sort of permutation indices
                sortperm!(col_buf_P, col_buf_I) 
                permute!(col_buf_I, col_buf_P)
                permute!(col_buf_V, col_buf_P)

                r = 1
                l = 1               # length of processed part
                i = col_buf_I[r]    # row-index of current element

                # main loop
                while r < m
                    r += 1
                    i2 = col_buf_I[r]
                    if i2 == i  # accumulate r-th to the l-th entry
                        col_buf_V[l] += col_buf_V[r]
                    else  # advance l, and move r-th to l-th
                        l += 1
                        i = i2
                        if l < r
                            col_buf_I[l] = i
                            col_buf_V[l] = col_buf_V[r]
                        end
                    end
                end

                # Direct Append (No sorting needed per user contract)
                append!(strip_rowval, view(col_buf_I, 1:l))
                append!(strip_nzval, view(col_buf_V, 1:l))
            end
            
            # Update colptr: The next column starts where the current one ends
            push!(strip_colptr, length(strip_nzval) + 1)
        end
        
        # Create the strip (n_states x chunk_width)
        chunk_width = end_col - start_col + 1
        strips[t] = SparseMatrixCSC(n_states, chunk_width, strip_colptr, strip_rowval, strip_nzval)
    end

    # --- 3. Assembly ---
    # Glue strips together side-by-side. 
    # This is a fast memory copy operation.
    H = hcat(strips...)

    if isupper && !complete_lower    
        # Wrap in Hermitian
        # Returns a view that behaves like the full matrix but stores 50% data
        return Hermitian(H, :U)
    else
        return H
    end
end

"""
    SparseHmltMatrix(sorted_mbs_block_list, sorted_scat_lists...)

Multithread generating Hamiltonian Matrix with pre-computed scattering terms.

# Arguments
- `subspace::HilbertSubspace`: Hilbert subspace basis
- `sorted_scat_lists::Vector{<: Scatter}`: lists of Scatter terms (one-body, two-body, etc.)

# Keywords
- `isupper::Bool = true`: if true, scat_lists only have the upper-triangular terms.
- `complete_lower::Bool = true`: when isupper=true, if complete_lower=false, return full sparse matrix; else, return Hermitian warp.
- `element_type::Type{<: AbstractFloat} = Float64`: Element type of the sparse matrix (Float64, Float32, Float16)
- `index_type::Type{<: Integer} = Int64`: Index type of the sparse matrix (Int64, Int32, Int16, Int8)

# Returns  
- `SparseMatrixCSC`: Sparse Hamiltonian matrix
- `Hermitian{SparseMatrixCSC}`: with a Hermitian warp, saves memory but cost more time in multiplication

"""
SparseHmltMatrix = ED_HamiltonianMatrix_threaded_CSCdirect
# SparseHmltMatrix = ED_HamiltonianMatrix_threaded_COO2CSC





"""
    krylov_matrix_solve(H::SparseMatrixCSC{ComplexF64, Int64}, N_eigen::Int64=6; 
        ishermitian::Bool=false, krylovkit_kwargs...) -> (vals, vecs)

Solve the sparse Hamiltonian matrix using KrylovKit's eigsolve function for the lowest `N_eigen` eigenvalues and eigenvectors, with :SR (smallest real) eigenvalue selection

# Arguments
- `H::SparseMatrixCSC{Complex{eltype}, idtype}`: Sparse Hamiltonian matrix to diagonalize
- `N_eigen::Int64=6`: Number of eigenvalues/eigenvectors to compute (default: 6)

# Keywords
- `ishermitian::Bool=true`: Whether the matrix is Hermitian (default: true)
- `krylovkit_kwargs...`: Additional keyword arguments to pass to KrylovKit.eigsolve

# Returns
- `vals::Vector{eltype}`: Eigenvalues (energies) in ascending order
- `vecs::Vector{Vector{Complex{eltype}}}`: Corresponding eigenvectors
- `info`: Convergence information from KrylovKit

# Examples
```julia
# Solve for 3 lowest eigenstates
vals, vecs, info = krylov_matrix_solve(H_matrix, 3)
println("Ground state energy: ", vals[1])
```
"""
function krylov_matrix_solve(
    H::Union{Hermitian{Complex{eltype}, SparseMatrixCSC{Complex{eltype}}}, SparseMatrixCSC{Complex{eltype}}}, 
    N_eigen::Int64; ishermitian::Bool = true, krylovkit_kwargs...
)::Tuple{Vector{eltype}, Vector{Vector{Complex{eltype}}}, Any} where {eltype<:AbstractFloat}

    vec0 = rand(Complex{eltype}, H.m)
    N_eigen = min(N_eigen, H.m)
    eigsolve(H, vec0, N_eigen, :SR; ishermitian, krylovkit_kwargs...)
end

