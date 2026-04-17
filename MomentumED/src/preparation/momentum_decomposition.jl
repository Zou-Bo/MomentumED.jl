"""
This file provides:
Iterators over MBS64 combinations of `t` electrons in `n` states (in one component) in sorted order;
Recursive iteration function over multi-component MBS64 with given para and electron number of each component;
Generating Hilbert subspaces distinguished by total momentum using the previous function.
"""


"""
    mbslist_onecomponent(para::EDPara, N_in_one::Int64[, mask], start_end::Int64...)

Construct the MBS list(iterator) of N electrons in one conserved component.

Generates all possible MBS64 states with exactly N_in_one particles in the
specified component, using combinations of orbital indices.

# Arguments
- `para::EDPara`: Parameter structure containing momentum and component information
- `N_in_one::Int64`: Number of particles in the conserved component

# Returns
- `Iterator{MBS64}`: Sorted list of MBS64 states with specified particle number

# Example
```julia
para = EDPara(k_list=[0 1; 0 0], Nc_mix=1, Nc_conserve=1)
states = ED_mbslist_onecomponent(para, 2)  # 2 particles in 2 orbitals
```
"""
function mbslist_onecomponent(N_one_conserve::Int64, N_in_one::Int64, start_end::Int64...)
    @assert 0 <= N_in_one <= N_one_conserve "Invalid number of electrons in one component"
    return ColexMBS64(N_one_conserve, N_in_one, start_end...)
end
mbslist_onecomponent(para::EDPara, N_in_one::Int64, start_end::Int64...) = mbslist_onecomponent(para.Nk * para.Nc_mix, N_in_one, start_end...)
function mbslist_onecomponent(N_one_conserve::Int64, N_in_one::Int64, mask::Union{Nothing, Vector{Int64}}, start_end::Int64...)
    isnothing(mask) && return mbslist_onecomponent(N_one_conserve, N_in_one, start_end...)
    @assert 0 <= N_in_one <= N_one_conserve "Invalid number of electrons in one component"
    @assert isempty(mask) || 1 <= minimum(mask) && maximum(mask) <= N_one_conserve "Invalid orbital index in mask for one component, mask=$mask, Nstate=$Nstate"
    return ColexMBS64Mask(N_one_conserve, N_in_one, mask, start_end...)
end
mbslist_onecomponent(para::EDPara, N_in_one::Int64, mask::Union{Nothing, Vector{Int64}}, start_end::Int64...) = mbslist_onecomponent(para.Nk * para.Nc_mix, N_in_one, mask, start_end...)

"""
    MBS_totalmomentum(para::EDPara, mbs::MBS64)

Calculate the total momentum (K1, K2) of a many-body state.
The momentum is mod G if G is nonzero (from para.Gk).
"""
function MBS_totalmomentum(k_list::Matrix{Int64}, Gk::NTuple{dim, Int64}, mbs::MBS64;
    Nk = size(k_list, 2))::NTuple{dim, Int64} where {dim}
    # momentum are integers
    k = ntuple(_ -> 0, Val(dim))
    for_each_occ(mbs) do i
        i_k = mod1(i, Nk)
        k = ntuple(d -> k[d] + k_list[d, i_k], Val(dim))
    end
    return momentum_residue(k, Gk)
end
MBS_totalmomentum(para::EDPara, mbs::MBS64) = MBS_totalmomentum(para.k_list, para.Gk, mbs; Nk = para.Nk)

"""
    MBS_totalmomentum(para::EDPara, i_list::Tuple{Vararg{Int64}})

Calculate the total momentum (K1, K2) from a list of occupied orbital indices.
`i_list` can contain repeated numbers,
and the momentum of that orbital will be added multiple times.
"""
function MBS_totalmomentum(para::EDPara{dim}, i_list::Tuple{Vararg{Int64}})::NTuple{dim, Int64} where {dim}
    # momentum are integers
    k = ntuple(_ -> 0, Val(dim))
    Gk = para.Gk
    @inbounds for i in i_list
        i_k = mod1(i, para.Nk)
        k = ntuple(d -> k[d] + para.k_list[d, i_k], Val(dim))
    end
    return momentum_residue(k, Gk)
end

"""
    mbslist_recursive_iteration!(subspace_lists, subspace_k1, subspace_k2, para, N_each_component, accumulated_mbs, accumulated_momentum; mask=nothing)

An internal recursive function that constructs many-body states and sorts them into momentum subspaces.

The function works by processing one conserved component at a time. It iterates through all valid states for the current component and then calls itself recursively for the next component, accumulating the state (`MBS64`) and total momentum.

# Recursion Logic
The recursion proceeds from the last element of `N_each_component` to the first. In each step, it generates all possible states for the current component based on the particle number `abs(N_each_component[end])` and combines them (via `*`) with the `accumulated_mbs` from the previous steps. The momentum is also updated.

When the recursion is complete (`N_each_component` is empty), the final state and its total momentum are known, and the state is pushed into the correct momentum subspace in `subspaces`.

# Particle vs. Hole Creation
The sign of the number in `N_each_component` determines whether particles or holes are created:
- **Positive `N`**: Creates `N` particles in an empty sea of orbitals for the current component.
- **Negative `N`**: Creates `abs(N)` holes in a filled sea of orbitals. This is used for calculations like the particle reduced density matrix where one considers states in the complementary space.

# Keywords
The `mask` argument, if provided, restricts the set of orbitals that can be occupied (for positive `N`) or made into holes (for negative `N`) for the single-component states.
`mask` is a Vector{Int64} and treated as sorted.

"""
function mbslist_recursive_iteration!(subspace_lists::Vector{Vector{MBS64{bits}}}, 
    subspace_k::Vector{NTuple{dim, Int64}}, para::EDPara{dim}, N_each_component::Vector{Int64}, 
    accumulated_mbs::MBS64 = reinterpret(MBS64{0}, 0), 
    accumulated_momentum::NTuple{dim, Int64} = ntuple(_ -> 0, Val(dim)), start_end::Int64...; 
    mask::Union{Nothing, Vector{Int64}} = nothing,
    selection_rule::Function,
) where {bits, dim}

    if !isempty(N_each_component) 
        PRINT_RECURSIVE_MOMENTUM_DIVISION && println("Enter loop with $N_each_component. 
        Momentum $(accumulated_momentum[1]) $(accumulated_momentum[2])\n\t$accumulated_mbs\n")
        if !isnothing(mask)
            num_component_orbitals = para.Nk * para.Nc_mix * (length(N_each_component) - 1)
            i = searchsortedlast(mask, num_component_orbitals)
            mask_this_component = mask[i+1:end] .- num_component_orbitals
            for mbs_smaller in mbslist_onecomponent(para, abs(N_each_component[end]), mask_this_component, start_end...)
                PRINT_RECURSIVE_MOMENTUM_DIVISION && println("generated mbs in the new component:\n$mbs_smaller")
                new_momentum = sign(N_each_component[end]) .* MBS_totalmomentum(para, mbs_smaller)
                mbslist_recursive_iteration!(subspace_lists, subspace_k, para, 
                    N_each_component[begin:end-1],          # remaining components
                    accumulated_mbs * mbs_smaller,          # updated MBS64
                    accumulated_momentum .+ new_momentum;   # updated momentum 
                    mask = mask[begin:i], 
                    selection_rule
                )
            end
        else
            for mbs_smaller in mbslist_onecomponent(para, abs(N_each_component[end]), start_end...)
                PRINT_RECURSIVE_MOMENTUM_DIVISION && println("generated mbs in the new component:\n$mbs_smaller")
                new_momentum = sign(N_each_component[end]) .* MBS_totalmomentum(para, mbs_smaller)
                mbslist_recursive_iteration!(subspace_lists, subspace_k, para, 
                    N_each_component[begin:end-1],          # remaining components
                    accumulated_mbs * mbs_smaller,          # updated MBS64
                    accumulated_momentum .+ new_momentum;   # updated momentum 
                    mask = nothing, 
                    selection_rule
                )
            end
        end

    else
        if selection_rule(accumulated_mbs)
            Gk = para.Gk
            k = momentum_residue(accumulated_momentum, Gk)
            PRINT_RECURSIVE_MOMENTUM_DIVISION && println("\tfinally: momentum $k\t$accumulated_mbs")
            index = findfirst(==(k), subspace_k)
            if !isnothing(index)
                push!(subspace_lists[index], accumulated_mbs)
            end
        end
    end
end

"""
    ED_momentum_subspaces(para::EDPara, N_each_component; kwargs...) -> Tuple{Vector{HilbertSubspace}, Vector{Int64}, Vector{Int64}}

Constructs Hilbert subspaces, each corresponding to a specific total momentum.

This is the main user-facing function to generate the basis states for a momentum-conserved calculation. It orchestrates the construction of many-body states by calling the recursive helper function `mbslist_recursive_iteration!`. The resulting states are partitioned into `HilbertSubspace` objects, each containing all states with a specific total momentum `(K1, K2)`.

# Arguments
- `para::EDPara`: The parameter object containing system details like momentum vectors and component info.
- `N_each_component`: A tuple or vector specifying the number of particles in each conserved component. The sign of the number has a special meaning:
    - **Positive `N`**: Represents `N` particles.
    - **Negative `N`**: Represents `abs(N)` holes in a fully filled band. This is useful for certain calculations like the particle reduced density matrix (see `MomentumED/src/analysis/particle_reduced_density_matrix.jl`).

# Keyword Arguments
- `dict::Bool = false`: If `true`, a dictionary mapping each state to its index is created within each `HilbertSubspace`.
- `index_type::Type = Int64`: The integer type for the dictionary indices.
- `momentum_restriction::Bool = false`: If `true`, only generate subspaces for momenta specified by `k1range`, `k2range`, or `momentum_list`.
- `k1range::Tuple{Int64, Int64}`: The range of `k1` total momentum to generate.
- `k2range::Tuple{Int64, Int64}`: The range of `k2` total momentum to generate.
- `momentum_list::Vector{Tuple{Int64, Int64}}`: A list of specific `(K1, K2)` total momenta to generate.
- `mask::Union{Nothing, Vector{Int64}} = nothing`: Restricts the set of available single-particle orbitals for the first component processed by the recursive algorithm. For positive `N`, only orbitals in the mask are occupied. For negative `N`, holes are created only from orbitals in the mask.

# Returns
- `subspaces::Vector{HilbertSubspace}`: A vector of Hilbert subspaces. Each element contains a list of `MBS64` states belonging to a specific momentum.
- `subspace_k1::Vector{Int64}`: A vector where `subspace_k1[i]` is the `k1` momentum of `subspaces[i]`.
- `subspace_k2::Vector{Int64}`: A vector where `subspace_k2[i]` is the `k2` momentum of `subspaces[i]`.

# Example
```julia
para = EDPara(k_list=[0 1; 0 0], Nc_conserve=2)
# Generate subspaces for a system with 1 particle in the first component and 2 in the second
subspaces, subspace_k1, subspace_k2 = ED_momentum_subspaces(para, (1, 2))
```
"""
function ED_momentum_subspaces(para::EDPara{dim}, N_each_component;
    momentum_restriction::Bool = false, momentum_list::Vector{NTuple{dim, Int64}} = NTuple{dim, Int64}[],
    mask::Union{Nothing, Vector{Int64}} = nothing,
    selection_rule::Function = Returns(true)
    )::Tuple{Vector{HilbertSubspace}, Vector{NTuple{dim, Int64}}} where {dim}

    @assert length(N_each_component) == para.Nc_conserve "The length of number_list must be equal to the number of conserved components $(para.Nc_conserve)"
    Gk = para.Gk

    # Determine momentum ranges
    if momentum_restriction
        # Preprocess momentum list to be within Gk and no duplicates
        for i in eachindex(momentum_list)
            momentum_list[i] = momentum_residue(momentum_list[i], Gk)
        end
        unique!(momentum_list)
    else
        empty!(momentum_list)

        k_range = ntuple(Val(dim)) do d
            if iszero(Gk[d])
                min_kd = minimum(para.k_list[d, :]) * sum(N_each_component)
                max_kd = maximum(para.k_list[d, :]) * sum(N_each_component)
                min_kd:max_kd
            else
                0:Gk[d]-1
            end
        end
        momentum_list = vec(collect(Iterators.product(k_range...)))
    end

    bits = para.Nk * para.Nc
    # to generate approximated chunk division
    n_threads = Threads.nthreads()
    n_outercomponent = binomial(para.Nk * para.Nc_mix, abs(N_each_component[end]))
    n_chunks = n_outercomponent < 2n_threads ? 1 : n_threads
    if !isnothing(mask)
        n_chunks = 1
        sort!(mask)
    end
    chunk_size, chunk_res = divrem(n_outercomponent, n_chunks)
    chunk_ends = [t * chunk_size + min(chunk_res, t) for t in 1:n_chunks]
    chunk_starts = [1; chunk_ends[begin:end-1] .+ 1]
    # multi-thread iteratively generate basis of each chunk
    local_list_of_lists = [[Vector{MBS64{bits}}() for sn in eachindex(momentum_list)] for t in 1:n_chunks]
    Threads.@threads :static for t in 1:n_chunks
        mbslist_recursive_iteration!(
            local_list_of_lists[t], momentum_list, 
            para, collect(N_each_component),
            reinterpret(MBS64{0}, 0),              # accumulated_mbs
            (0, 0),                                # accumulated_momentum
            chunk_starts[t], chunk_ends[t];        # give the chunk start and end only for the outermost iteration
            mask, selection_rule
        )
    end
    # combine chunks
    subspaces = Vector{HilbertSubspace{bits}}(undef, length(momentum_list))
    for sn in eachindex(momentum_list)
        list = Vector{MBS64{bits}}()
        sizehint!(list, sum(t -> length(local_list_of_lists[t][sn]), 1:n_chunks))
        for t in eachindex(local_list_of_lists)
            append!(list, local_list_of_lists[t][sn])
            empty!(local_list_of_lists[t][sn])
        end
        subspaces[sn] = HilbertSubspace(list)
    end

    empty_mask = length.(subspaces) .!= 0
    subspaces = subspaces[empty_mask]
    subspace_k = momentum_list[empty_mask]

    return subspaces, subspace_k
end

function ED_momentum_subspaces(para::EDPara{dim}, N_each_component::Int64;
    momentum_restriction::Bool = false, momentum_list::Vector{NTuple{dim, Int64}} = NTuple{dim, Int64}[],
    mask::Union{Nothing, Vector{Int64}} = nothing,
    selection_rule::Function = Returns(true)
    )::Tuple{Vector{HilbertSubspace}, Vector{NTuple{dim, Int64}}} where {dim}

    return ED_momentum_subspaces(para, (N_each_component, );
        momentum_restriction, momentum_list, mask, selection_rule)

end
