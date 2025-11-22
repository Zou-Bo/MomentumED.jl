"""
This file provides:
Iterators over MBS64 combinations of `t` electrons in `n` states (in one component) in sorted order;
Recursive iteration function over multi-component MBS64 with given para and electron number of each component;
Generating Hilbert subspaces distinguished by total momentum using the previous function.
"""


"""
    mbslist_onecomponent(para::EDPara, N_in_one::Int64[, mask])

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
para = EDPara(k_list=[0 1; 0 0], Nc_hopping=1, Nc_conserve=1)
states = ED_mbslist_onecomponent(para, 2)  # 2 particles in 2 orbitals
```
"""
function mbslist_onecomponent(para::EDPara, N_in_one::Int64)
    Nstate = para.Nk * para.Nc_hopping
    @assert 0 <= N_in_one <= Nstate "Invalid number of electrons in one component"
    return ColexMBS64(Nstate, N_in_one)
end
function mbslist_onecomponent(para::EDPara, N_in_one::Int64, mask::Union{Nothing, Vector{Int64}})
    isnothing(mask) && return mbslist_onecomponent(para, N_in_one)
    Nstate = para.Nk * para.Nc_hopping
    @assert 0 <= N_in_one <= Nstate "Invalid number of electrons in one component"
    @assert isempty(mask) || 1 <= minimum(mask) && maximum(mask) <= Nstate "Invalid orbital index in mask for one component, mask=$mask, Nstate=$Nstate"
    return ColexMBS64Mask(Nstate, N_in_one, mask)
end

"""
    MBS_totalmomentum(para::EDPara, mbs::MBS64)

Calculate the total momentum (K1, K2) of a many-body state.
The momentum is mod G if G is nonzero (from para.Gk).
"""
function MBS_totalmomentum(k_list::Matrix{Int64}, Gk::NTuple{2, Int64}, mbs::MBS64;
    Nk = size(k_list, 2))::NTuple{2, Int64}
    # momentum are integers
    k1 = 0; k2 = 0
    for i in occ_list(mbs)
        momentum = view(k_list, 1:2, mod1(i, Nk) )
        k1 += momentum[1]
        k2 += momentum[2]
    end
    iszero(Gk[1]) || (k1 = mod(k1, Gk[1]))
    iszero(Gk[2]) || (k2 = mod(k2, Gk[2]))
    return k1, k2
end
MBS_totalmomentum(para::EDPara, mbs::MBS64) = MBS_totalmomentum(para.k_list, para.Gk, mbs; Nk = para.Nk)

"""
    MBS_totalmomentum(para::EDPara, i_list::Tuple{Vararg{Int64}})

Calculate the total momentum (K1, K2) from a list of occupied orbital indices.
`i_list` can contain repeated numbers,
and the momentum of that orbital will be added multiple times.
"""
function MBS_totalmomentum(para::EDPara, i_list::Tuple{Vararg{Int64}})::NTuple{2, Int64}
    # momentum are integers
    k1 = 0; k2 = 0; Gk = para.Gk
    for i in i_list
        momentum = @view para.k_list[:, mod1(i, para.Nk)]
        k1 += momentum[1]
        k2 += momentum[2]
    end
    iszero(Gk[1]) || (k1 = mod(k1, Gk[1]))
    iszero(Gk[2]) || (k2 = mod(k2, Gk[2]))
    return k1, k2
end

"""
    mbslist_recursive_iteration!(subspaces, subspace_k1, subspace_k2, para, N_each_component, accumulated_mbs, accumulated_momentum; mask=nothing)

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
`mask` is a Vector{Int64} and must be sorted.

`check_mask_sorted::Bool=false` will assert `mask` is sorted if it's provided. 
It will only check once at the begining of outermost iteration.
"""
function mbslist_recursive_iteration!(subspaces::Vector{HilbertSubspace{bits}}, 
    subspace_k1::Vector{Int64}, subspace_k2::Vector{Int64}, 
    para::EDPara, N_each_component::Vector{Int64}, 
    accumulated_mbs::MBS64 = reinterpret(MBS64{0}, 0), 
    accumulated_momentum::Tuple{Int64, Int64} = (0, 0); 
    mask::Union{Nothing, Vector{Int64}} = nothing,
    check_mask_sorted::Bool = false,
    selection_rule::Function = Returns(true),
) where {bits}

    if check_mask_sorted && !isnothing(mask)
        @assert issorted(mask)
    end

    if !isempty(N_each_component) 
        PRINT_RECURSIVE_MOMENTUM_DIVISION && println("Enter loop with $N_each_component. 
        Momentum $(accumulated_momentum[1]) $(accumulated_momentum[2])\n\t$accumulated_mbs\n")
        if !isnothing(mask)
            num_component_orbitals = para.Nk * para.Nc_hopping* (length(N_each_component) - 1)
            i = searchsortedlast(mask, num_component_orbitals)
            mask_this_component = mask[i+1:end] .- num_component_orbitals
            for mbs_smaller in mbslist_onecomponent(para, abs(N_each_component[end]), mask_this_component)
                PRINT_RECURSIVE_MOMENTUM_DIVISION && println("generated mbs in the new component:\n$mbs_smaller")
                new_momentum = sign(N_each_component[end]) .* MBS_totalmomentum(para, mbs_smaller)
                mbslist_recursive_iteration!(subspaces, subspace_k1, subspace_k2, para, 
                    N_each_component[begin:end-1],          # remaining components
                    accumulated_mbs * mbs_smaller,          # updated MBS64
                    accumulated_momentum .+ new_momentum;   # updated momentum 
                    mask = mask[begin:i], 
                    selection_rule
                )
            end
        else
            for mbs_smaller in mbslist_onecomponent(para, abs(N_each_component[end]))
                PRINT_RECURSIVE_MOMENTUM_DIVISION && println("generated mbs in the new component:\n$mbs_smaller")
                new_momentum = sign(N_each_component[end]) .* MBS_totalmomentum(para, mbs_smaller)
                mbslist_recursive_iteration!(subspaces, subspace_k1, subspace_k2, para, 
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
            k1, k2 = accumulated_momentum
            Gk = para.Gk
            iszero(Gk[1]) || (k1 = mod(k1, Gk[1]))
            iszero(Gk[2]) || (k2 = mod(k2, Gk[2]))
            PRINT_RECURSIVE_MOMENTUM_DIVISION && println("\tfinally: momentum $k1, $k2\t$accumulated_mbs")
            index = findfirst((subspace_k1 .== k1) .& (subspace_k2 .== k2))
            if !isnothing(index)
                push!(subspaces[index].list, accumulated_mbs)
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
function ED_momentum_subspaces(para::EDPara, N_each_component;
    dict::Bool = false, index_type::Type = Int64,  momentum_restriction::Bool = false, 
    k1range::Tuple{Int64, Int64} = (-2,2), k2range::Tuple{Int64, Int64} = (-2,2),
    momentum_list::Vector{Tuple{Int64, Int64}} = Tuple{Int64, Int64}[],
    mask::Union{Nothing, Vector{Int64}} = nothing,
    )::Tuple{Vector{HilbertSubspace}, Vector{Int64}, Vector{Int64}}

    @assert length(N_each_component) == para.Nc_conserve "The length of number_list must be equal to the number of conserved components $(para.Nc_conserve)"

    Gk = para.Gk

    # Preprocess momentum list to be within Gk and no duplicates
    for i in eachindex(momentum_list)
        k1, k2 = momentum_list[i]
        iszero(Gk[1]) || (k1 = mod(k1, Gk[1]))
        iszero(Gk[2]) || (k2 = mod(k2, Gk[2]))
        momentum_list[i] = (k1, k2)
    end
    unique!(momentum_list)

    # Determine momentum ranges
    if momentum_restriction
        k1min, k1max = minmax(k1range...)
        k2min, k2max = minmax(k2range...)
        # Adjust k1range and k2range if they are too large
        if !iszero(Gk[1]) && k1max - k1min + 1 > Gk[1]
            k1min, k1max = (0, Gk[1]-1)
        end
        if !iszero(Gk[2]) && k2max - k2min + 1 > Gk[2]
            k2min, k2max = (0, Gk[2]-1)
        end
    else
        if !iszero(Gk[1])
            k1min, k1max = (0, Gk[1]-1)
        else
            k1min, k1max = extrema(para.k_list[1,:]) .* sum(N_each_component)
        end
        if !iszero(Gk[2])
            k2min, k2max = (0, Gk[2]-1)
        else
            k2min, k2max = extrema(para.k_list[2,:]) .* sum(N_each_component)
        end
    end


    # make momentum lists first
    subspace_k1 = Int64[];
    subspace_k2 = Int64[];
    for K1 in k1min:k1max, K2 in k2min:k2max
        iszero(Gk[1]) || (K1 = mod(K1, Gk[1]))
        iszero(Gk[2]) || (K2 = mod(K2, Gk[2]))
        if isempty(momentum_list) || (K1, K2) ∈ momentum_list
            push!(subspace_k1, K1)
            push!(subspace_k2, K2)
        end
    end

    bits = para.Nk * para.Nc
    subspaces = [HilbertSubspace(MBS64{bits}[]; index_type) for _ in eachindex(subspace_k1)]
    mbslist_recursive_iteration!(
        subspaces, subspace_k1, subspace_k2, 
        para, collect(N_each_component);
        mask = isnothing(mask) ? nothing : sort!(mask)
    )

    empty_mask = length.(subspaces) .!= 0
    subspaces = subspaces[empty_mask]
    subspace_k1 = subspace_k1[empty_mask]
    subspace_k2 = subspace_k2[empty_mask]

    if dict
        for i in eachindex(subspaces)
            make_dict!(subspaces[i]; index_type) 
        end
    end
    
    # # Find k=0 block index
    # block_num_0 = findfirst(eachindex(block_k1)) do bn
    #     block_k1[bn] == 0 && block_k2[bn] == 0
    # end
    
    return subspaces, subspace_k1, subspace_k2 #, something(block_num_0, 0)
end
