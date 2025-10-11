
"""
This file provides:
Iterators over MBS64 combinations of t electrons in n states (in one component) in sorted order;
Iterators over multi-component MBS64 with given para and elenctron number of each component;
Generating Hilbert subspaces distinguished by total momentum using the previous iterator.
"""

using Combinatorics

#The Combinations iterator in colex order (meaning sorted MBS64 list)
struct ColexMBS64
    n::Int
    t::Int
end

# starting point
@inline function Base.iterate(c::ColexMBS64)
    (MBS64(c.n, 1:c.t), [collect(1:c.t); c.n+1])
end

@inline function Base.iterate(c::ColexMBS64, s)
    if c.t == 0
        return
    end
    for i in 1:c.t
        if s[i] < s[i+1] -1
            s[i] += 1
            for j in 1:i-1
                s[j] = j
            end
            return (MBS64(c.n, view(s, 1:c.t)), s)
        end
    end
    return
end

Base.length(c::ColexMBS64) = binomial(c.n, c.t)

Base.eltype(c::ColexMBS64) = MBS64{c.n}

"""
    mbslist_onecomponent(para::EDPara, N_in_one::Int64)

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

"""
    ED_mbslist(para::EDPara, N_each_component::NTuple{N, Int64}) where {N}

Construct a list of MBS with electron numbers (N1, N2, ...) in each conserved component.

Generates the complete Hilbert space basis by taking the Kronecker product of
single-component bases, ensuring the correct particle number in each conserved
component.

# Arguments
- `para::EDPara`: Parameter structure containing momentum and component information
- `N_each_component::NTuple{N, Int64}`: Tuple of particle numbers for each conserved component

# Returns
- `Vector{MBS64}`: Complete basis of MBS64 states with specified particle distribution

# Example
```julia
para = EDPara(k_list=[0 1; 0 0], Nc_hopping=1, Nc_conserve=2)
states = ED_mbslist(para, (1, 1))  # 1 particle in each of 2 conserved components
```

# Notes
Uses Kronecker product to combine states from different conserved components.
The total dimension is the product of individual component dimensions.
"""
# function ED_mbslist(para::EDPara, N_each_component::NTuple{N, Int64}) where {N}
#     @assert N == para.Nc_conserve "The length of number_list must be equal to the number of conserved components $(para.Nc_conserve)"
#     list = ED_mbslist_onecomponent(para, N_each_component[begin])
#     for i in eachindex(N_each_component)[2:end]
#         list = kron(ED_mbslist_onecomponent(para, N_each_component[i]), list)
#     end
#     return list
# end



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
    MBS_totalmomentum(para::EDPara, i_list::Int64...)

Calculate the total momentum (K1, K2) from a list of occupied orbital indices.
"""
function MBS_totalmomentum(para::EDPara, i_list::Int64...)
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





function mbslist_recurusive_iteration!(subspaces::Vector{HilbertSubspace{bits}}, 
    subspace_k1::Vector{Int64}, subspace_k2::Vector{Int64}, 
    para::EDPara, N_each_component::Vector{Int64}, 
    accumulated_mbs::MBS64 = reinterpret(MBS64{0}, 0), 
    accumulated_momentum::Tuple{Int64, Int64} = (0, 0); print::Bool = false
) where {bits}

    if !isempty(N_each_component) 
        print && println("Enter loop with $N_each_component. 
        Momentum $(accumulated_momentum[1]) $(accumulated_momentum[2])\n\t $accumulated_mbs\n")
        for mbs_smaller in mbslist_onecomponent(para, N_each_component[end])
            print && println("generated mbs in the new component:\n$mbs_smaller")
            new_momentum = MBS_totalmomentum(para, mbs_smaller)
            mbslist_recurusive_iteration!(subspaces, subspace_k1, subspace_k2, para, 
                N_each_component[begin:end-1],          # remaining components
                accumulated_mbs * mbs_smaller,          # updated MBS64
                accumulated_momentum .+ new_momentum    # updated momentum 
            )
        end
    else
        k1, k2 = accumulated_momentum
        Gk = para.Gk
        iszero(Gk[1]) || (k1 = mod(k1, Gk[1]))
        iszero(Gk[2]) || (k2 = mod(k2, Gk[2]))
        print && println("finally: momentum $k1, $k2\n$accumulated_mbs")
        index = findfirst((subspace_k1 .== k1) .& (subspace_k2 .== k2))
        if index !== nothing
            push!(subspaces[index].list, accumulated_mbs)
        end
    end
end

"""
    ED_momentum_block_division(para::EDPara, mbs_list::Vector{MBS64{bits}};
        momentum_restriction = false, k1range=(-2,2), k2range=(-2,2),
        momentum_list::Vector{Tuple{Int64, Int64}} = Vector{Tuple{Int64, Int64}}(),
        ) where {bits}

Divide a list of MBS64 states into momentum blocks based on total momentum.

This function takes a complete basis of MBS64 states and divides them into blocks
where each block contains states with the same total (k1, k2) momentum. This enables
momentum-conserved diagonalization by processing each block independently.

Duplicates, up to mod Gk, in the suggested momentum_list will be removed.

# Arguments
- `para::EDPara`: Parameter structure containing momentum and component information
- `mbs_list::Vector{MBS64{bits}}`: Complete basis of MBS64 states
- `momentum_restriction::Bool`: Whether to restrict momentum range (default: false)
- `k1range::Tuple{Int64, Int64}`: Range for k1 momentum (default: (-2,2))
- `k2range::Tuple{Int64, Int64}`: Range for k2 momentum (default: (-2,2))
- `momentum_list::Vector{Tuple{Int64, Int64}}`: Specific momenta to include (default: all)

# Returns
- `blocks::Vector{Vector{MBS64{bits}}}`: List of momentum blocks, each containing states with same momentum
- `block_k1::Vector{Int64}`: k1 momentum values for each block
- `block_k2::Vector{Int64}`: k2 momentum values for each block
- `zero_block_index::Int64`: Index of k1=k2=0 block (0 if not exists)

# Example
```julia
para = EDPPara(k_list=[0 1; 0 0], Nc_hopping=1, Nc_conserve=1)
states = ED_mbslist(para, (2,))  # Generate basis
blocks, k1_list, k2_list, zero_idx = ED_momentum_block_division(para, states)
```

# Notes
This function is essential for momentum-conserved exact diagonalization, as it enables
parallel processing of different momentum sectors independently.
"""
function ED_momentum_subspaces(para::EDPara, N_each_component::Union{Vector{Int64}, Tuple{Vararg{Int64}}};
    dict::Bool = false, index_type::Type = Int64,  momentum_restriction::Bool = false, 
    k1range::Tuple{Int64, Int64} = (-2,2), k2range::Tuple{Int64, Int64} = (-2,2),
    momentum_list::Vector{Tuple{Int64, Int64}} = Tuple{Int64, Int64}[],
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
            k1min, k1max = extrema(k_list[1,:]) .* sum(N_each_component)
        end
        if !iszero(Gk[2])
            k2min, k2max = (0, Gk[2]-1)
        else
            k2min, k2max = extrema(k_list[2,:]) .* sum(N_each_component)
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
    subspaces = [HilbertSubspace(MBS64{bits}[]; index_type) 
        for _ in eachindex(subspace_k1)
    ]
    mbslist_recurusive_iteration!(
        subspaces, subspace_k1, subspace_k2, 
        para, collect(N_each_component)
    )

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
