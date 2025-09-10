"""
mbslist.jl - Many-Body State List Generation

This module provides functions for generating lists of MBS64 states with
specified particle numbers per conserved component, organized for efficient
momentum-block diagonalization.
"""

using Combinatorics

"""
    ED_mbslist_onecomponent(para::EDPara, N_in_one::Int64)

Construct the MBS list of N electrons in one conserved component.

Generates all possible MBS64 states with exactly N_in_one particles in the
specified component, using combinations of orbital indices.

# Arguments
- `para::EDPara`: Parameter structure containing momentum and component information
- `N_in_one::Int64`: Number of particles in the conserved component

# Returns
- `Vector{MBS64}`: Sorted list of MBS64 states with specified particle number

# Example
```julia
para = EDPara(k_list=[0 1; 0 0], Nc_hopping=1, Nc_conserve=1)
states = ED_mbslist_onecomponent(para, 2)  # 2 particles in 2 orbitals
```
"""
function ED_mbslist_onecomponent(para::EDPara, N_in_one::Int64)
    Nstate = para.Nk * para.Nc_hopping
    @assert 0 <= N_in_one <= Nstate "Invalid number of electrons in one component"
    sort([MBS64(Nstate, combi...) for combi in collect(combinations(1:Nstate, N_in_one)) ])
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
function ED_mbslist(para::EDPara, N_each_component::NTuple{N, Int64}) where {N}
    @assert N == para.Nc_conserve "The length of number_list must be equal to the number of conserved components $(para.Nc_conserve)"
    list = ED_mbslist_onecomponent(para, N_each_component[begin])
    for i in eachindex(N_each_component)[2:end]
        list = kron(ED_mbslist_onecomponent(para, N_each_component[i]), list)
    end
    return list
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
function ED_momentum_block_division(para::EDPara, mbs_list::Vector{MBS64{bits}};
    momentum_restriction = false, k1range=(-2,2), k2range=(-2,2),
    momentum_list::Vector{Tuple{Int64, Int64}} = Vector{Tuple{Int64, Int64}}(),
    ) where {bits}

    # Calculate momentum for each MBS
    TotalMomentum = Matrix{Int64}(undef, 2, length(mbs_list))
    k1_list = Vector{Int64}(undef, length(mbs_list))
    k2_list = Vector{Int64}(undef, length(mbs_list))
    
    for (idx, mbs) in enumerate(mbs_list)
        k1, k2 = MBS64_totalmomentum(para, mbs)
        TotalMomentum[1, idx] = k1
        TotalMomentum[2, idx] = k2
        k1_list[idx] = k1
        k2_list[idx] = k2
    end

    blocks = typeof(mbs_list)[]
    block_k1 = Int64[]
    block_k2 = Int64[]
    
    # Determine momentum ranges
    if isempty(mbs_list)
        return blocks, block_k1, block_k2, 0
    end
    
    k1min, k2min = minimum(TotalMomentum, dims=2)
    k1max, k2max = maximum(TotalMomentum, dims=2)

    if momentum_restriction
        k1min = max(k1range[1], k1min)
        k2min = max(k2range[1], k2min)
        k1max = min(k1range[2], k1max)
        k2max = min(k2range[2], k2max)
    end

    # Group by momentum
    if isempty(momentum_list)
        for K1 in k1min:k1max, K2 in k2min:k2max
            mask = findall((k1_list .== K1) .& (k2_list .== K2))
            if !iszero(length(mask))
                push!(blocks, mbs_list[mask])
                push!(block_k1, K1)
                push!(block_k2, K2)
            end
        end
    else
        for K1 in k1min:k1max, K2 in k2min:k2max
            if (K1, K2) ∈ momentum_list
                mask = findall((k1_list .== K1) .& (k2_list .== K2))
                if !iszero(length(mask))
                    push!(blocks, mbs_list[mask])
                    push!(block_k1, K1)
                    push!(block_k2, K2)
                end
            end
        end
    end
    
    # Find k=0 block index
    block_num_0 = findfirst(eachindex(block_k1)) do bn
        block_k1[bn] == 0 && block_k2[bn] == 0
    end
    
    return blocks, block_k1, block_k2, something(block_num_0, 0)
end
