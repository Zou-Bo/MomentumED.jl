"""
    ED_sortedScatterList_onebody(para::EDPara) -> Vector{Scatter{1}}

Generate sorted lists of one-body Scatter terms from the parameters.

Extracts one-body terms from EDpara.H_onebody for multi-component systems and converts
them to Scatter terms with proper normal ordering.

# Arguments
- `para::EDPara`: Parameter structure containing system configuration

# Returns
- `Vector{Scatter{1}}`: Sorted list of one-body Scatter terms

# Details
- Maps component indices to global orbital indices using: `global_index = k + Nk * (ch - 1) + Nk * Nch * (cc - 1)`
- Applies normal ordering to avoid double-counting
- Uses `sort_merge_scatlist` to eliminate duplicates and sort terms
- Only includes non-zero amplitude terms

# Example
```julia
para = EDPara(k_list=k_list, Gk=(3, 5), V_int=V_int)
Scatter1 = ED_sortedScatterList_onebody(para)
```
"""
function ED_sortedScatterList_onebody(para::EDPara)::Vector{Scatter{1}}
    sct_list1 = Vector{Scatter{1}}()
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

            # Create Scatter term with normal ordering
            i_in >= i_ot && push!(sct_list1, NormalScatter(V, i_ot, i_in; upper_hermitian = true))
        end
    end
    
    return sort_merge_scatlist(sct_list1)
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
- Uses `MBS_totalmomentum(para, i, j)` to compute total momentum for each pair
- Essential for efficient two-body Scatter term generation
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
        K_total = MBS_totalmomentum(para, i, j)
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

global PRINT_TWOBODY_SCATTER_PAIRS::Bool = false

"""
    scat_pair_group_coordinate(pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara;
                   kshift::Tuple{Float64, Float64} = (0.0, 0.0), output::Bool = false) -> Vector{Scatter{2}}

Generate all Scatter terms between momentum pairs with the same total momentum. 
Use the interaction function that accepts coordinate format momenta.
Allow input of momentum shift for twisted boundary conditions.

Creates two-body Scatter terms for all possible transitions between momentum pairs
that conserve total momentum, including all component index combinations.

# Arguments
- `pair_group::Vector{Tuple{Int64,Int64}}`: List of momentum index pairs with same total momentum
- `para::EDPara`: Parameter structure containing system configuration

# Keywords
- `kshift::Tuple{Float64, Float64}=(0.0, 0.0)`: Momentum shift for twisted boundary conditions

# Returns
- `Vector{Scatter{2}}`: List of two-body Scatter terms for this momentum group

# Details
- Iterates over all input/output momentum pair combinations within the group
- Generates all component index combinations for each momentum pair
- Maps momentum and component indices to global orbital indices
- Applies normal ordering: `minmax(i1, i2) >= minmax(f1, f2)`
- Calculates Scatter amplitudes using `int_amp` function with momentum shift
- Includes both direct and exchange contributions
- Handles identical orbital pairs with proper exclusion

# Physics
The Scatter amplitude includes:
- Direct term: `int_amp(i1, i2, f1, f2, para; kshift=kshift)`
- Exchange term: `int_amp(i2, i1, f1, f2, para; kshift=kshift)`
- Total amplitude: `amp = amp_direct - amp_exchange`

# Example
```julia
# Get momentum pairs with total momentum (0, 0)
groups = group_momentum_pairs(para)
zero_momentum_pairs = groups[(0, 0)]
# Generate all Scatter terms for this momentum group
Scatter_terms = scat_pair_group(zero_momentum_pairs, para)
```
"""
function scat_pair_group_coordinate(pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara;
    shifts::Matrix{T})::Vector{Scatter{2}} where {T <: Real}
    
    # @assert size(shifts) == (2, para.Nc_conserve)

    Nc = para.Nc
    Nk = para.Nk
    Gk1, Gk2 = para.Gk
    sys_size = (Gk1 != 0 && Gk2 != 0) ? Nk : 1

    if T <: Integer || T <: Rational
        klist = para.k_list // 1
        kshift = shifts // 1
        if Gk1 != 0
            klist[1, :] .//= Gk1
            kshift[1, :] .//= Gk1
        end
        if Gk2 != 0
            klist[2, :] .//= Gk2
            kshift[2, :] .//= Gk2
        end
    else
        klist = copy(para.k_list)
        kshift = copy(shifts)
        if Gk1 != 0
            klist[1, :] ./= Gk1
            kshift[1, :] ./= Gk1
        end
        if Gk2 != 0
            klist[2, :] ./= Gk2
            kshift[2, :] ./= Gk2
        end
    end



    Scatter_list = Vector{Scatter{2}}()
    # Iterate over all input and output pairs
    for (ki1, ki2) in pair_group, (kf1, kf2) in pair_group
        PRINT_TWOBODY_SCATTER_PAIRS && println()
        PRINT_TWOBODY_SCATTER_PAIRS && println("ki1, ki2, kf1, kf2 = ($ki1, $ki2), ($kf1, $kf2)")
        # Generate all component index combinations
        for ci1 in 1:Nc, ci2 in 1:Nc, cf1 in 1:Nc, cf2 in 1:Nc
            
            # Map to global orbital indices
            # Global index = momentum_index + Nk * (component_index - 1)
            f1 = kf1 + Nk * (cf1 - 1)
            f2 = kf2 + Nk * (cf2 - 1)
            i1 = ki1 + Nk * (ci1 - 1)
            i2 = ki2 + Nk * (ci2 - 1)

            # no duplicate input/output indices
            if i1 == i2 || f1 == f2
                continue
            end

            if ki1 == ki2 && i1 < i2
                continue
            end

            if kf1 == kf2 && f1 < f2
                continue
            end

            # conjugate Scatter only need to count onece, as the Hamiltonian is generated with upper half Hermitian()
            if minmax(i1, i2) >= minmax(f1, f2)

                # Calculate the direct and exchange amplitudes
                PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i1, Nk), fldmod1(i2, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk),"        ")
                amp_direct = para.V_int(
                    (klist[1, kf1] + kshift[1, cf1], klist[2, kf1] + kshift[2, cf1]),
                    (klist[1, kf2] + kshift[1, cf2], klist[2, kf2] + kshift[2, cf2]),
                    (klist[1, ki2] + kshift[1, ci2], klist[2, ki2] + kshift[2, ci2]),
                    (klist[1, ki1] + kshift[1, ci1], klist[2, ki1] + kshift[2, ci1]),
                    cf1, cf2, ci2, ci1
                )

                # exchange i1 and i2
                PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i2, Nk), fldmod1(i1, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk))
                amp_exchange = para.V_int(
                    (klist[1, kf1] + kshift[1, cf1], klist[2, kf1] + kshift[2, cf1]),
                    (klist[1, kf2] + kshift[1, cf2], klist[2, kf2] + kshift[2, cf2]),
                    (klist[1, ki1] + kshift[1, ci1], klist[2, ki1] + kshift[2, ci1]),
                    (klist[1, ki2] + kshift[1, ci2], klist[2, ki2] + kshift[2, ci2]),
                    cf1, cf2, ci1, ci2
                )

                amp = (amp_direct - amp_exchange) / sys_size
                iszero(amp) || push!(Scatter_list, NormalScatter(amp, f1, f2, i2, i1; upper_hermitian = true))
                PRINT_TWOBODY_SCATTER_PAIRS && println()
            end
        
        end
    end
    
    return Scatter_list
end

"""
    scat_pair_group_index(pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara;
                   output::Bool = false) -> Vector{Scatter{2}}

Generate all Scatter terms between momentum pairs with the same total momentum. 
Use the interaction function that accepts indices of momenta in `para.k_list`.
No input of momentum shift for twisted boundary conditions; update the `para` instead.

Creates two-body Scatter terms for all possible transitions between momentum pairs
that conserve total momentum, including all component index combinations.

# Arguments
- `pair_group::Vector{Tuple{Int64,Int64}}`: List of momentum index pairs with same total momentum
- `para::EDPara`: Parameter structure containing system configuration

# Keywords
- `output::Bool=false`: Print all the Scatter terms (before normal ordering) for debugging purposes

# Returns
- `Vector{Scatter{2}}`: List of two-body Scatter terms for this momentum group

# Details
- Iterates over all input/output momentum pair combinations within the group
- Generates all component index combinations for each momentum pair
- Maps momentum and component indices to global orbital indices
- Applies normal ordering: `minmax(i1, i2) >= minmax(f1, f2)`
- Calculates Scatter amplitudes using `int_amp` function with momentum shift
- Includes both direct and exchange contributions
- Handles identical orbital pairs with proper exclusion

# Physics
The Scatter amplitude includes:
- Direct term: `int_amp(i1, i2, f1, f2, para; kshift=kshift)`
- Exchange term: `int_amp(i2, i1, f1, f2, para; kshift=kshift)`
- Total amplitude: `amp = amp_direct - amp_exchange`

# Example
```julia
# Get momentum pairs with total momentum (0, 0)
groups = group_momentum_pairs(para)
zero_momentum_pairs = groups[(0, 0)]
# Generate all Scatter terms for this momentum group
Scatter_terms = scat_pair_group(zero_momentum_pairs, para)
```
"""
function scat_pair_group_index(pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara;
    )::Vector{Scatter{2}}
    
    Nc = para.Nc
    Nk = para.Nk
    Gk1, Gk2 = para.Gk
    sys_size = (Gk1 != 0 && Gk2 != 0) ? Nk : 1


    Scatter_list = Vector{Scatter{2}}()
    # Iterate over all input and output pairs
    for (ki1, ki2) in pair_group, (kf1, kf2) in pair_group
        PRINT_TWOBODY_SCATTER_PAIRS && println()
        PRINT_TWOBODY_SCATTER_PAIRS && println("ki1, ki2, kf1, kf2 = ($ki1, $ki2), ($kf1, $kf2)")
        # Generate all component index combinations
        for ci1 in 1:Nc, ci2 in 1:Nc, cf1 in 1:Nc, cf2 in 1:Nc
            
            # Map to global orbital indices
            # Global index = momentum_index + Nk * (component_index - 1)
            f1 = kf1 + Nk * (cf1 - 1)
            f2 = kf2 + Nk * (cf2 - 1)
            i1 = ki1 + Nk * (ci1 - 1)
            i2 = ki2 + Nk * (ci2 - 1)

            # no duplicate input/output indices
            if i1 == i2 || f1 == f2
                continue
            end

            if ki1 == ki2 && i1 < i2
                continue
            end

            if kf1 == kf2 && f1 < f2
                continue
            end

            # inverse Scatter only need to count onece, as the Hamiltonian is generated with upper half Hermitian()
            if minmax(i1, i2) >= minmax(f1, f2)

                # Calculate the direct and exchange amplitudes
                PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i1, Nk), fldmod1(i2, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk),"        ")
                amp_direct = para.V_int(
                    kf1, kf2, ki2, ki1,
                    cf1, cf2, ci2, ci1
                )

                # exchange i1 and i2
                PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i2, Nk), fldmod1(i1, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk))
                amp_exchange = para.V_int(
                    kf1, kf2, ki1, ki2,
                    cf1, cf2, ci1, ci2
                )

                amp = (amp_direct - amp_exchange) / sys_size
                iszero(amp) || push!(Scatter_list, NormalScatter(amp, f1, f2, i2, i1; upper_hermitian = true))
                PRINT_TWOBODY_SCATTER_PAIRS && println()
            end
        
        end
    end
    
    return Scatter_list
end

"""
    ED_sortedScatterList_twobody(para::EDPara; kshift::Tuple{Float64, Float64} = (0.0, 0.0)) -> Vector{Scatter{2}}

Generate sorted lists of two-body Scatter terms from the parameters.

Uses the interaction function from EDPara.V_int to calculate Scatter amplitudes
for all possible two-body processes, grouped by total momentum conservation.
It recognizes which momentum input format is accepted by `para.V_int` function.
If the momentum index is used, no input of momentum shift for twisted boundary conditions is allowed.

# Arguments
- `para::EDPara`: Parameter structure containing system configuration

# Keywords
- `kshift=nothing`: Momentum shift for twisted boundary conditions

# Returns
- `Vector{Scatter{2}}`: Sorted list of two-body Scatter terms

# Details
- Groups momentum pairs by total momentum for efficiency
- Generates all component index combinations for each momentum pair
- Applies normal ordering: `minmax(i1, i2) >= minmax(f1, f2)`
- Includes both direct and exchange contributions: `amp = amp_direct - amp_exchange`
- Uses momentum shift in interaction calculations: `(k_list .+ kshift) ./ Gk`
- Applies `sort_merge_scatlist` to eliminate duplicates and sort terms

# Example
```julia
para = EDPara(k_list=k_list, Gk=(3, 5), V_int=V_int)
# Without momentum shift
Scatter2 = ED_sortedScatterList_twobody(para)
# With twisted boundary conditions
Scatter2_shifted = ED_sortedScatterList_twobody(para; kshift=(0.1, 0.1))
```
"""
function ED_sortedScatterList_twobody(para::EDPara; 
    kshift = nothing)::Vector{Scatter{2}}

    momentum_groups = group_momentum_pairs(para)
    
    sct_list2 = Vector{Scatter{2}}()
    if para.momentum_coordinate

        if isnothing(kshift)
            shifts = zeros(Int64, 2, para.Nc_conserve)
        elseif kshift isa Tuple
            shifts = Matrix{eltype(kshift)}(undef, 2, para.Nc_conserve)
            shifts[1,:] .= kshift[1]
            shifts[2,:] .= kshift[2]
        elseif kshift isa Vector
            @assert length(kshift) == para.Nc_conserve "length of kshift isn't equal to para.Nc_conserve."
            shifts = Matrix{eltype(kshift[1])}(undef, 2, para.Nc_conserve)
            for i in 1:para.Nc_conserve
                shifts[:,i] .= kshift[i]
            end
        elseif kshift isa Matrix
            @assert size(kshift) == (2, para.Nc_conserve) "size of kshift isn't equal to para.Nc_conserve."
            shifts = kshift
        else
            throw(AssertionError("kshift could be nothing, Tuple{R,R}, Vector{Tuple{R,R}}, or Matrix{R}, where R<:Real."))
        end

        for (K_total, pairs) in momentum_groups
            append!(sct_list2, scat_pair_group_coordinate(pairs, para; shifts=shifts))
        end
    else
        if !isnothing(kshift)
            @warn "kshift is ignored when para.V_int accepts momentum indices instead of coordinates."
        end

        for (K_total, pairs) in momentum_groups
            append!(sct_list2, scat_pair_group_index(pairs, para))
        end
    end

    
    return sort_merge_scatlist(sct_list2)
end
