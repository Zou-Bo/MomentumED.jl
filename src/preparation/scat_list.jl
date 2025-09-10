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
- Uses `MBS_totalmomentum(para, i, j)` to compute total momentum for each pair
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

"""
    scat_pair_group(pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara;
                   kshift::Tuple{Float64, Float64} = (0.0, 0.0), output::Bool = false) -> Vector{Scattering{2}}

Generate all scattering terms between momentum pairs with the same total momentum.

Creates two-body scattering terms for all possible transitions between momentum pairs
that conserve total momentum, including all component index combinations.

# Arguments
- `pair_group::Vector{Tuple{Int64,Int64}}`: List of momentum index pairs with same total momentum
- `para::EDPara`: Parameter structure containing system configuration

# Keywords
- `kshift::Tuple{Float64, Float64}=(0.0, 0.0)`: Momentum shift for twisted boundary conditions
- `output::Bool=false`: Print all the scattering terms (before normal ordering) for debugging purposes

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
    
    Nc = para.Nc
    Nk = para.Nk
    Gk1, Gk2 = para.Gk
    frac_klist = similar(para.k_list, Float64)
    sys_size = 1
    if Gk1 != 0
        frac_klist[1, :] = para.k_list[1, :] ./ Gk1 .+ kshift[1]
        sys_size *= Gk1
    end
    if Gk2 != 0
        frac_klist[2, :] = para.k_list[2, :] ./ Gk2 .+ kshift[2]
        sys_size *= Gk2
    end

    scattering_list = Vector{Scattering{2}}()
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

            # inverse scattering only need to count onece, as the Hamiltonian is generated with upper half Hermitian()
            if minmax(i1, i2) >= minmax(f1, f2)

                # Calculate the direct and exchange amplitudes
                output && print(fldmod1(i1, Nk), fldmod1(i2, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk),"        ")
                amp_direct = para.V_int(
                    (frac_klist[1, kf1], frac_klist[2, kf1]),
                    (frac_klist[1, kf2], frac_klist[2, kf2]),
                    (frac_klist[1, ki2], frac_klist[2, ki2]),
                    (frac_klist[1, ki1], frac_klist[2, ki1]),
                    cf1, cf2, ci2, ci1
                )

                # exchange i1 and i2
                output && print(fldmod1(i2, Nk), fldmod1(i1, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk))
                amp_exchange = para.V_int(
                    (frac_klist[1, kf1], frac_klist[2, kf1]),
                    (frac_klist[1, kf2], frac_klist[2, kf2]),
                    (frac_klist[1, ki1], frac_klist[2, ki1]),
                    (frac_klist[1, ki2], frac_klist[2, ki2]),
                    cf1, cf2, ci1, ci2
                )

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
