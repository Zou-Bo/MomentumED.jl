# todo list
# function to transform momentum
# update docstring
# allow onebody function to have keyword arguments and momentum coordinate input

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
function ED_sortedScatterList_onebody(para::EDPara; MBS_type::Type{<:MBS64} = MBS64{para.Nc*para.Nk},
    element_type::Type{F} = Float64 )::Vector{Scatter{Complex{F}, MBS_type}} where{F <: AbstractFloat}

    Nk = para.Nk
    Nc = para.Nc
    bits = Nc * Nk
    @assert MBS_type == MBS64{bits} "MBS bits has to be para.Nc*para.Nk."
    
    sct_list1 = Vector{Scatter{Complex{F}, MBS_type}}()
    # Extract one-body terms from H_one(kf, ki, cf, ci) and convert to Scatter terms
    if para.one_momentum_coordinate
        for ci in 1:Nc, cf in 1:Nc, ki in 1:Nk, kf in 1:Nk
            V = para.H_one(kf, ki, cf, ci) |> Complex{F}
            if !iszero(V)
                # Map component indices to global orbital indices
                i_out = kf + Nk * (cf - 1)  # output orbital
                i_in = ki + Nk * (ci - 1)  # input orbital

                # Create Scatter term with normal ordering
                i_in >= i_out && push!(sct_list1, Scatter(V, i_out, i_in; bits, upper_hermitian = true))
            end
        end
    else
        error("H_one currently only supports momentum coordinate.")
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
function group_momentum_pairs(para::EDPara;
    momentum_transformation::Union{Nothing, Function} = nothing)
    
    # Dictionary to store momentum groups
    momentum_groups = Dict{Tuple{Int64,Int64}, Vector{Tuple{Int64,Int64}}}()
    
    # Generate all possible pairs (including identical pairs)
    for i in 1:para.Nk, j in 1:i  # i >= j to avoid duplicates
        # Calculate total momentum using existing function
        pair_indices = (i, j)
        K_total = MBS_totalmomentum(para, pair_indices)

        # Add to appropriate group
        if haskey(momentum_groups, K_total)
            push!(momentum_groups[K_total], pair_indices)
        else
            momentum_groups[K_total] = [pair_indices]
        end
    end
    
    if isnothing(momentum_transformation)
        return momentum_groups
    else
        # what should it do?
    end
end

"""
    scat_pair_group_coordinate(pair_group, para, shifts) -> Vector{Scatter{2}}

Generate all Scatter terms between momentum pairs with the same total momentum.
This internal function uses an interaction function `V_int` that accepts momentum coordinates.

# Arguments
- `pair_group::Vector{Tuple{Int64,Int64}}`: List of momentum index pairs with the same total momentum.
- `para::EDPara`: Parameter structure containing system configuration.
- `shifts::Matrix{<:Real}`: A matrix of size `(2, Nc_conserve)` specifying the momentum shifts (twisted boundary conditions) for each conserved component.

# Returns
- `Vector{Scatter{2}}`: A list of two-body `Scatter` terms for this momentum group.

# Details
- Iterates over all input/output momentum pair combinations within the group.
- Generates all component index combinations for each momentum pair.
- Maps momentum and component indices to global orbital indices.
- Applies normal ordering: `minmax(i1, i2) >= minmax(f1, f2)`.
- Calculates Scatter amplitudes using `para.V_int` with momentum shifts.
- Includes both direct (`V(f1,f2,i2,i1)`) and exchange (`V(f1,f2,i1,i2)`) contributions.
"""
function scat_pair_group_coordinate(pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara;
    MBS_type::Type{<:MBS64} = MBS64{para.Nc*para.Nk}, element_type::Type{F},
    shifts::Matrix{T}, V_int, int_kwargs... )::Vector{Scatter{Complex{F}, MBS_type}} where {T <: Real, F <: AbstractFloat}
    
    # @assert size(shifts) == (2, para.Nc_conserve)

    Nc = para.Nc
    Nk = para.Nk
    Gk1, Gk2 = para.Gk
    sys_size = (Gk1 != 0 && Gk2 != 0) ? Nk : 1

    if T <: Integer || T <: Rational
        klist = copy(para.k_list) // 1
        kshift = copy(shifts) // 1
        if Gk1 != 0
            klist[1, :] .//= Gk1
            kshift[1, :] .//= Gk1
        end
        if Gk2 != 0
            klist[2, :] .//= Gk2
            kshift[2, :] .//= Gk2
        end
    else
        klist = float.(copy(para.k_list))
        kshift = float.(copy(shifts))
        if Gk1 != 0
            klist[1, :] ./= Gk1
            kshift[1, :] ./= Gk1
        end
        if Gk2 != 0
            klist[2, :] ./= Gk2
            kshift[2, :] ./= Gk2
        end
    end



    Scatter_list = Vector{Scatter{Complex{F}, MBS_type}}()
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

            # conserved component index determines momentum shift
            # component_index = cm_index + Nc_mix * (cc_index - 1)
            ccf1 = fld1(cf1, para.Nc_mix)
            ccf2 = fld1(cf2, para.Nc_mix)
            cci2 = fld1(ci2, para.Nc_mix)
            cci1 = fld1(ci1, para.Nc_mix)

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
                amp_direct = V_int(
                    (klist[1, kf1] + kshift[1, ccf1], klist[2, kf1] + kshift[2, ccf1]),
                    (klist[1, kf2] + kshift[1, ccf2], klist[2, kf2] + kshift[2, ccf2]),
                    (klist[1, ki2] + kshift[1, cci2], klist[2, ki2] + kshift[2, cci2]),
                    (klist[1, ki1] + kshift[1, cci1], klist[2, ki1] + kshift[2, cci1]),
                    cf1, cf2, ci2, ci1; int_kwargs...
                ) |> Complex{F}

                # exchange i1 and i2
                PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i2, Nk), fldmod1(i1, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk))
                amp_exchange = V_int(
                    (klist[1, kf1] + kshift[1, ccf1], klist[2, kf1] + kshift[2, ccf1]),
                    (klist[1, kf2] + kshift[1, ccf2], klist[2, kf2] + kshift[2, ccf2]),
                    (klist[1, ki1] + kshift[1, cci1], klist[2, ki1] + kshift[2, cci1]),
                    (klist[1, ki2] + kshift[1, cci2], klist[2, ki2] + kshift[2, cci2]),
                    cf1, cf2, ci1, ci2; int_kwargs...
                ) |> Complex{F}

                amp = (amp_direct - amp_exchange) / sys_size
                iszero(amp) || push!(Scatter_list, Scatter(amp, f1, f2, i2, i1; 
                    upper_hermitian = true, bits = para.Nc * para.Nk
                ))
                PRINT_TWOBODY_SCATTER_PAIRS && println()
            end
        
        end
    end
    
    return Scatter_list
end

"""
    scat_pair_group_index(pair_group, para) -> Vector{Scatter{2}}

Generate all Scatter terms between momentum pairs with the same total momentum.
This internal function uses an interaction function `V_int` that accepts momentum indices.

# Arguments
- `pair_group::Vector{Tuple{Int64,Int64}}`: List of momentum index pairs with the same total momentum.
- `para::EDPara`: Parameter structure containing system configuration.

# Returns
- `Vector{Scatter{2}}`: A list of two-body `Scatter` terms for this momentum group.

# Details
- This function is used when `para.V_int` expects integer indices instead of coordinates.
- It does not handle momentum shifts (twisted boundary conditions); the `para` object itself should be updated if necessary.
- Iterates over all input/output momentum pair combinations within the group.
- Applies normal ordering and calculates direct and exchange amplitudes.
"""
function scat_pair_group_index(pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara;
    MBS_type::Type{<:MBS64} = MBS64{para.Nc*para.Nk}, element_type::Type{F},
    V_int, int_kwargs... )::Vector{Scatter{Complex{F}, MBS_type}} where{F <: AbstractFloat}
    
    Nc = para.Nc
    Nk = para.Nk
    Gk1, Gk2 = para.Gk
    sys_size = (Gk1 != 0 && Gk2 != 0) ? Nk : 1

    Scatter_list = Vector{Scatter{Complex{F}, MBS_type}}()
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
                amp_direct = V_int(
                    kf1, kf2, ki2, ki1,
                    cf1, cf2, ci2, ci1; int_kwargs...
                ) |> Complex{F}

                # exchange i1 and i2
                PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i2, Nk), fldmod1(i1, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk))
                amp_exchange = V_int(
                    kf1, kf2, ki1, ki2,
                    cf1, cf2, ci1, ci2; int_kwargs...
                ) |> Complex{F}

                amp = (amp_direct - amp_exchange) / sys_size
                iszero(amp) || push!(Scatter_list, Scatter(amp, f1, f2, i2, i1; 
                    upper_hermitian = true, bits = para.Nc*para.Nk
                ))
                PRINT_TWOBODY_SCATTER_PAIRS && println()
            end
        
        end
    end
    
    return Scatter_list
end

"""
    ED_sortedScatterList_twobody(para::EDPara; kshift=nothing) -> Vector{Scatter{2}}

Generate a sorted list of two-body `Scatter` terms from the interaction potential.

This function orchestrates the generation of all two-body scattering terms. It first groups momentum pairs by their total momentum to ensure conservation, then calls the appropriate internal function (`scat_pair_group_coordinate` or `scat_pair_group_index`) based on the signature of the interaction potential `para.V_int`.

# Arguments
- `para::EDPara`: The parameter structure containing system configuration, including the `V_int` function.

# Keywords
- `kshift=nothing`: Specifies a momentum shift for twisted boundary conditions. This is only applicable if `para.V_int` accepts momentum coordinates. The value can be:
    - `nothing` (default): No shift.
    - `Tuple{Real, Real}`: A uniform shift `(kx, ky)` applied to all conserved components.
    - `Vector{Tuple{Real, Real}}`: A specific shift for each conserved component.
    - `Matrix{Real}`: A `2 x Nc_conserve` matrix specifying the shift for each component.

# Returns
- `Vector{Scatter{2}}`: A sorted and merged list of all unique two-body `Scatter` terms.

# Details
- Automatically detects whether `para.V_int` uses momentum coordinates or indices.
- If using indices, `kshift` is ignored.
- The final list is processed by `sort_merge_scatlist` to eliminate duplicates and sort the terms.

# Example
```julia
para = EDPara(k_list=k_list, Gk=(3, 5), V_int=V_int)
# Generate with no momentum shift
Scatter2 = ED_sortedScatterList_twobody(para)
# Generate with a uniform twisted boundary condition
Scatter2_shifted = ED_sortedScatterList_twobody(para; kshift=(0.1, 0.1))
```
"""
function ED_sortedScatterList_twobody(para::EDPara; MBS_type::Type{<:MBS64} = MBS64{para.Nc*para.Nk},
    momentum_transformation::Union{Nothing, Function} = nothing, element_type::Type{F} = Float64,
    replace_interaction::Union{Nothing, Function} = nothing, 
    rep_int_use_momentum_coordinate::Bool = para.two_momentum_coordinate,
    kshift = nothing, int_kwargs... )::Vector{Scatter{Complex{F}, MBS_type}} where{F <: AbstractFloat}

    if isnothing(replace_interaction)
        replace_interaction = para.H_two
    end

    momentum_groups = group_momentum_pairs(para; momentum_transformation)
    
    sct_list2 = Vector{Scatter{Complex{F}, MBS_type}}()
    if rep_int_use_momentum_coordinate

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
            append!(sct_list2, scat_pair_group_coordinate(pairs, para; shifts, element_type, V_int = replace_interaction, int_kwargs...))
        end
    else
        if !isnothing(kshift)
            @warn "kshift is ignored when para.V_int accepts momentum indices instead of coordinates."
        end

        for (K_total, pairs) in momentum_groups
            append!(sct_list2, scat_pair_group_index(pairs, para; element_type, V_int = replace_interaction, int_kwargs...))
        end
    end

    
    return sort_merge_scatlist(sct_list2)
end
