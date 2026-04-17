# todo list
# function to transform momentum
# update docstring

@inline function shifted_k_coordinate_rational(para::EDPara{dim}, shifts::Matrix) where {dim}
    klist = para.k_list .// 1
    kshift = shifts .// 1
    for d in 1:dim
        Gk = para.Gk[d]
        if !iszero(Gk)
            klist[d, :] .//= Gk
            kshift[d, :] .//= Gk
        end
    end
    return klist, kshift
end
@inline function shifted_k_coordinate_float(para::EDPara{dim}, shifts::Matrix) where {dim}
    klist = float.(para.k_list)
    kshift = float.(shifts)
    for d in 1:dim
        Gk = para.Gk[d]
        if !iszero(Gk)
            klist[d, :] ./= Gk
            kshift[d, :] ./= Gk
        end
    end
    return klist, kshift
end
@inline function shift_decipher(kshift, Ncc, dim::Int64)
    if isnothing(kshift)
        shifts = zeros(Int64, dim, Ncc)
    elseif kshift isa Tuple
        shifts = Matrix{eltype(kshift)}(undef, dim, Ncc)
        for d in 1:dim
            shifts[d,:] .= kshift[d]
        end
    elseif kshift isa Vector
        @assert length(kshift) == Ncc "length of kshift isn't equal to para.Nc_conserve."
        shifts = Matrix{eltype(kshift[1])}(undef, dim, Ncc)
        for i in 1:Ncc
            shifts[:,i] .= kshift[i]
        end
    elseif kshift isa Matrix
        @assert size(kshift) == (dim, Ncc) "size of kshift isn't equal to para.Nc_conserve."
        shifts = kshift
    else
        throw(AssertionError("kshift could be nothing, Tuple{R,R}, Vector{Tuple{R,R}}, or Matrix{R}, where R<:Real."))
    end
    return shifts
end

"""
    ED_scatterlist_onebody(para::EDPara) -> Vector{Scatter{1}}

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
Scatter1 = ED_scatterlist_onebody(para)
```
"""
function ED_scatterlist_onebody(para::EDPara{dim}; 
    MBS_type::Type{MBS} = MBS64{para.Nc*para.Nk}, element_type::Type{F} = Float64,
    kshift = nothing, H_one::NamedTuple = NamedTuple()
    )::Vector{Scatter{Complex{F}, MBS}} where{ F <: AbstractFloat, MBS <: MBS64, dim}

    Nk = para.Nk
    Nc = para.Nc
    Ncc = para.Nc_conserve
    Ncm = para.Nc_mix
    bits = Nc * Nk
    @assert MBS_type == MBS64{bits} "MBS bits has to be para.Nc*para.Nk."
    
    sct_list1 = Vector{Scatter{Complex{F}, MBS_type}}()
    # Extract one-body terms from H_one(k, cf, ci) and convert to Scatter terms
    if para.one_momentum_coordinate

        shifts = shift_decipher(kshift, Ncc, dim)
        T = eltype(shifts)

        if T <: Integer || T <: Rational
            klist, kshift = shifted_k_coordinate_rational(para, shifts)
        else
            klist, kshift = shifted_k_coordinate_float(para, shifts)
        end

        for cc in 1:Ncc, cmi in 1:Ncm, cmf in 1:Ncm, 
            cf = cmf + (cc - 1) * Ncm 
            ci = cmi + (cc - 1) * Ncm 
            for k in 1:Nk
                # Map component indices to global orbital indices
                i_out = k + Nk * (cf - 1)  # output orbital
                i_in  = k + Nk * (ci - 1)  #  input orbital
                if i_in >= i_out
                    V = para.H_one(
                        ntuple(d -> klist[d, k] + kshift[d, cc], Val(dim)),
                        cf, ci; H_one... 
                    ) |> Complex{F}
                    iszero(V) || push!(sct_list1, Scatter(V, i_out, i_in; bits, upper_hermitian = true))
                end
            end
        end

    else

        if !isnothing(kshift)
            @warn "kshift is ignored when para.H_one accepts momentum indices instead of coordinates."
        end

        for cc in 1:Ncc, cmi in 1:Ncm, cmf in 1:Ncm, 
            cf = cmf + (cc - 1) * Ncm 
            ci = cmi + (cc - 1) * Ncm 
            for k in 1:Nk
                # Map component indices to global orbital indices
                i_out = k + Nk * (cf - 1)  # output orbital
                i_in  = k + Nk * (ci - 1)  #  input orbital
                if i_in >= i_out
                    V = para.H_one(k, cf, ci; H_one...) |> Complex{F}
                    iszero(V) || push!(sct_list1, Scatter(V, i_out, i_in; bits, upper_hermitian = true))
                end
            end
        end
    end

    return sort_merge_scatlist(sct_list1)
end

"""
    group_momentum_pairs(para::EDPara) -> Dict{NTuple{dim, Int64}, Vector{Tuple{Int64,Int64}}}

Generate grouped momentum pairs by their total momentum.

Creates a dictionary mapping total momentum quantum numbers to lists of 
momentum index pairs that conserve that total momentum.

# Arguments
- `para::EDPara`: Parameter structure containing system configuration

# Returns
- `Dict{NTuple{dim, Int64}, Vector{Tuple{Int64,Int64}}}`: Dictionary where:
  - Keys are total momentum tuples `(K1, K2)`
  - Values are vectors of momentum index pairs `[(i,j), ...]` with that total momentum

# Details
- Generates all possible pairs `(i,j)` with `i >= j` to avoid duplicates
- Uses `MBS_totalmomentum(para, (i, j))` to compute total momentum for each pair
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
function group_momentum_pairs(para::EDPara{dim};
    momentum_transformation::Union{Nothing, Function} = nothing) where {dim}
    
    # Dictionary to store momentum groups
    momentum_groups = Dict{NTuple{dim, Int64}, Vector{Tuple{Int64,Int64}}}()
    
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
        error("Please leave momentum_transformation nothing.")
        # what should it do?
    end
end

# """
#     scat_pair_group_coordinate(pair_group, para, shifts) -> Vector{Scatter{2}}

# Generate all Scatter terms between momentum pairs with the same total momentum.
# This internal function uses an interaction function `V_int` that accepts momentum coordinates.

# # Arguments
# - `pair_group::Vector{Tuple{Int64,Int64}}`: List of momentum index pairs with the same total momentum.
# - `para::EDPara`: Parameter structure containing system configuration.
# - `shifts::Matrix{<:Real}`: A matrix of size `(2, Nc_conserve)` specifying the momentum shifts (twisted boundary conditions) for each conserved component.

# # Returns
# - `Vector{Scatter{2}}`: A list of two-body `Scatter` terms for this momentum group.

# # Details
# - Iterates over all input/output momentum pair combinations within the group.
# - Generates all component index combinations for each momentum pair.
# - Maps momentum and component indices to global orbital indices.
# - Applies normal ordering: `minmax(i1, i2) >= minmax(f1, f2)`.
# - Calculates Scatter amplitudes using `para.V_int` with momentum shifts.
# - Includes both direct (`V(f1,f2,i2,i1)`) and exchange (`V(f1,f2,i1,i2)`) contributions.
# """
# function scat_pair_group_coordinate!(scatter_list::Vector{Scatter{Complex{F}, MBS}},
#     pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara; element_type::Type{F},
#     shifts::Matrix{T}, H_two::NamedTuple = NamedTuple() ) where {T <: Real, F <: AbstractFloat, MBS <: MBS64}
    
#     # @assert size(shifts) == (2, para.Nc_conserve)

#     Nk = para.Nk
#     Nc = para.Nc
#     Ncm = para.Nc_mix
#     Gk1, Gk2 = para.Gk
#     sys_size = (Gk1 != 0 && Gk2 != 0) ? Nk : 1

#     if T <: Integer || T <: Rational
#         klist, kshift = shifted_k_coordinate_rational(para, shifts)
#     else
#         klist, kshift = shifted_k_coordinate_float(para, shifts)
#     end

#     # Iterate over all input and output pairs
#     for (ki1, ki2) in pair_group, (kf1, kf2) in pair_group
#         PRINT_TWOBODY_SCATTER_PAIRS && println()
#         PRINT_TWOBODY_SCATTER_PAIRS && println("ki1, ki2, kf1, kf2 = ($ki1, $ki2), ($kf1, $kf2)")
#         # Generate all component index combinations
#         for ci1 in 1:Nc, ci2 in 1:Nc, cf1 in 1:Nc, cf2 in 1:Nc
            
#             # Map to global orbital indices
#             # Full index = momentum_index + Nk * (component_index - 1)
#             f1 = kf1 + Nk * (cf1 - 1)
#             f2 = kf2 + Nk * (cf2 - 1)
#             i1 = ki1 + Nk * (ci1 - 1)
#             i2 = ki2 + Nk * (ci2 - 1)

#             # conserved component index determines momentum shift
#             # component_index = cm_index + Nc_mix * (cc_index - 1)
#             ccf1 = fld1(cf1, Ncm)
#             ccf2 = fld1(cf2, Ncm)
#             cci2 = fld1(ci2, Ncm)
#             cci1 = fld1(ci1, Ncm)

#             # no duplicate input/output indices
#             if i1 == i2 || f1 == f2
#                 continue
#             end

#             if ki1 == ki2 && i1 < i2
#                 continue
#             end

#             if kf1 == kf2 && f1 < f2
#                 continue
#             end

#             # conjugate Scatter only need to count onece, as the Hamiltonian is generated with upper half Hermitian()
#             if minmax(i1, i2) >= minmax(f1, f2)

#                 # Calculate the direct and exchange amplitudes
#                 PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i1, Nk), fldmod1(i2, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk),"        ")
#                 amp_direct = para.H_two(
#                     (klist[1, kf1] + kshift[1, ccf1], klist[2, kf1] + kshift[2, ccf1]),
#                     (klist[1, kf2] + kshift[1, ccf2], klist[2, kf2] + kshift[2, ccf2]),
#                     (klist[1, ki2] + kshift[1, cci2], klist[2, ki2] + kshift[2, cci2]),
#                     (klist[1, ki1] + kshift[1, cci1], klist[2, ki1] + kshift[2, cci1]),
#                     cf1, cf2, ci2, ci1; H_two...
#                 ) |> Complex{F}

#                 # exchange i1 and i2
#                 PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i2, Nk), fldmod1(i1, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk))
#                 amp_exchange = para.H_two(
#                     (klist[1, kf1] + kshift[1, ccf1], klist[2, kf1] + kshift[2, ccf1]),
#                     (klist[1, kf2] + kshift[1, ccf2], klist[2, kf2] + kshift[2, ccf2]),
#                     (klist[1, ki1] + kshift[1, cci1], klist[2, ki1] + kshift[2, cci1]),
#                     (klist[1, ki2] + kshift[1, cci2], klist[2, ki2] + kshift[2, cci2]),
#                     cf1, cf2, ci1, ci2; H_two...
#                 ) |> Complex{F}

#                 amp = (amp_direct - amp_exchange) / sys_size
#                 iszero(amp) || push!(scatter_list, Scatter(amp, f1, f2, i2, i1; 
#                     upper_hermitian = true, bits = Nc * Nk
#                 ))
#                 PRINT_TWOBODY_SCATTER_PAIRS && println()
#             end
#         end
#     end
# end

# """
#     scat_pair_group_index(pair_group, para) -> Vector{Scatter{2}}

# Generate all Scatter terms between momentum pairs with the same total momentum.
# This internal function uses an interaction function `V_int` that accepts momentum indices.

# # Arguments
# - `pair_group::Vector{Tuple{Int64,Int64}}`: List of momentum index pairs with the same total momentum.
# - `para::EDPara`: Parameter structure containing system configuration.

# # Returns
# - `Vector{Scatter{2}}`: A list of two-body `Scatter` terms for this momentum group.

# # Details
# - This function is used when `para.V_int` expects integer indices instead of coordinates.
# - It does not handle momentum shifts (twisted boundary conditions); the `para` object itself should be updated if necessary.
# - Iterates over all input/output momentum pair combinations within the group.
# - Applies normal ordering and calculates direct and exchange amplitudes.
# """
# function scat_pair_group_index!(scatter_list::Vector{Scatter{Complex{F}, MBS}},
#     pair_group::Vector{Tuple{Int64,Int64}}, para::EDPara; element_type::Type{F},
#     H_two::NamedTuple = NamedTuple() ) where{F <: AbstractFloat, MBS <: MBS64}
    
#     Nc = para.Nc
#     Nk = para.Nk
#     Gk1, Gk2 = para.Gk
#     sys_size = (Gk1 != 0 && Gk2 != 0) ? Nk : 1

#     # Iterate over all input and output pairs
#     for (ki1, ki2) in pair_group, (kf1, kf2) in pair_group
#         PRINT_TWOBODY_SCATTER_PAIRS && println()
#         PRINT_TWOBODY_SCATTER_PAIRS && println("ki1, ki2, kf1, kf2 = ($ki1, $ki2), ($kf1, $kf2)")
#         # Generate all component index combinations
#         for ci1 in 1:Nc, ci2 in 1:Nc, cf1 in 1:Nc, cf2 in 1:Nc
            
#             # Map to global orbital indices
#             # Global index = momentum_index + Nk * (component_index - 1)
#             f1 = kf1 + Nk * (cf1 - 1)
#             f2 = kf2 + Nk * (cf2 - 1)
#             i1 = ki1 + Nk * (ci1 - 1)
#             i2 = ki2 + Nk * (ci2 - 1)

#             # no duplicate input/output indices
#             if i1 == i2 || f1 == f2
#                 continue
#             end

#             if ki1 == ki2 && i1 < i2
#                 continue
#             end

#             if kf1 == kf2 && f1 < f2
#                 continue
#             end

#             # inverse Scatter only need to count onece, as the Hamiltonian is generated with upper half Hermitian()
#             if minmax(i1, i2) >= minmax(f1, f2)

#                 # Calculate the direct and exchange amplitudes
#                 PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i1, Nk), fldmod1(i2, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk),"        ")
#                 amp_direct = para.H_two(
#                     kf1, kf2, ki2, ki1,
#                     cf1, cf2, ci2, ci1; H_two...
#                 ) |> Complex{F}

#                 # exchange i1 and i2
#                 PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i2, Nk), fldmod1(i1, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk))
#                 amp_exchange = para.H_two(
#                     kf1, kf2, ki1, ki2,
#                     cf1, cf2, ci1, ci2; H_two...
#                 ) |> Complex{F}

#                 amp = (amp_direct - amp_exchange) / sys_size
#                 iszero(amp) || push!(scatter_list, Scatter(amp, f1, f2, i2, i1; 
#                     upper_hermitian = true, bits = Nc * Nk
#                 ))
#                 PRINT_TWOBODY_SCATTER_PAIRS && println()
#             end
#         end
#     end
# end

"""
    ED_scatterlist_twobody(para::EDPara; kshift=nothing) -> Vector{Scatter{2}}

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
Scatter2 = ED_scatterlist_twobody(para)
# Generate with a uniform twisted boundary condition
Scatter2_shifted = ED_scatterlist_twobody(para; kshift=(0.1, 0.1))
```
"""
function ED_scatterlist_twobody(para::EDPara{dim}; MBS_type::Type{<:MBS64} = MBS64{para.Nc*para.Nk},
    momentum_transformation::Union{Nothing, Function} = nothing, element_type::Type{F} = Float64,
    kshift = nothing, H_two::NamedTuple = NamedTuple() )::Vector{Scatter{Complex{F}, MBS_type}} where{F <: AbstractFloat, dim}

    Nk = para.Nk
    Nc = para.Nc
    Ncm = para.Nc_mix
    Ncc = para.Nc_conserve
    bits = Nc * Nk
    @assert MBS_type == MBS64{bits} "MBS bits has to be para.Nc*para.Nk."

    Gk = para.Gk
    sys_size = all(Gk .!= 0) ? Nk : 1

    momentum_groups = group_momentum_pairs(para; momentum_transformation)
    
    sct_list2 = Vector{Scatter{Complex{F}, MBS_type}}()
    if para.two_momentum_coordinate
        shifts = shift_decipher(kshift, Ncc, dim)
        T = eltype(shifts)

        if T <: Integer || T <: Rational
            klist, kshift = shifted_k_coordinate_rational(para, shifts)
        else
            klist, kshift = shifted_k_coordinate_float(para, shifts)
        end

        for (K_total, pairs) in momentum_groups
            for (ki1, ki2) in pairs, (kf1, kf2) in pairs
                PRINT_TWOBODY_SCATTER_PAIRS && println()
                PRINT_TWOBODY_SCATTER_PAIRS && println("ki1, ki2, kf1, kf2 = ($ki1, $ki2), ($kf1, $kf2)")
                # Generate all component index combinations
                for ci1 in 1:Nc, ci2 in 1:Nc, cf1 in 1:Nc, cf2 in 1:Nc
                    
                    # Map to global orbital indices
                    # Full index = momentum_index + Nk * (component_index - 1)
                    f1 = kf1 + Nk * (cf1 - 1)
                    f2 = kf2 + Nk * (cf2 - 1)
                    i1 = ki1 + Nk * (ci1 - 1)
                    i2 = ki2 + Nk * (ci2 - 1)

                    # conserved component index determines momentum shift
                    # component_index = cm_index + Nc_mix * (cc_index - 1)
                    ccf1 = fld1(cf1, Ncm)
                    ccf2 = fld1(cf2, Ncm)
                    cci2 = fld1(ci2, Ncm)
                    cci1 = fld1(ci1, Ncm)

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

                        coordinate_kf1 = ntuple(d -> klist[d, kf1] + kshift[d, ccf1], Val(dim))
                        coordinate_kf2 = ntuple(d -> klist[d, kf2] + kshift[d, ccf2], Val(dim))
                        coordinate_ki1 = ntuple(d -> klist[d, ki1] + kshift[d, cci1], Val(dim))
                        coordinate_ki2 = ntuple(d -> klist[d, ki2] + kshift[d, cci2], Val(dim))

                        # Calculate the direct and exchange amplitudes
                        PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i1, Nk), fldmod1(i2, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk),"        ")
                        amp_direct = para.H_two(
                            coordinate_kf1, coordinate_kf2, coordinate_ki2, coordinate_ki1,
                            cf1, cf2, ci2, ci1; H_two...
                        ) |> Complex{F}

                        # exchange i1 and i2
                        PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i2, Nk), fldmod1(i1, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk))
                        amp_exchange = para.H_two(
                            coordinate_kf1, coordinate_kf2, coordinate_ki1, coordinate_ki2,
                            cf1, cf2, ci1, ci2; H_two...
                        ) |> Complex{F}

                        amp = (amp_direct - amp_exchange) / sys_size
                        iszero(amp) || push!(sct_list2, Scatter(amp, f1, f2, i2, i1; bits, upper_hermitian = true))
                        PRINT_TWOBODY_SCATTER_PAIRS && println()
                    end
                end
            end
        end
    else
        if !isnothing(kshift)
            @warn "kshift is ignored when para.H_two accepts momentum indices instead of coordinates."
        end
        for (K_total, pairs) in momentum_groups
            for (ki1, ki2) in pairs, (kf1, kf2) in pairs
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
                        amp_direct = para.H_two(
                            kf1, kf2, ki2, ki1,
                            cf1, cf2, ci2, ci1; H_two...
                        ) |> Complex{F}

                        # exchange i1 and i2
                        PRINT_TWOBODY_SCATTER_PAIRS && print(fldmod1(i2, Nk), fldmod1(i1, Nk), fldmod1(f1, Nk), fldmod1(f2, Nk))
                        amp_exchange = para.H_two(
                            kf1, kf2, ki1, ki2,
                            cf1, cf2, ci1, ci2; H_two...
                        ) |> Complex{F}

                        amp = (amp_direct - amp_exchange) / sys_size
                        iszero(amp) || push!(sct_list2, Scatter(amp, f1, f2, i2, i1; bits, upper_hermitian = true))
                        PRINT_TWOBODY_SCATTER_PAIRS && println()
                    end
                end
            end
        end
    end
    return sort_merge_scatlist(sct_list2)
end

# ED_sortedScatterList_onebody = ED_scatterlist_onebody
# ED_sortedScatterList_twobody = ED_scatterlist_twobody