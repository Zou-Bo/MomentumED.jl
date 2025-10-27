"""
    EDPara - Parameters for momentum-conserved exact diagonalization
    
    This struct contains all parameters needed for momentum-conserved exact diagonalization
    calculations, including momentum conservation rules, component structure, and interaction
    potentials.
"""

"""
    mutable struct EDPara

Stores all parameters for a momentum-conserved exact diagonalization calculation.

The constructor is keyword-based.

# Keyword Arguments
- `k_list::Matrix{Int64}`: **Required**. A matrix where each column `k_list[:, i]` represents a momentum vector `(k_x, k_y)`.
- `Gk::Tuple{Int64, Int64} = (0, 0)`: The reciprocal lattice vectors `(G1, G2)`. Momentum is conserved modulo `Gk`. A value of `0` means no periodicity in that direction.
- `Nc_hopping::Int64 = 1`: The number of "hopping" components whose particle numbers are not conserved.
- `Nc_conserve::Int64 = 1`: The number of components where the particle number is conserved (e.g., valley).
- `H_onebody::Array{ComplexF64,4}`: The one-body part of the Hamiltonian. Defaults to zeros.
- `V_int::Function`: A function defining the two-body interaction potential. See "Interaction Function" below. Defaults to a function that returns zero.
- `FF_inf_angle::Function`: A function defining the Berry connection for infinitesimal transformations. See "Berry Connection Function" below. Defaults to a function that returns zero.

# Fields (in addition to keyword arguments)
- `Nk::Int64`: The number of momentum states, derived from `size(k_list, 2)`.
- `Nc::Int64`: The total number of components, `Nc_hopping * Nc_conserve`.
- `momentum_coordinate::Bool`: A flag that is automatically set to `true` if `V_int` accepts momentum coordinates (`Tuple{Float64,Float64}`), and `false` if it accepts momentum indices (`Int64`).

# Interaction Function (`V_int`)
The `V_int` function must accept 8 arguments and return a `ComplexF64`. 
`V_int(kf1, kf2, ki2, ki1, cf1::Int64, cf2::Int64, ci2::Int64, ci1::Int64)` -> `ComplexF64`.
Input momenta can have one of two signatures:
1.  Coordinate-based: `ki` and `kf` are momentum coordinates as `Tuple{<:Real, <:Real}`.
2.  Index-based: `ki` and `kf` are momentum indices as `Int64`, refering to the momenta in `k_list`.
The constructor automatically detects which signature is used.

# Berry Connection Function (`FF_inf_angle`)
The `FF_inf_angle` function must accept 3 arguments `(k_f, k_i, c)` and return a `Float64`. `k_f` and `k_i` are momentum coordinates.

# Orbital Indexing
The single orbital index `i` is mapped from a multi-index `(i_k, i_ch, i_cc)` as follows:
`i = i_k + Nk * (i_ch - 1) + (Nk * Nc_hopping) * (i_cc - 1)`
where `i_k` is the momentum index, `i_ch` is the hopping component index, and `i_cc` is the conserved component index.

# Validation
- The total number of orbitals (`Nk * Nc`) must not exceed 64.
- The provided `V_int` and `FF_inf_angle` functions must conform to one of the valid signatures.
"""
mutable struct EDPara
    # momemta are in integers

    # G integer (momentum is conserved mod G; where G=0 means no mod)
    Gk::Tuple{Int64, Int64}
    # k_list[:, i] = (k_x, k_y)
    k_list::Matrix{Int64}

    Nk::Int64 # number of momentum states
    Nc_hopping::Int64 # number of components with hopping (not conserved)
    Nc_conserve::Int64  # number of components with conserved quantum numbers
    Nc::Int64

    H_onebody::Array{ComplexF64,4} 
    V_int::Function
    momentum_coordinate::Bool
    FF_inf_angle::Function

    function EDPara(; 
        Gk::Tuple{Int64, Int64} = (0, 0), 
        k_list::Matrix{Int64},
        Nc_hopping::Int64 = 1,
        Nc_conserve::Int64 = 1,
        H_onebody::Array{ComplexF64,4} = zeros(ComplexF64, Nc_hopping, Nc_hopping, Nc_conserve, size(k_list, 2)),
        V_int::Function = (kf1, kf2, ki1, ki2, cf1, cf2, ci1, ci2) -> 0.0 + 0.0im,
        FF_inf_angle::Function = (k_f, k_i, c) -> 0.0
    )

        # Calculate derived fields
        Nk = size(k_list, 2)
        Nc = Nc_conserve * Nc_hopping
        momentum_coordinate = true
        
        # Validation
        @assert Nc > 0 "Number of components must be positive"
        @assert Nk*Nc <= 64 "The Hilbert space dimension must not exceed 64 bits."
        
        # Validate V_int function signature - accept Tuple{<:Real,<:Real} or Int64 momentum format
        try
            x = V_int((0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0), 1, 1, 1, 1)
            @assert x isa ComplexF64
        catch
            momentum_coordinate = false
            try
                x = V_int(1, 1, 1, 1, 1, 1, 1, 1)
                @assert x isa ComplexF64
            catch
                throw(AssertionError("""
                V_int function must accept 8 arguments:
                    either (kf1::Tuple{<:Real,<:Real}, kf2::Tuple{<:Real,<:Real}, ki1::Tuple{<:Real,<:Real}, ki2::Tuple{<:Real,<:Real}, cf1::Int64, cf2::Int64, ci1::Int64, ci2::Int64)
                    or (kf1::Int64, kf2::Int64, ki1::Int64, ki2::Int64, cf1::Int64, cf2::Int64, ci1::Int64, ci2::Int64)"));
                and return ComplexF64.
                Current function fails to give ComplexF64 V_int((0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0), 1, 1, 1, 1) or V_int(1, 1, 1, 1, 1, 1, 1, 1)
                """))
            end
        end


        # Validate FF_inf_angle function signature - must accept Tuple{Float64,Float64} momentum format
        if momentum_coordinate
            try
                x = FF_inf_angle((0.0,0.0), (0.0,0.0), 1)
                @assert x isa Float64
            catch
                throw(AssertionError("""
                FF_inf_angle function must accept 3 arguments: (k_f::Tuple{Float64,Float64}, k_i::Tuple{Float64,Float64}, c::Int),
                and return Float64.
                Current function fails to give Float64 FF_inf_angle((0.0,0.0), (0.0,0.0), 1)
                """))
            end
        end

        new(Gk, k_list, Nk, Nc_hopping, Nc_conserve, Nc, H_onebody, V_int, momentum_coordinate, FF_inf_angle)
    end
end

