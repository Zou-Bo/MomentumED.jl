"""
    EDPara - Parameters for momentum-conserved exact diagonalization
    
    This struct contains all parameters needed for momentum-conserved exact diagonalization
    calculations, including momentum conservation rules, component structure, and interaction
    potentials.
"""

"""
    mutable struct EDPara

Stores all parameters for a momentum-conserved exact diagonalization calculation.

# Constructor

    EDPara(; 
        k_list::Matrix{Int64},
        Gk::Tuple{Int64, Int64} = (0, 0), 
        Nc_mix::Int64 = 1,
        Nc_conserve::Int64 = 1,
        H_one = (kf, ki, cf, ci) -> 0.0 + 0.0im,
        H_two = (kf1, kf2, ki1, ki2, cf1, cf2, ci1, ci2) -> 0.0 + 0.0im,
    )

# Keyword Arguments
- `k_list::Matrix{Int64}`: **Required**. A matrix where each column `k_list[:, i]` represents a momentum vector `(k_x, k_y)`.
- `Gk::Tuple{Int64, Int64} = (0, 0)`: The reciprocal lattice vectors `(G1, G2)`. Momentum is conserved modulo `Gk`. A value of `0` means no periodicity in that direction.
- `Nc_mix::Int64 = 1`: The number of "hopping" components whose particle numbers are not conserved.
- `Nc_conserve::Int64 = 1`: The number of components where the particle number is conserved (e.g., valley).
- `H_one`: A function or callable object that generates the one-body part of the Hamiltonian. Defaults to zero.
- `H_two`: A function or callable object defining the two-body interaction potential. Defaults to zero.

# Fields (in addition to keyword arguments)
- `Nk::Int64`: The number of momentum states, derived from `size(k_list, 2)`.
- `Nc::Int64`: The total number of components, `Nc_mix * Nc_conserve`.
- `one_momentum_coordinate::Bool`: A flag that is automatically set to `true` if `H_one` accepts momentum coordinates (`Tuple{Float64,Float64}`), and `false` if it accepts momentum indices (`Int64`).
- `two_momentum_coordinate::Bool`: A flag that is automatically set to `true` if `H_two` accepts momentum coordinates (`Tuple{Float64,Float64}`), and `false` if it accepts momentum indices (`Int64`).

# Band Dispersion Function (`H_one`) and Interaction Function (`H_two`)
The `H_one` function must accept 4 arguments and return a complex number.
`H_one(kf, ki, cf, ci)` -> a complex number.
The `H_two` function must accept 8 arguments and return a complex number.. 
`H_two(kf1, kf2, ki2, ki1, cf1::Int64, cf2::Int64, ci2::Int64, ci1::Int64)` -> a complex number.
Input momenta can have one of two signatures:
1.  Coordinate-based: `ki` and `kf` are momentum coordinates as `Tuple{<:Real, <:Real}`.
2.  Index-based: `ki` and `kf` are momentum indices as `Int64`, refering to the momenta in `k_list`.
The constructor automatically detects which signature is used.

# Orbital Indexing
The single orbital index `i` is mapped from a multi-index `(i_k, i_ch, i_cc)` as follows:
`(i - 1) = (i_k - 1) + Nk * (i_ch - 1) + (Nk * Nc_hopping) * (i_cc - 1)`
where `i_k` is the momentum index, `i_ch` is the hopping component index, and `i_cc` is the conserved component index.
For the component only, `(ic - 1) = (i_ch - 1) + Nc_hopping * (i_cc - 1)`, and `(i - 1) = (i_k - 1) + Nk * (ic - 1)`.
The "-1" is due to 1-based indexing in Julia.

# Validation
- The total number of orbitals (`Nk * Nc`) must not exceed 64.
- The provided `H_one` and `H_two` functions must conform to one of the valid signatures. 
"""
struct EDPara{H1, H2}
    # momemta are in integers

    # k_list[:, i] = (k_x, k_y)
    k_list::Matrix{Int64}
    # G integer (momentum integer is conserved mod G; where G=0 means no mod)
    Gk::Tuple{Int64, Int64}

    Nk::Int64 # number of momenta
    Nc_mix::Int64 # number of components that will be mixed by Hamiltonian (not conserved)
    Nc_conserve::Int64  # number of components with conserved quantum numbers
    Nc::Int64 # total number of components = Nc_mix * Nc_conserve

    H_one::H1
    one_momentum_coordinate::Bool
    H_two::H2
    two_momentum_coordinate::Bool

    function EDPara(; 
        k_list::Matrix{Int64},
        Gk::Tuple{Int64, Int64} = (0, 0), 
        Nc_mix::Int64 = 1,
        Nc_conserve::Int64 = 1,
        H_one::H1 = Returns(0.0 + 0.0im),
        H_two::H2 = Returns(0.0 + 0.0im),
        one_momentum_coordinate::Union{Nothing, Bool} = nothing,
        two_momentum_coordinate::Union{Nothing, Bool} = nothing,
    ) where {H1, H2}

        # Calculate derived fields
        Nk = size(k_list, 2)
        Nc = Nc_conserve * Nc_mix

        # Validation
        @assert Nc > 0 "Number of components must be positive"
        @assert Nk*Nc <= 64 "The total orbital number exceeds 64 bits."
        
        # Validate H_one function signature - accept Tuple{<:Real,<:Real} or Int64 momentum format
        if !isnothing(one_momentum_coordinate)
            if one_momentum_coordinate
                @assert hasmethod(H_one, Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}, Int64, Int64}) "H_one is expected to accept (kf::Tuple{<:Real,<:Real}, ki::Tuple{<:Real,<:Real}, cf::Int64, ci::Int64), but H_one does not have the correct method signature."
            else
                @assert hasmethod(H_one, Tuple{Int64, Int64, Int64, Int64}) "H_one is expected to accept (kf::Int64, ki::Int64, cf::Int64, ci::Int64), but H_one does not have the correct method signature."
            end
        else
            if hasmethod(H_one, Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}, Int64, Int64})
                one_momentum_coordinate = true
            elseif hasmethod(H_one, Tuple{Int64, Int64, Int64, Int64})
                one_momentum_coordinate = false
            else
                throw(AssertionError("""
                H_one function must accept 4 arguments:
                    either (kf::Tuple{<:Real,<:Real}, ki::Tuple{<:Real,<:Real}, cf::Int64, ci::Int64)
                    or (kf::Int64, ki::Int64, cf::Int64, ci::Int64);
                and return a complex number.
                Current function fails to give H_one((0.0,0.0), (0.0,0.0), 1, 1) or H_one(1, 1, 1, 1)
                """))
            end
        end
        
        # Validate H_two function signature - accept Tuple{<:Real,<:Real} or Int64 momentum format
        if !isnothing(two_momentum_coordinate)
            if two_momentum_coordinate
                @assert hasmethod(H_two, Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}, Int64, Int64, Int64, Int64}) "H_two is expected to accept (kf1::Tuple{<:Real,<:Real}, kf2::Tuple{<:Real,<:Real}, ki1::Tuple{<:Real,<:Real}, ki2::Tuple{<:Real,<:Real}, cf1::Int64, cf2::Int64, ci1::Int64, ci2::Int64), but H_two does not have the correct method signature."
            else
                @assert hasmethod(H_two, Tuple{Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64}) "H_two is expected to accept (kf1::Int64, kf2::Int64, ki1::Int64, ki2::Int64, cf1::Int64, cf2::Int64, ci1::Int64, ci2::Int64), but H_two does not have the correct method signature."
            end
        else
            if hasmethod(H_two, Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}, Int64, Int64, Int64, Int64})
                two_momentum_coordinate = true
            elseif hasmethod(H_two, Tuple{Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64})
                two_momentum_coordinate = false
            else
                throw(AssertionError("""
                H_two function must accept 8 arguments:
                    either (kf1::Tuple{<:Real,<:Real}, kf2::Tuple{<:Real,<:Real}, ki1::Tuple{<:Real,<:Real}, ki2::Tuple{<:Real,<:Real}, cf1::Int64, cf2::Int64, ci1::Int64, ci2::Int64)
                    or (kf1::Int64, kf2::Int64, ki1::Int64, ki2::Int64, cf1::Int64, cf2::Int64, ci1::Int64, ci2::Int64)"));
                and return ComplexF64.
                Current function fails to give ComplexF64 H_two((0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0), 1, 1, 1, 1) or H_two(1, 1, 1, 1, 1, 1, 1, 1)
                """))
            end
        end

        new{H1, H2}(k_list, Gk, Nk, Nc_mix, Nc_conserve, Nc, H_one, one_momentum_coordinate, H_two, two_momentum_coordinate)
    end
end

