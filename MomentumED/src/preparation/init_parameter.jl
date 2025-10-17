"""
    EDPara - Parameters for momentum-conserved exact diagonalization
    
    This struct contains all parameters needed for momentum-conserved exact diagonalization
    calculations, including momentum conservation rules, component structure, and interaction
    potentials.
"""

"""
    mutable struct EDPara

Parameters for momentum-conserved exact diagonalization calculations.

# Fields
- `Gk::Tuple{Int64, Int64}`: Momentum conservation mod G (default: (0, 0))
- `k_list::Matrix{Int64}`: k_list[:, i] = (k_1, k_2) momentum states
- `Nk::Int64`: Number of momentum states
- `Nc_hopping::Int64`: Number of components with hopping (not conserved)
- `Nc_conserve::Int64`: Number of components with conserved quantum numbers
- `Nc::Int64`: Total number of components
- `H_onebody::Array{ComplexF64,4}`: One-body Hamiltonian terms
- `V_int::Function`: Interaction potential function
- `momentum_coordinate::Bool`: The V_int function use coordinate or index format for input momentum 
- `FF_inf_angle::Function`: Berry connection step integral, argument of infinitesimal δk form factor

# Orbital Indexing
The orbital index formula is: i = i_k + Nk * (i_ch-1) + (Nk * Nc_hopping) * (i_cc-1)
or i = i_k + Nk * (i_c-1), where i_c = i_ch + Nc_hopping * (i_cc-1)

# Validation
- `Nc > 0`: Number of components must be positive
- `Nk*Nc <= 64`: Hilbert space dimension must not exceed 64 bits for MBS64 compatibility
- `V_int` function must have correct signature: (kf1, kf2, ki2, ki1, cf1, cf2, ci2, ci1) -> ComplexF64
- `FF_inf_angle` function must have correct signature: (k_f, k_i, c) -> Float64
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

    """
    Constructor for EDPara with keyword arguments and validation.
    """
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
        
        # Validate V_int function signature - accept Tuple{Float64,Float64} or Int64 momentum format
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
                    either (kf1::Tuple{Float64,Float64}, kf2::Tuple{Float64,Float64}, ki1::Tuple{Float64,Float64}, ki2::Tuple{Float64,Float64}, cf1::Int64, cf2::Int64, ci1::Int64, ci2::Int64)
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

