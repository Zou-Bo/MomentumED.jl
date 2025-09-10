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
- `k_list::Matrix{Int64}`: k_list[:, i] = (k_x, k_y) momentum states
- `Nk::Int64`: Number of momentum states
- `Nc_hopping::Int64`: Number of components with hopping (not conserved)
- `Nc_conserve::Int64`: Number of components with conserved quantum numbers
- `Nc::Int64`: Total number of components
- `H_onebody::Array{ComplexF64,4}`: One-body Hamiltonian terms
- `V_int::Function`: Interaction potential function

# Orbital Indexing
The orbital index formula is: i = i_k + Nk * (i_ch-1) + (Nk * Nc_hopping) * (i_cc-1)
or i = i_k + Nk * (i_c-1), where i_c = i_ch + Nc_hopping * (i_cc-1)

# Validation
- `Nc > 0`: Number of components must be positive
- `Nk*Nc <= 64`: Hilbert space dimension must not exceed 64 bits for MBS64 compatibility
- `V_int` function must have correct signature: (kf1, kf2, ki1, ki2, cf1, cf2, ci1, ci2) -> ComplexF64
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

    # (to be fill later by user)
    H_onebody::Array{ComplexF64,4} 
    V_int::Function

    """
    Constructor for EDPara with keyword arguments and validation.
    """
    function EDPara(; Gk::Tuple{Int64, Int64}=(0, 0), 
                     k_list::Matrix{Int64},
                     Nc_hopping::Int64=1,
                     Nc_conserve::Int64=1,
                     H_onebody::Array{ComplexF64,4}=zeros(ComplexF64, Nc_hopping, Nc_hopping, Nc_conserve, size(k_list, 2)),
                     V_int::Function)

        # Calculate derived fields
        Nk = size(k_list, 2)
        Nc = Nc_conserve * Nc_hopping
        
        # Validation
        @assert Nc > 0 "Number of components must be positive"
        @assert Nk*Nc <= 64 "The Hilbert space dimension must not exceed 64 bits."
        
        # Validate V_int function signature - must accept new Tuple{Float64,Float64} momentum format
        if !hasmethod(V_int, Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64},Tuple{Float64,Float64},Tuple{Float64,Float64},Int,Int,Int,Int})
            throw(AssertionError("V_int function must accept 8 arguments with new kshift format: (kf1::Tuple{Float64,Float64}, kf2::Tuple{Float64,Float64}, ki1::Tuple{Float64,Float64}, ki2::Tuple{Float64,Float64}, cf1::Int, cf2::Int, ci1::Int, ci2::Int)"))
        end

        new(Gk, k_list, Nk, Nc_hopping, Nc_conserve, Nc, H_onebody, V_int)
    end
end

