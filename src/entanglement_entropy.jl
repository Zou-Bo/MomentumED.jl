"""
    entanglement_entropy.jl - Entanglement Entropy Calculations for Momentum Eigenstates

This module provides minimal bipartite entanglement entropy calculations for momentum 
eigenstates using bit position bipartitions. Implements von Neumann and Renyi 
entropy calculations with direct computation from eigenvectors without storage.

Core Functions:
- ED_etg_entropy: Main interface for entropy calculations
- rdm_eigenvalue_small: Helper for determining smaller subsystem
- rdm_eigenvalue: Helper for reduced density matrix eigenvalue computation
"""

"""
    ED_etg_entropy(block::Vector{MBS64{bits}}, 
                   eigenvector::Vector{ComplexF64}, 
                   bit_mask::MBS64{bits};
                   alpha::Float64=1.0) where {bits}

Compute entanglement entropy for a momentum eigenstate with specified bit mask bipartition.

# Arguments
- `block::Vector{MBS64{bits}}`: Vector of MBS64 states representing the momentum block
- `eigenvector::Vector{ComplexF64}`: Eigenvector from diagonalization
- `bit_mask::MBS64{bits}`: Bit mask where all bits in subsystem A are 1
- `alpha::Float64=1.0`: Rényi entropy order (α=1 for von Neumann entropy)

# Returns
- `Float64`: Entanglement entropy value

# Examples
```julia
# Simple 4-site system with bit mask
bits = 4
block = [MBS64{bits}(0), MBS64{bits}(1), MBS64{bits}(2), MBS64{bits}(3)]
eigenvector = [0.5+0im, 0.5+0im, 0.5+0im, 0.5+0im]
bit_mask = MBS64{bits}(0b0011)  # First 2 bits in subsystem A
entropy = ED_etg_entropy(block, eigenvector, bit_mask)

# Rényi entropy of order 2
entropy_2 = ED_etg_entropy(block, eigenvector, bit_mask, alpha=2.0)
```

# Notes
- Supports bit position bipartitions only
- Direct calculation without storing reduced density matrix
- Handles numerical precision for near-zero eigenvalues
- For α=1, computes von Neumann entropy: S = -Tr(ρ_A log ρ_A)
- For α≠1, computes Rényi entropy: S_α = (1-α)^(-1) log Tr(ρ_A^α)
- Automatically computes RDM for smaller subsystem for efficiency
"""
function ED_etg_entropy(block::Vector{MBS64{bits}}, 
                       eigenvector::Vector{ComplexF64}, 
                       bit_mask::MBS64{bits};
                       alpha::Float64=1.0) where {bits}
    
    # Validate inputs
    @assert length(block) == length(eigenvector) "Block and eigenvector must have same length"
    
    # 1. Direct entropy calculation from eigenstate using MBS64 bit mask
    # No reduced density matrix storage - compute directly
    eigenvals = rdm_eigenvalue_small(block, eigenvector, bit_mask)
    
    # 2. Compute entropy based on alpha
    if abs(alpha - 1.0) < 1e-10
        # Von Neumann entropy: S = -Tr(ρ_A log ρ_A)
        entropy = -sum(eigenvals .* log.(eigenvals))
    else
        # Rényi entropy: S_α = (1-α)^(-1) log Tr(ρ_A^α)
        if abs(alpha) < 1e-10
            # Special case for α→0: logarithm of rank
            entropy = log(length(eigenvals))
        else
            trace_alpha = sum(eigenvals .^ alpha)
            entropy = log(trace_alpha) / (1.0 - alpha)
        end
    end
    
    return entropy
end

"""
    rdm_eigenvalue_small(block::Vector{MBS64{bits}}, 
                        state::Vector{ComplexF64}, 
                        bit_mask::MBS64{bits}) where {bits}

Compute eigenvalues of the reduced density matrix, automatically choosing the smaller subsystem.

Direct eigenvalue computation using MBS64 block and bit mask, determining which subsystem 
is smaller for efficiency.

# Arguments
- `block::Vector{MBS64{bits}}`: Vector of MBS64 states representing the momentum block
- `state::Vector{ComplexF64}`: State vector (eigenvector)
- `bit_mask::MBS64{bits}`: Bit mask where all bits in subsystem A are 1

# Returns
- `Vector{Float64}`: Eigenvalues of the reduced density matrix for the smaller subsystem

# Notes
- Automatically determines which subsystem is smaller for efficiency
- Computes RDM for smaller subsystem to minimize memory usage
- Uses complement mask for subsystem B when it is smaller
"""
function rdm_eigenvalue_small(block::Vector{MBS64{bits}}, 
                            state::Vector{ComplexF64}, 
                            bit_mask::MBS64{bits}) where {bits}
    # Direct eigenvalue computation using MBS64 block and bit mask
    # Determine which subsystem is smaller for efficiency
    n_bits_A = count_ones(bit_mask.n)
    n_bits_B = bits - n_bits_A
    
    if n_bits_A <= n_bits_B
        # Compute RDM for subsystem A (smaller)
        return rdm_eigenvalue(block, state, bit_mask)
    else
        # Compute RDM for subsystem B (smaller) using complementary mask
        complement_mask = reinterpret(MBS64{bits}, ((UInt64(1) << (bits - 1)) << 1 + 1) - bit_mask.n)
        return rdm_eigenvalue(block, state, complement_mask)
    end
end

"""
    rdm_eigenvalue(block::Vector{MBS64{bits}}, 
                  state::Vector{ComplexF64}, 
                  mask::MBS64{bits}) where {bits}

Compute eigenvalues of the reduced density matrix for a given bit mask.

Computes RDM eigenvalues for the smaller subsystem using given mask, working directly 
with the bit representation for efficiency.

# Arguments
- `block::Vector{MBS64{bits}}`: Vector of MBS64 states representing the momentum block
- `state::Vector{ComplexF64}`: State vector (eigenvector)
- `mask::MBS64{bits}`: Bit mask for the subsystem to compute RDM for

# Returns
- `Vector{Float64}`: Eigenvalues of the reduced density matrix

# Notes
- Uses direct computation to avoid storing large matrices
- Handles sparse state vectors efficiently
- Filters out negligible eigenvalues for numerical stability
- Works with bit representation for maximum efficiency
"""
function rdm_eigenvalue(block::Vector{MBS64{bits}}, 
                       state::Vector{ComplexF64}, 
                       mask::MBS64{bits}) where {bits}
    # Compute RDM eigenvalues for the smaller subsystem using given mask
    eigenvals = zeros(Float64, 2^count_ones(mask.n))
    
    for (idx, coeff) in enumerate(state)
        if abs(coeff) > 1e-8
            # Extract subsystem configuration using mask
            config = (block[idx].n & mask.n)
            config_idx = Int(config) + 1  # 1-based indexing
            eigenvals[config_idx] += abs2(coeff)
        end
    end
    
    return filter(x -> x > 1e-8, eigenvals)
end