# Physics Implementation

This document describes the physics concepts implemented in the package.

## Momentum Conservation

The package exploits translational symmetry to block-diagonalize the Hamiltonian by total momentum sectors.
A momemtum-unconserved Hamiltonian will cause errors.

### Block Division

The Hilbert space is divided into blocks with identical total momentum:
```julia
blocks, block_k1, block_k2, k0number = ED_momentum_block_division(para, mbs_list)
```

This reduces the eigenvalue problem size from the full Hilbert space to individual momentum blocks.

## Many-Body State Representation

### Bit Encoding

Many-body states are represented using bit strings:
```julia
struct MBS64{bits} <: Integer
    n::UInt64
end
```

- Each bit represents orbital occupation (0 = empty, 1 = occupied)
- `bits` parameter determines maximum system size (up to 64 orbitals)
- Efficient bitwise operations for state manipulation

### State Operations

The MBS64 type provides basic operations for bit manipulation:
```julia
# Create a 4-bit state with orbitals 1 and 3 occupied
mbs = MBS64{4}(0b1010)  # Binary representation

# Access the underlying integer
state_value = mbs.n
```

## Hamiltonian Construction

### Scattering Formalism

The Hamiltonian is constructed using abstract N-body scattering terms:


```julia
struct Scattering{N}
    Amp::ComplexF64
    out::NTuple{N, Int64}
    in::NTuple{N, Int64}
end

# One-body: V * c†_i c_j
s1 = Scattering(1.0-1.0im, 1, 2)  # Creates c†_1 c_2 term

# Two-body: V * c†_i1 c†_i2 c_j2 c_j1  
s2 = Scattering(0.5, 1, 2, 4, 3)  # Creates c†_1 c†_2 c_4 c_3 term
```


### Scattering List Generation

The package generates sorted lists of scattering terms:

```julia
# One-body terms from EDPara.H_onebody matrix
scat_list1 = ED_sortedScatteringList_onebody(para)

# Two-body terms from interaction function
scat_list2 = ED_sortedScatteringList_twobody(para)
```


## Multi-Component Systems

### Component Indexing

The package handles additional quantum numbers beyond momentum:

```julia
# Global orbital index mapping
global_index = k + Nk * (ch - 1) + Nk * Nch * (cc - 1)
```

Where:
- `k`: momentum index (1 to Nk)
- `ch`: hopping channel index (1 to Nc_hopping)
- `cc`: conserved component index (1 to Nc_conserve)

### One-body Terms

Stored in 4D array: `H_onebody[ch1, ch2, cc, k]`
- Represents hopping between channels `ch2 → ch1`
- Conserves component index `cc`
- Momentum index `k` is conserved

### Two-body Interactions

Custom interaction functions receive momentum coordinates and component indices:
```julia
function V_int(k_coords_f1, k_coords_f2, k_coords_i1, k_coords_i2, cf1=1, cf2=1, ci1=1, ci2=1)
    # k_coords_*: Tuple{Float64, Float64} momentum coordinates
    # cf1, cf2: final component indices
    # ci1, ci2: initial component indices
    # Return complex scattering amplitude
end
```

The function receives momentum coordinates already shifted and normalized by the Gk vector.

## Entanglement Entropy

The package computes entanglement entropy for bit-position bipartitions:

```julia
# Create bit mask for subsystem A (first 2 bits)
bit_mask = MBS64{bits}(0b0011)

# Compute entanglement entropy
entropy = ED_etg_entropy(block, eigenvector, bit_mask)

# Rényi entropy of order 2
entropy_2 = ED_etg_entropy(block, eigenvector, bit_mask, alpha=2.0)
```

### Implementation

- Uses bit masking to identify subsystem A and B
- Direct computation without storing reduced density matrix
- Computes von Neumann entropy: `S = -Tr(ρ_A log ρ_A)`
- Supports Rényi entropy: `S_α = (1-α)^(-1) log Tr(ρ_A^α)`
- Automatically computes RDM for smaller subsystem for efficiency

## Berry Connection

For topological analysis, the package computes many-body Berry connections between kshift points:

```julia
# Calculate geometric phase between two kshift points
geometric_phase = ED_connection_integral(kshift1, kshift2, ψ1, ψ2, momentum_axis_angle)

# Calculate Berry connection (average connection)
berry_conn = ED_connection_integral(kshift1, kshift2, ψ1, ψ2, momentum_axis_angle; average_connection=true)
```

### Implementation

- Computes complex inner product: `⟨ψ2|ψ1⟩`
- Extracts geometric phase: `φ = arg(⟨ψ2|ψ1⟩)`
- Berry connection: `A = φ / ||δk||`
- Accounts for non-orthogonal momentum space coordinates
- Handles numerical stability for very close points

## KrylovKit Integration

The package uses KrylovKit for sparse matrix diagonalization:

```julia
# Sparse matrix construction from scattering terms
H = HmltMatrix_threaded(block, scat_list1, scat_list2, multi_thread)

# KrylovKit eigensolve
vals, vecs, info = eigsolve(H, vec0, N_eigen, :SR, ishermitian=true)
```

### Convergence Control

The eigensolver includes:
- Random initial vector for good convergence
- Convergence warning system
- Support for different eigenvalue selections (`:SR`, `:LR`, `:LM`)
- Automatic handling of convergence failures

## Momentum Mesh Shift

The package supports twisted boundary conditions via k-mesh shift:

```julia
# Apply momentum shift to interaction amplitudes
amp = int_amp(i1, i2, f1, f2, para; kshift=(0.1, 0.2))
```

### Implementation

- Shifts momentum coordinates: `(k_list .+ kshift) ./ Gk`
- Maintains backward compatibility with default `(0.0, 0.0)`
- Affects all momentum-dependent calculations in interaction functions