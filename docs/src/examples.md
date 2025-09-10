# Examples

This section describes the example notebooks included with the package.

## Example 1: FQH Laughlin State

**File**: `examples/example1_FQH_Laughlin.ipynb`

This notebook demonstrates the implementation of fractional quantum Hall effect calculations using the momentum-conserved exact diagonalization method.

### System Setup

The example implements a 3×5 triangular lattice k-mesh with:
- 15 k-points total (multiple of 3 for 1/3 filling)
- 5 electrons for ν = 1/3 filling
- Landau level projection with magnetic length scale
- Screened Coulomb interaction with form factor

### Key Functions Used

```julia
# Parameter structure with k-mesh
para = EDPara(k_list=k_list, Gk=Gk, V_int=V_int)

# State generation and momentum blocking
mbs_list = ED_mbslist(para, Ne)
blocks, block_k1, block_k2, k0number = ED_momentum_block_division(para, mbs_list)

# Scattering list generation
scat_list1 = ED_sortedScatteringList_onebody(para)
scat_list2 = ED_sortedScatteringList_twobody(para)

# KrylovKit diagonalization
energies, eigenvectors = EDsolve(blocks[1], scat_list1, scat_list2, 5)
```

### Physics Implementation

The notebook includes:
- Magnetic translation group phase factors
- Landau level projection via form factors
- Screened Coulomb interaction: `V(q) = W₀/|ql| * tanh(|qD|) * exp(-0.5*q²l²)`
- Momentum conservation enforcement

## Example 2: Bilayer FQH Halperin State

**File**: `examples/example2_BilayerFQH_Halperin.ipynb`

This notebook extends the single-layer implementation to bilayer systems with layer degrees of freedom.

### Multi-component System

The bilayer system uses:
- **Nc_hopping = 2**: Two layers for hopping
- **Nc_conserve = 1**: Single conserved quantity (total particles)
- **Layer-dependent interactions**: Different intra-layer and inter-layer couplings
- **Inter-layer tunneling**: One-body terms between layers

### Key Implementation Details

```julia
# Bilayer interaction function with layer conservation
function V_int_bilayer(k_coords_f1, k_coords_f2, k_coords_i1, k_coords_i2, cf1=1, cf2=1, ci1=1, ci2=1)
    # Layer conservation check
    if ci1 != cf1 || ci2 != cf2
        return 0.0 + 0.0im
    end
    
    # k_coords_* are tuples (kx, ky) already normalized by Gk
    q1 = k_coords_f1[1] - k_coords_i1[1]
    q2 = k_coords_f2[1] - k_coords_i2[1]
    
    # Calculate base interaction using momentum coordinates
    V_base = VFF(q1, q2)
    
    # Apply layer-dependent factors
    if ci1 != ci2  # Different layers
        ql_mag = sqrt(q1^2 + q2^2 - q1*q2) * Gl
        V_base *= exp(-ql_mag * d_l)  # Inter-layer attenuation
    end
    
    return V_base
end
```

### One-body Tunneling Terms

The notebook implements inter-layer tunneling:
```julia
# One-body Hamiltonian matrix: H_onebody[c1, c2, cc, k]
H_onebody_bilayer = zeros(ComplexF64, Nc_hopping, Nc_hopping, 1, Nk)

# Add tunneling terms: t = 0.5 * W0 between different layers
for k_idx in 1:Nk
    H_onebody_bilayer[1, 2, 1, k_idx] = 0.5 * W0  # Layer 2 → Layer 1
    H_onebody_bilayer[2, 1, 1, k_idx] = 0.5 * W0  # Layer 1 → Layer 2
end
```

### Component Indexing

The notebook demonstrates the global orbital indexing scheme:
```
Global index = k + Nk * (ch - 1) + Nk * Nch * (cc - 1)
```
Where:
- `k`: momentum index (1 to Nk)
- `ch`: hopping channel index (1 to Nc_hopping)  
- `cc`: conserved component index (1 to Nc_conserve)

## Running the Examples

To run these examples:

1. Install the package:
```julia
using Pkg
Pkg.add(url="https://github.com/Zou-Bo/MomentumED.jl")
```

2. Open the Jupyter notebooks in the `examples/` directory
3. Execute cells sequentially - each notebook is self-contained

## Key Package Features Demonstrated

Both examples showcase:

- **Momentum Block Division**: Automatic separation by total momentum quantum numbers
- **Scattering Formalism**: Efficient Hamiltonian construction using scattering terms
- **KrylovKit Integration**: Sparse matrix diagonalization with convergence control
- **Multi-component Support**: Handling of additional quantum numbers beyond momentum
- **Entanglement Analysis**: Built-in functions for computing entanglement entropy with bit masks
- **Berry Connection**: Topological analysis capabilities between kshift points

## Performance Considerations

The examples demonstrate:
- **Memory Efficiency**: Bit-based state representation reduces memory usage
- **Sparse Methods**: KrylovKit eigensolvers handle large sparse matrices efficiently
- **Block Diagonalization**: Momentum conservation significantly reduces matrix sizes
- **Multi-threading**: Parallel Hamiltonian construction when available