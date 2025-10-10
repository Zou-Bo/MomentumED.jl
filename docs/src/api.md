# API Reference

## Core Data Structures

### EDPara
Central parameter container for system configuration.

**Constructor:**
- `k_list::Matrix{Int64}`: Momentum point represented by integers
- `Gk::Tuple{Int64, Int64}`: Reciprocal lattice constant in integers; zero means no Umklapp Scatter (default (0,0))
- `Nc_conserve::Int64`: Number of conserved components (default 1)
- `Nc_hopping::Int64`: Number of hopping channels (default 1)
- `V_int::Function`: Two-body interaction function
- `H_onebody::Array{ComplexF64, 4}`: One-body Hamiltonian terms, index [ch1, ch2, cc, k]


```julia
EDPara(; k_list, Gk, V_int, H_onebody=zeros(ComplexF64, 1, 1, 1, 1))
```

## Main Functions

### System Setup

#### ED_mbslist(para::EDPara, N_particles)
Generate all many-body states for given particle number.

**Arguments:**
- `para::EDPara`: System parameters
- `N_particles`: Number of particles (tuple for each-component)

**Returns:**
- `Vector{MBS64}`: List of many-body states

#### ED_momentum_block_division(para::EDPara, mbs_list)
Divide Hilbert space into momentum blocks.

**Arguments:**
- `para::EDPara`: System parameters
- `mbs_list::Vector{MBS64}`: List of many-body states

**Returns:**
- `blocks::Vector{Vector{MBS64}}: States in each momentum block
- `block_k1::Vector{Int64}`: K1 momentum for each block
- `block_k2::Vector{Int64}`: K2 momentum for each block
- `k0number::Int64`: Index of momentum=0 block

### Hamiltonian Construction

#### ED_sortedScatterList_onebody(para::EDPara)
Generate sorted list of one-body Scatter terms.

**Arguments:**
- `para::EDPara`: System parameters

**Returns:**
- `Vector{Scatter{1}}`: Sorted one-body Scatter terms

#### ED_sortedScatterList_twobody(para::EDPara)
Generate sorted list of two-body Scatter terms.

**Arguments:**
- `para::EDPara`: System parameters

**Returns:**
- `Vector{Scatter{2}}`: Sorted two-body Scatter terms

### Diagonalization

#### EDsolve(block, scat_list1, scat_list2, Neigen; kwargs)
Solve eigenvalue problem for momentum block.

**Arguments:**
- `block::Vector{MBS64}`: Many-body states in block
- `scat_list1::Vector{Scatter{1}}`: One-body Scatter terms
- `scat_list2::Vector{Scatter{2}}`: Two-body Scatter terms
- `Neigen::Int64`: Number of eigenvalues to compute

**Keyword Arguments:**
- `showtime::Bool=false`: Show timing information
- `converge_warning::Bool=true`: Show convergence warnings
- `which::Symbol=:SR`: Which eigenvalues to compute (`:SR`, `:LR`, `:LM`)

**Returns:**
- `values::Vector{Float64}`: Eigenvalues
- `vectors::Vector{Vector{ComplexF64}}`: Eigenvectors

### Analysis Functions

#### ED_etg_entropy(eigenvectors, block, partition_indices)
Calculate entanglement entropy for spatial partition.

**Arguments:**
- `eigenvectors::Vector{Vector{ComplexF64}}`: Eigenvectors
- `block::Vector{MBS64}`: Many-body states in block
- `partition_indices::Vector{Int64}`: Orbital indices for partition A

**Returns:**
- `Vector{Float64}`: Entanglement entropy for each eigenvector

#### ED_connection_integral(para, eigenvectors, blocks, block_k1, block_k2)
Calculate many-body Berry connection.

**Arguments:**
- `para::EDPara`: System parameters
- `eigenvectors::Matrix{Vector{ComplexF64}}`: All eigenvectors
- `blocks::Vector{Vector{MBS64}}`: All momentum blocks
- `block_k1::Vector{Int64}`: K1 momenta
- `block_k2::Vector{Int64}`: K2 momenta

**Returns:**
- `Matrix{ComplexF64}`: Berry connection matrix

### Utility Functions

#### ED_occupancy_to_momentum(occupancy, k_list, Gk)
Convert occupancy array to momentum representation.

**Arguments:**
- `occupancy::Vector{Int64}`: Occupation numbers
- `k_list::Matrix{Int64}`: Momentum point indices
- `Gk::Tuple{Int64, Int64}`: Grid dimensions

**Returns:**
- `Vector{ComplexF64}`: Momentum space wavefunction

#### group_momentum_pairs(k_list, Gk)
Group momentum pairs for efficient Scatter calculations.

**Arguments:**
- `k_list::Matrix{Int64}`: Momentum point indices
- `Gk::Tuple{Int64, Int64}`: Grid dimensions

**Returns:**
- `Vector{Tuple{Int64, Int64, Int64}}`: Grouped momentum pairs

## Multi-Component Systems

The package supports multi-component systems with separate conservation laws:

### Component Indexing
- Global orbital index: `i_global = k + Nk * (ch - 1) + Nk * Nch * (cc - 1)`
- `k`: momentum index
- `ch`: hopping channel index
- `cc`: conserved component index

### One-Body Terms
Accessed via `para.H_onebody[ch1, ch2, cc, k]` for hopping between channels.

### Two-Body Terms
Handled by the interaction function `para.V_int` with proper component mapping.

## Performance Tips

1. **Pre-computation**: Generate Scatter lists once and reuse for all momentum blocks
2. **Memory Management**: Use `showtime=true` to monitor memory usage
3. **Parallel Processing**: The package automatically uses multiple threads when available
4. **Sparse Methods**: KrylovKit eigensolvers are memory-efficient for large systems

## Error Handling

The package validates inputs and provides descriptive error messages for:
- Invalid momentum indices
- Inconsistent system parameters
- Memory allocation failures
- Convergence issues in eigensolvers