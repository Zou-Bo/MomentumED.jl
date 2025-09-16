# MomentumED.jl

[![CI](https://github.com/Zou-Bo/MomentumED.jl/workflows/CI/badge.svg)](https://github.com/Zou-Bo/MomentumED.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://Zou-Bo.github.io/MomentumED.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://Zou-Bo.github.io/MomentumED.jl/dev)
[![Code Style](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaFormatter/JuliaFormatter.jl)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Julia package for exact diagonalization with momentum conservation.

## Overview

This package implements exact diagonalization for quantum systems with momentum conservation. The Hamiltonian is block-diagonalized by total momentum sectors using KrylovKit for sparse matrix eigenvalue problems.

## Features

- **Momentum Block Division**: Separates Hilbert space by total momentum quantum numbers
- **Bit-based State Representation**: MBS64 type for many-body states using integer encoding
- **Scattering Formalism**: Hamiltonian construction using one-body and two-body scattering terms
- **KrylovKit Integration**: Sparse matrix diagonalization using KrylovKit's eigsolve function
- **Multi-component Systems**: Support for conserved and non-conserved component indices
- **Entanglement Calculation**: Functions for computing entanglement entropy from eigenvectors
- **Berry Connection**: Many-body Berry connection calculation for topological analysis

## Installation

Install the package from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Zou-Bo/MomentumED.jl")
```

## Usage

The package provides these main functions:

```julia
using MomentumED

# Define k-mesh for 2D system
k_list = [0 1 2 0 1 2 0 1 2 0 1 2 0 1 2;
          0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
Gk = (3, 5)

# Example system with one compotent
Nc_hopping = 1 # default number if not being configured explcitly
Nc_conserved = 1 # default number if not being configured explcitly

# Define one-body Hamiltonian (4-dim array)
H0 = ComplexF64[ #= Your Hamiltonian elements here =# 
  cospi(2 * k_list[1, k] / Gk[1]) + cospi(2 * k_list[2, k] / Gk[2]) # Simple band dispersion
  for ch_out in 1:Nc_hopping, ch_in in 1:Nc_hopping, cc in 1:Nc_conserved, k in axes(k_list, 2)
]

# Define interaction function, giving the amplitude before c†_{f1} c†_{f2} c_{i2} c_{i1}
# inputs are in order of the creation/annilation operators
function V_int(k_coords_f1, k_coords_f2, k_coords_i2, k_coords_i1, cf1=1, cf2=1, ci2=1, ci1=1)
    # k_coords_* are tuples (k1, k2)
    # each element is either the momentum (when Gk=0) or the ratio of momentum to Gk (when Gk!=0)
    # Your interaction potential here
    return 1.0 + 0.0im  # Simple constant interaction will induce no interaction term because of Fermion exchange. 
end

# Create parameter structure with keywords
para = EDPara(k_list=k_list, Gk=Gk, Nc_hopping=Nc_hopping, Nc_conserve=Nc_conserve, H_onebody=H0, V_int=V_int)

# Generate many-body states for 4 particles
mbs_list = ED_mbslist(para, 4)

# Divide into momentum blocks
blocks, block_k1, block_k2, k0number = ED_momentum_block_division(para, mbs_list)

# Generate scattering lists
scat_list1 = ED_sortedScatteringList_onebody(para)
scat_list2 = ED_sortedScatteringList_twobody(para)

# Solve first momentum block with 5 lowest eigenenergies
energies, eigenvectors = EDsolve(blocks[1], scat_list1, scat_list2, 5)

println("Ground state energy: ", energies[1])
```


## Core Components

- **EDPara**: Parameter container storing k-mesh, interaction functions, and component mappings
- **MBS64{bits}**: Many-body state representation with bit-based occupation encoding (up to 64 orbitals)
- **Scattering{N}**: Hamiltonian term representation for efficient sparse matrix construction
- **KrylovKit Integration**: Uses eigsolve for sparse eigenvalue problems with configurable convergence

## Dependencies

- **KrylovKit**: Sparse matrix eigenvalue solvers
- **LinearAlgebra, SparseArrays**: Core linear algebra functionality
- **Combinatorics**: Combinatorial utilities for state generation

## Documentation

- **[API Reference](api.md)**: Function signatures and usage
- **[Examples](examples.md)**: Example notebooks and tutorials

## Examples

Example notebooks are included:
- `examples/example1_FQH_Laughlin.ipynb`
- `examples/example2_BilayerFQH_Halperin.ipynb`

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{MomentumED.jl,
  author = {Zou, Bo},
  title = {{MomentumED.jl}: A Julia Package for Exact Diagonalization in Momentum Basis},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Zou-Bo/MomentumED.jl}}
}
```