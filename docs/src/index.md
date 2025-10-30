# MomentumED.jl

A Julia package for momentum-conserved exact diagonalization of quantum many-body systems.

## Overview

This project provides a powerful and flexible framework for exact diagonalization (ED) of quantum many-body systems, with a special focus on 2D systems where momentum is a conserved quantity. The project is organized into two main packages:

- **`EDCore.jl`**: A lightweight, dependency-free core library that provides the fundamental building blocks for generic ED calculations. It defines abstract representations for many-body states (`MBS64`, `MBS64Vector`), operators (`Scatter`, `MBOperator`), and Hilbert subspaces (`HilbertSubspace`).
- **`MomentumED.jl`**: A high-level application package that uses `EDCore.jl` to implement specialized tools for momentum-conserved systems. It provides functions to set up system parameters, generate momentum-block-diagonal Hamiltonians, solve for eigenstates, and perform post-calculation analysis. It is made with the assumptions that the interaction is symmetric in switching the two vertices and that the Hamiltonian is hermitian and momentum-conserving; if you are doing a specific problem that fails to fit in these assumptions, consider making your dedicated initiating process using `EDCore.jl`.

`MomentumED.jl` uses `KrylovKit.jl` to diagonalize the Hamiltonian and find the eigenvalues and eigenvectors.

`MomentumED.jl` also provides many-body state analysis methods to compute one-body reduced density matrix, many-body connection and many-body Chern number (NTW invariant), and particle-/orbital- reduced density matrix, entanglement entropy.

## Features

### `EDCore.jl` (Core Library)

- **Bit-based State Representation**: `MBS64{bits}` for efficient representation of many-body states in up to 64-orbital systems.
- **Generic Operator Algebra**: `Scatter{N}` and `MBOperator` types for constructing and manipulating N-body operators.
- **Abstract Hilbert Space**: `HilbertSubspace` and `MBS64Vector` to manage basis states and eigenvectors.

### `MomentumED.jl` (High-Level Application)

- **Momentum-Space ED**: Tools to build and solve Hamiltonians that are block-diagonal in total momentum.
- **System Setup**: The `EDPara` struct to easily configure system parameters like the k-list, interaction potential, and component structure.
- **Automated Basis Generation**: `ED_momentum_subspaces` function to automatically generate basis states for each momentum sector.
- **Hamiltonian Construction**: `ED_sortedScatterList_...` functions to generate one- and two-body operators from the system parameters.
- **Eigensolver**: A simple `EDsolve` interface that wraps `KrylovKit.jl` for efficient sparse diagonalization.
- **Many-Body Analysis**: Built-in functions for calculating one-body reduced density matrices, particle/orbital entanglement spectrum, and many-body Berry connection.

## Installation

This project contains two packages: `EDCore` and `MomentumED`. Since they are not yet registered and `MomentumED` depends on the local version of `EDCore`, the installation process requires you to clone the repository first.

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Zou-Bo/MomentumED.jl
    cd MomentumED.jl
    ```

2.  **Install the local packages:**
    Start Julia in the `MomentumED.jl` directory. Press `]` to enter the package manager, then run:
    ```julia
    pkg> dev ./EDCore
    pkg> dev ./MomentumED
    ```
    This will install the packages from their local subdirectories, allowing Julia to correctly resolve the dependency between them.

Once the packages are registered in the Julia General Registry, you will be able to install them simply with `Pkg.add("MomentumED")`.


## Structure and Dependency

The project is organized into two main packages:

- **`EDCore/`**: The core library providing generic, low-level ED tools.
    - `EDCore/src/types/`: Defines core data structures like `MBS64`, `MBS64Vector`, `Scatter`, `MBOperator`, and `HilbertSubspace`.
    - `EDCore/src/EDCore.jl`: Implements the core functionalities and operator algebra.
- **`MomentumED/`**: The high-level package for momentum-space ED.
    - `MomentumED/src/preparation/`: Functions for setting up calculations (parameter initialization, basis generation).
    - `MomentumED/src/method/`: Hamiltonian construction and eigensolving methods.
    - `MomentumED/src/analysis/`: Functions for post-calculation analysis (RDM, Berry connection, etc.).

The dependency tree for the packages is as follows:

```
MomentumED
├── EDCore
│   ├── LinearAlgebra
│   └── Combinatorics
├── KrylovKit
├── LinearAlgebra
└── SparseArrays
```


## Usage

The package provides these main functions:

```julia
using MomentumED

# Define k-mesh for 2D system
k_list = [0 1 2 0 1 2 0 1 2 0 1 2 0 1 2;
          0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
Gk = (3, 5) # momenta mod Gk are conserved

# Example system with one compotent
Nc_hopping = 1 # default number if not being configured explcitly
Nc_conserve = 1 # default number if not being configured explcitly

# Define one-body Hamiltonian (4-dim array)
H0 = ComplexF64[ #= Your Hamiltonian elements here =# 
  cospi(2 * k_list[1, k] / Gk[1]) + cospi(2 * k_list[2, k] / Gk[2]) # Simple band dispersion
  for ch_out in 1:Nc_hopping, ch_in in 1:Nc_hopping, cc in 1:Nc_conserve, k in axes(k_list, 2)
]

# Define interaction function, giving the amplitude before c†_{f1} c†_{f2} c_{i2} c_{i1}
# inputs are in order of the creation/annilation operators
function V_int(k_coords_f1, k_coords_f2, k_coords_i2, k_coords_i1, cf1=1, cf2=1, ci2=1, ci1=1)
    # k_coords_* are tuples (k1, k2)
    # each tuple element is either the momentum (when Gk=0) or the ratio of momentum to Gk (when Gk!=0)
    # Your interaction potential here
    return 1.0 + 0.0im  # Simple constant interaction will induce no interaction term because of Fermion exchange. 
end

# Create parameter structure with keywords
para = EDPara(k_list=k_list, Gk=Gk, Nc_hopping=Nc_hopping, Nc_conserve=Nc_conserve, H_onebody=H0, V_int=V_int)

# Generate momentum subspaces for a system of 4 electrons in the first (and only) conserved component
subspaces, subspace_k1, subspace_k2 = ED_momentum_subspaces(para, (4,))

# Generate Scatter lists
scat_list1 = ED_sortedScatterList_onebody(para)
scat_list2 = ED_sortedScatterList_twobody(para)

# Solve the first momentum block for the 5 lowest eigenenergies
energies, eigenvectors = EDsolve(subspaces[1], scat_list1, scat_list2; N=5)

println("Total momentum: (", subspace_k1[1], ", ",  subspace_k2[1],")  Ground state energy: ", energies[1])
```

## API Reference

The public API is organized into several categories, reflecting the typical workflow.

### Core Types (from `EDCore.jl`)

- **`MBS64{bits}`**: Represents a many-body state using bit encoding.
- **`HilbertSubspace`**: A container for a list of basis states (`MBS64`) that form a subspace.
- **`MBS64Vector`**: Represents an eigenvector, associating a raw vector with its `HilbertSubspace`.
- **`Scatter{N}`**: Represents an N-body operator term.
- **`MBOperator`**: A collection of `Scatter` terms that defines a full many-body operator like the Hamiltonian.

### System Preparation

- **`EDPara`**: A struct that holds all parameters for a calculation (k-list, interaction functions, etc.).
- **`ED_momentum_subspaces`**: Generates the basis states, automatically partitioning them into `HilbertSubspace` objects based on total momentum.
- **`ED_sortedScatterList_onebody`**: Creates the one-body part of the Hamiltonian as a list of `Scatter{1}` terms.
- **`ED_sortedScatterList_twobody`**: Creates the two-body part of the Hamiltonian as a list of `Scatter{2}` terms.

### Eigensolver

- **`EDsolve`**: The main function to find the lowest eigenvalues and eigenvectors of a Hamiltonian within a given subspace. Supports both sparse matrix and matrix-free (LinearMap) methods.

### Analysis

- **Entanglement Spectrum**:
    - `PES_1rdm`, `PES_MomtBlocks`, `PES_MomtBlock_rdm`: Functions for calculating the particle entanglement spectrum.
    - `OES_NumMomtBlocks`, `OES_NumMomtBlock_coef`: Functions for calculating the orbital entanglement spectrum.
- **Topological Properties**:
    - `ED_connection_step`, `ED_connection_gaugefixing!`: Functions for calculating the many-body Berry connection, used to find topological invariants like the Chern number.
- **Expectation Values**:
    - `ED_bracket`, `ED_bracket_threaded`: Low-level functions to compute the expectation value `<ψ₁|O|ψ₂>`.


### Multi-Component Systems

The package supports systems with multiple components:
- Conserved components
- Non-conserved components (also called hopping components in the code because usually there're hopping terms between them)

Using conserved components allows you to assign the particle number of each component when generating many-body state(mbs) list. The package will not check if the particle number is really conserved when generating Scatter list from given one-body term and two-body interaction. However, if a Scatter term scatters a mbs state outside the provided mbs list, the EDsolve() function will throw a "NonConserved" error.

## Performance

The package is optimized for performance:
- **Memory efficiency**: Bit encoding and sparse matrix methods
- **Parallel processing**: Multi-threaded when generating sparse Hamiltonian matrix.
- **Block diagonalization**: Significantly reduced matrix sizes

Typical performance explodes exponentially with system size and particle number.

## Examples

The package includes example notebooks in the "/examples/" folder.

## Documentation

Comprehensive documentation is available at:
- [Documentation](https://Zou-Bo.github.io/MomentumED.jl/) (In construction)

## Contribution

Let me know if you have any problems in using the package or find bugs.

## License

This package is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, you might cite:

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

---

This documentation is organized into two main parts, corresponding to the two packages in this repository:

- **`EDCore.jl`**: Contains the documentation for the low-level, generic building blocks of the exact diagonalization framework. This includes tutorials, manuals, and API references for the core data structures and their algebra.
- **`MomentumED.jl`**: Contains the documentation for the high-level functionalities specific to momentum-conserved systems. This section will provide guides and API references for setting up calculations, solving for eigenstates, and performing analyses within the momentum-space formalism.
