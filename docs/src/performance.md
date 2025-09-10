# Performance Guide

This document describes performance characteristics and optimization strategies implemented in the package.

## Memory Efficiency

### Bit-Based State Representation

The package uses bit-encoded many-body states to minimize memory usage:

```julia
struct MBS64{bits} <: Integer
    n::UInt64  # 64-bit integer for up to 64 orbitals
end
```

**Memory Usage**: 
- Each MBS64 state: 8 bytes (regardless of system size)
- N-orbital system: N × 8 bytes for complete basis
- Compared to explicit state vectors: significant memory reduction

### Sparse Matrix Storage

Hamiltonian matrices use sparse storage.

## Computational Efficiency

### Momentum Block Division

The package reduces computational complexity through momentum conservation:

```julia
# Divide Hilbert space into momentum blocks
blocks, block_k1, block_k2, k0number = ED_momentum_block_division(para, mbs_list)
```

### Scattering Formalism

Hamiltonian construction uses scattering terms:

```julia
# Generate sorted scattering lists
scat_list1 = ED_sortedScatteringList_onebody(para)
scat_list2 = ED_sortedScatteringList_twobody(para)
```

**Advantages**:
- Avoids explicit matrix element storage
- Efficient term generation and sorting
- Automatic Hermitian symmetry handling

## KrylovKit Integration

### Sparse Eigenvalue Solvers

The package uses KrylovKit for efficient sparse diagonalization:

```julia
# Krylov subspace methods
vals, vecs, info = eigsolve(H, vec0, N_eigen, :SR, ishermitian=true)
```

**Performance Benefits**:
- Only computes requested eigenvalues (not full spectrum)
- Memory-efficient for large sparse matrices
- Iterative methods with configurable convergence

### Convergence Control

```julia
# Convergence monitoring and warnings
if !(info.converged == true || info.converged == 1)
    @warn "Eigensolver did not converge. Residual norm: $(info.normres)"
end
```

**Features**:
- Automatic convergence detection
- Residual norm monitoring
- Configurable tolerance levels

## Multi-threading

### Parallel Hamiltonian Construction

The package supports multi-threaded matrix construction:

```julia
# Threaded sparse matrix assembly
H = HmltMatrix_threaded(block, scat_list1, scat_list2, multi_thread)
```

**Implementation**:
- Julia Threads.@threads for parallelization thread-local matrix construction
- Final assembly with thread safety

**Performance Scaling**:
- Linear scaling with number of CPU cores
- Automatic thread pool management


### Efficient Data Structures

Key data structure choices:
- **Vectors** for dynamic collections
- **Tuples** for fixed-size indexing
- **ComplexF64** for quantum amplitudes
- **UInt64** for bit operations

## Algorithmic Optimizations

### Sorted Scattering Lists

Scattering terms are sorted and merged.


**Benefits**:
- Eliminates duplicate terms
- Enables efficient term lookup
- Improves cache locality


## Performance Monitoring

### Timing Information

The package includes timing capabilities:

```julia
# Enable timing output
energies, eigenvectors = EDsolve(block, scat_list1, scat_list2, N_eigen; showtime=true)
```

**Output**:
- Matrix construction time
- Diagonalization time

### Convergence Monitoring

Track eigensolver convergence:

```julia
# Enable convergence warnings
energies, eigenvectors = EDsolve(block, scat_list1, scat_list2, N_eigen; converge_warning=true)
```

**Features**:
- Residual norm reporting
- Convergence failure warnings
- Automatic fallback strategies

## Scaling Characteristics

### System Size Scaling

The package scales with:
- **Orbitals**: Up to 64 (bit encoding limit)
- **Particles**: Limited by combinatorial growth
- **Momentum blocks**: Number depends on system symmetry

### Memory Scaling

Typical memory usage:
- **State storage**: O(N_states × 8 bytes)
- **Hamiltonian**: O(N_nonzero × sizeof(ComplexF64))
- **Eigenvectors**: O(N_states × N_eigen × sizeof(ComplexF64))

### Computational Scaling

Diagonalization complexity:
- **Full diagonalization**: O(d³) where d is block dimension
- **Krylov methods**: O(d × N_eigen × iterations)
- **Matrix construction**: O(N_terms × N_states)

## Configuration Options

### Thread Control

Configure multi-threading usage:

```julia
# Enable/disable multi-threading
energies, eigenvectors = EDsolve(block, scat_list1, scat_list2, N_eigen; multi_thread=true)
```

### Convergence Settings

Control eigensolver behavior:

```julia
# Convergence and warning settings
energies, eigenvectors = EDsolve(block, scat_list1, scat_list2, N_eigen; 
                                  converge_warning=true, showtime=false)
```

### Memory Optimization

Optimize memory usage patterns:
- Use appropriate block sizes
- Limit number of requested eigenvalues
- Enable memory-efficient sparse formats