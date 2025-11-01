# MomentumED API Reference

## Export List
```julia
# submodules
export EDCore, Preparation, Methods, Analysis

# main solving function
export EDsolve

# from EDCore
export EDCore, MBS64, NormalScatter, MBOperator
export HilbertSubspace, MBS64Vector, Scatter
public get_bits, get_body, make_dict!, delete_dict!
public isphysical, isupper, isnormal, isnormalupper, isdiagonal
export ED_bracket, ED_bracket_threaded
public ColexMBS64, ColexMBS64Mask

# preparation
export EDPara, ED_momentum_subspaces
export ED_sortedScatterList_onebody
export ED_sortedScatterList_twobody

# methods
public ED_HamiltonianMatrix_threaded, LinearMap

# analysis - reduced density matrix for entanglement spectrum
export PES_1rdm, PES_MomtBlocks, PES_MomtBlock_rdm
export OES_NumMomtBlocks, OES_NumMomtBlock_coef

# analysis - many-body connection
export ED_connection_step, ED_connection_gaugefixing!

# environment variables
public PRINT_RECURSIVE_MOMENTUM_DIVISION
public PRINT_TWOBODY_SCATTER_PAIRS
```
## Main EDsolve Function

```@docs
EDsolve
```

## Preparation

```@autodocs
Modules = [MomentumED.Preparation]
```
## Methods

```@autodocs
Modules = [MomentumED.Methods]
```

## Analysis

```@autodocs
Modules = [MomentumED.Analysis]
```