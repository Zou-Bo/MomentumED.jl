# EDCore API Reference

## Many-Body Basis

### Types
```@docs
MBS64
Base.show(::IO, ::MBS64)
```

### Functions
```@docs
get_bits(::MBS64{bits}) where{bits}
isphysical
occ_list
make_mask64
MBS64(bits, occ_list)
MBS64_complete
Base.:*(::MBS64{b}, ::MBS64{b}) where {b}
Base.:+(::MBS64{b}, ::MBS64{b}) where {b}
Base.isless(::MBS64{b}, ::MBS64{b}) where {b}
Base.:(==)(::MBS64{b1}, ::MBS64{b2}) where {b1, b2}
Base.hash(::MBS64)
isoccupied
Base.isempty(::MBS64{bits}, ::UInt64) where {bits}
occupy!
Base.empty!(::MBS64{bits}, ::UInt64) where {bits}
flip!
scat_occ_number
``` 

### Generating Sorted List
```@docs
ColexMBS64
ColexMBS64Mask
```


## Hilbert Subspace

### Types
```@docs
HilbertSubspace
Base.show(::IO, ::MIME"text/plain", ::HilbertSubspace)
```

### Functions
```@docs
idtype
get_bits(::HilbertSubspace)
Base.length(::HilbertSubspace)
make_dict!
delete_dict!
Base.get(::HilbertSubspace{bits}, ::MBS64{bits}) where{bits}
get_from_list
get_from_dict
```

## Many-Body Vector

### Types
```@docs
MBS64Vector
Base.show(::IO, ::MIME"text/plain", ::mbs_vec::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}
```

### Functions
```@docs
Base.length(::MBS64Vector)
Base.size(::MBS64Vector)
Base.similar(::MBS64Vector)
LinearAlgebra.dot(::MBS64Vector{bits, F}, ::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}
```

## Scattering Term

### Types
```@docs
Scatter
NormalScatter
Scatter(V, outin::Int64...)
Base.show(::IO, ::Scatter{N}) where {N}
```

### Functions
```@docs
get_body
isnormal
isnormalupper
isdiagonal
Base.adjoint(::Scatter{N}) where {N}
Base.isless(::Scatter{N1}, ::Scatter{N2}) where {N1, N2}
Base.:(==)(::Scatter{N1}, ::Scatter{N2}) where {N1, N2}
Base.:+(::Scatter{N}, ::Scatter{N}) where {N}
Base.:*(::Number, ::Scatter{N}) where {N}
sort_merge_scatlist
```

## Many-Body Operator

### Types
```@docs
MBOperator
Base.show(::IO, ::MBOperator)
```

### Functions
```@docs
isupper
Base.adjoint(::MBOperator)
LinearAlgebra.adjoint!(::MBOperator)
```

## Multiplication and Bracket

### Functions
```@docs
Base.:*(::Scatter, ::MBS64)
Base.:*(::MBS64, ::Scatter)
Base.:*(::MBOperator, ::MBS64Vector)
ED_bracket
ED_bracket_threaded
``` 

## End