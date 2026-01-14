using LinearAlgebra

"""
    struct MBS64Vector{bits, F <: AbstractFloat}
        vec::Vector{Complex{F}}
        space::HilbertSubspace{bits}
    end

A vector of components bounded to a `HilbertSubspace` that defines the space and basis.
It represents a general many-body state in the basis of the given `MBS64{bits}` list.

To save the memory usage, use the same `HilbertSubspace` for all `MBS64Vector`s in the same subspace.

Constructor:

    MBS64Vector(vec::Vector{Complex{F}}, space::HilbertSubspace{bits}) where {bits, F <: AbstractFloat}

Constructs an `MBS64Vector` from a vector of complex coefficients and a `HilbertSubspace`.
The length of the vector must match the dimension of the Hilbert subspace.
"""
struct MBS64Vector{bits, F <: AbstractFloat}
    vec::Vector{Complex{F}}
    space::HilbertSubspace{bits}

    function MBS64Vector(vec::Vector{Complex{F}}, space::HilbertSubspace{bits}) where {bits, F <: AbstractFloat}
        @boundscheck @assert length(vec) == length(space) "vector length mismatches Hilbert space dimension."
        new{bits, F}(vec, space)
    end
end

import Base: show, length, size, similar, getindex
import LinearAlgebra: dot
# """
#     Base.getindex(mbs_vec::MBS64Vector{bits, F}, mbs::MBS64{bits}) where {bits, F <: AbstractFloat}

# Retrieves the amplitude for a given many-body state `mbs` from the `MBS64Vector`.
# Returns `zero(Complex{F})` if the state is not found in the Hilbert subspace.
# """
# function Base.getindex(mbs_vec::MBS64Vector{bits, F}, mbs::MBS64{bits}) where {bits, F <: AbstractFloat}
#     idx = get(mbs_vec.space, mbs)
#     if idx == 0
#         return zero(Complex{F})
#     else
#         return mbs_vec.vec[idx]
#     end
# end

"""
    Base.show(io::IO, mime::MIME"text/plain", mbs_vec::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}

Displays a human-readable representation of the `MBS64Vector`, including its type parameters and the underlying vector.
"""
function Base.show(io::IO, ::MIME"text/plain", mbs_vec::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}
    print(io, "MBS64Vector{$bits, $F}, ")
    show(io, MIME("text/plain"), mbs_vec.vec)
end

"""
    length(mbs_vec::MBS64Vector)

Returns the length of the `MBS64Vector`.
"""
length(mbs_vec::MBS64Vector) = length(mbs_vec.vec)

"""
    size(mbs_vec::MBS64Vector)

Returns the size of the `MBS64Vector` as a tuple (length,).
"""
size(mbs_vec::MBS64Vector) = (length(mbs_vec.vec), )

"""
    similar(mbs_vec::MBS64Vector{bits, F})::MBS64Vector{bits, F} where {bits, F <: AbstractFloat}

Creates a new `MBS64Vector` with the same `HilbertSubspace` and element type, but with uninitialized contents.
"""
function similar(mbs_vec::MBS64Vector{bits, F})::MBS64Vector{bits, F} where {bits, F <: AbstractFloat}
    @inbounds return MBS64Vector(similar(mbs_vec.vec), mbs_vec.space)
end
# function Base.copy!(mbs_to::MBS64Vector{bits, F}, mbs_from::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}
#     @assert mbs_to.space == mbs_from.space "mbs vectors are not in the same subspace."
#     mbs_to.vec .= mbs_from.vec
#     return nothing
# end
# function Base.copy(mbs_from::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}
#     mbs_to = similar(mbs_from)
#     mbs_to.vec .= mbs_from.vec
#     return mbs_to
# end

"""
    dot(mbs_bra::MBS64Vector{bits, F}, mbs_ket::MBS64Vector{bits, F})::Complex{F}
    mbs_bra ⋅ mbs_ket

Computes the dot product between two `MBS64Vector`s.
Both vectors must belong to the same Hilbert subspace.
"""
function dot(mbs_bra::MBS64Vector{bits, F}, mbs_ket::MBS64Vector{bits, F})::Complex{F} where {bits, F <: AbstractFloat}
    @boundscheck @assert mbs_bra.space == mbs_ket.space "mbs vectors are not in the same subspace."
    return dot(mbs_bra.vec, mbs_ket.vec)
end