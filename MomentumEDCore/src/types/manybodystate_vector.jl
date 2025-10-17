
using LinearAlgebra

"""
    struct MBS64Vector{bits, F<: Real}
        vec::Vector{Complex{F}}
        space::HilbertSubspace{bits}
    end

A vector of components bounded to a hash table that specifies the Hilbert subspace.
It represents a general many-body states in the basis of given MBS64{bits} list.

To save the memory usage, all MBS64Vectors in the same subspace use the same hash table.
"""
struct MBS64Vector{bits, F <: AbstractFloat}
    vec::Vector{Complex{F}}
    space::HilbertSubspace{bits}

    function MBS64Vector(vec::Vector{Complex{F}}, space::HilbertSubspace{bits}) where {bits, F <: AbstractFloat}
        @boundscheck @assert length(vec) == length(space) "vector length mismatches Hilbert space dimension."
        new{bits, F}(vec, space)
    end
end

import Base: show, length, size, similar
import LinearAlgebra: dot
function Base.show(io::IO, ::MIME"text/plain", mbs_vec::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}
    print(io, "MBS64Vector{$bits, $F}, ")
    show(io, MIME("text/plain"), mbs_vec.vec)
end
length(mbs_vec::MBS64Vector) = length(mbs_vec.vec)
size(mbs_vec::MBS64Vector) = (length(mbs_vec.vec), )
function similar(mbs_vec::MBS64Vector{bits, F})::MBS64Vector{bits, F} where {bits, F <: AbstractFloat}
    @inbounds return MBS64Vector(similar(mbs_vec.vec), mbs_vec.space)
end
function Base.copy!(mbs_to::MBS64Vector{bits, F}, mbs_from::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}
    @assert mbs_to.space == mbs_from.space "mbs vectors are not in the same subspace."
    mbs_to.vec .= mbs_from.vec
    return nothing
end
function Base.copy(mbs_from::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}
    mbs_to = similar(mbs_from)
    mbs_to.vec .= mbs_from.vec
    return mbs_to
end
function dot(mbs_bra::MBS64Vector{bits, F}, mbs_ket::MBS64Vector{bits, F})::Complex{F} where {bits, F <: AbstractFloat}
    @boundscheck @assert mbs_bra.space == mbs_ket.space "mbs vectors are not in the same subspace."
    return dot(mbs_bra.vec, mbs_ket.vec)
end