
using LinearAlgebra

"""
    my_searchsortedfirst(sorted_list, x)

Search for the index of the first occurrence of element x in sorted list.
Returns 0 if element is not found.
"""
function my_searchsortedfirst(sorted_list, x)::Int64
    index = searchsortedfirst(sorted_list, x)
    if index > lastindex(sorted_list) || sorted_list[index] != x
        return 0
    else
        return index
    end
end

"""
    create_state_mapping(sorted_mbs_block_list)

Create a dictionary mapping from MBS64 states to their indices for O(1) lookup.
This eliminates the my_searchsortedfirst bottleneck by providing direct state-to-index mapping.

# Arguments
- `sorted_mbs_block_list::Vector{MBS64{bits}}`: Sorted list of MBS64 basis states

# Returns
- `Dict{MBS64{bits}, Int}`: Mapping from state integer representation to matrix index
"""
function create_state_mapping(mbs_list::Vector{MBS64{bits}}, index_type::Type = Int64) where {bits}

    @assert index_type <: Integer "index_type should be a Integer."
    @assert length(mbs_list) <= typemax(index_type) "index type $index_type cannot cover the length of given MBS64 list."
    @assert issorted(mbs_list) "Using unsorted mbs list will cause failure in ensuring the scatttering terms in upper triangular."

    mapping = Dict{MBS64{bits}, index_type}()
    for (i, state) in enumerate(mbs_list)
        mapping[state] = i
    end
    return mapping
end

"""

"""
mutable struct HilbertSubspace{bits}
    list::Vector{MBS64{bits}}
    dict::Dict{MBS64{bits}, <: Integer}

    function HilbertSubspace(sorted_list::Vector{MBS64{bits}}; dict::Bool = false, index_type::Type = Int64) where {bits}
        @assert issorted(sorted_list)
        if dict
            new{bits}(sorted_list, create_state_mapping(sorted_list, index_type))
        else
            new{bits}(sorted_list, Dict{MBS64{bits}, index_type}())
        end
    end
end

function idtype(space::HilbertSubspace)::Type
    valtype(space.dict)
end
function make_dict!(space::HilbertSubspace; index_type::Type = idtype(space))
    space.dict = create_state_mapping(space.list, index_type)
end
function delete_dict!(space::HilbertSubspace)
    empty!(space.dict)
end

function Base.get(space::HilbertSubspace{bits}, mbs::MBS64{bits})::Int64 where {bits}
    if length(space.dict) != 0
        return get(space.dict, mbs, 0)
    else
        return my_searchsortedfirst(space.list, mbs)
    end
end
function get_from_list(space::HilbertSubspace{bits}, mbs::MBS64{bits})::Int64 where {bits}
    my_searchsortedfirst(space.list, mbs)
end
function get_from_dict(space::HilbertSubspace{bits}, mbs::MBS64{bits})::Int64 where {bits}
    @boundscheck @assert length(space.dict) != 0 "Trying to find index from empty dictionary of a Hilbert space."
    get(space.dict, mbs, 0)
end

Base.length(space::HilbertSubspace) = length(space.list)

function Base.show(io::IO, ::MIME"text/plain", space::HilbertSubspace{bits}) where {bits}
    println(io, "Hilbert subspace {dim = $bits}, dict = $(!isempty(space.dict))")
    show(io, MIME"text/plain"(), space.list)
end

"""
    struct MBS64Vector{bits, F<: Real}
        vec::Vector{Complex{F}}
        space::HilbertSubspace{bits}
    end

A vector of components bounded to a hash table that specifies the Hilbert subspace.
It represents a general many-body states in the basis of given MBS64{bits} list.

To save the memory usage, all MBS64Vectors in the same subspace use the same hash table.
"""
struct MBS64Vector{bits, F <: Real}
    vec::Vector{Complex{F}}
    space::HilbertSubspace{bits}

    function MBS64Vector(vec::Vector{Complex{F}}, space::HilbertSubspace{bits}) where {bits, F <: AbstractFloat}
        @boundscheck @assert length(vec) == length(space) "vector length mismatches Hilbert space dimension."
        new{bits, F}(vec, space)
    end
end

import Base: length, similar, copy, size
import LinearAlgebra: dot
function Base.show(io::IO, ::MIME"text/plain", mbs_vec::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}
    print(io, "MBS64Vector{$bits, $F}, ")
    show(io, MIME"text/plain"(), mbs_vec.vec)
end
length(mbs_vec::MBS64Vector) = length(mbs_vec.vec)
size(mbs_vec::MBS64Vector) = (length(mbs_vec.vec), )
function similar(mbs_vec::MBS64Vector{bits, F})::MBS64Vector{bits, F} where {bits, F <: AbstractFloat}
    @inbounds return MBS64Vector(similar(mbs_vec.vec), mbs_vec.space)
end
function copy!(mbs_to::MBS64Vector{bits, F}, mbs_from::MBS64Vector{bits, F}) where {bits, F <: AbstractFloat}
    @assert mbs_to.space == mbs_from.space "mbs vectors are not in the same subspace."
    mbs_to.vec .= mbs_from.vec
    return nothing
end
function dot(mbs_bra::MBS64Vector{bits, F}, mbs_ket::MBS64Vector{bits, F})::Complex{F} where {bits, F <: AbstractFloat}
    @boundscheck @assert mbs_bra.space == mbs_ket.space "mbs vectors are not in the same subspace."
    return dot(mbs_bra.vec, mbs_ket.vec)
end