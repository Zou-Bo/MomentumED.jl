"""
    HilbertSubspace{bits}

Represents a Hilbert subspace defined by a list of many-body states (MBS64).
Optionally includes a dictionary for efficient state-to-index lookup.

# Fields
- `list::Vector{MBS64{bits}}`: A sorted list of MBS64 states forming the basis of the subspace.
- `length::Int64`: The dimension of the subspace, equal to `length(list)`.

Notice that dict takes lots of memory and can only accelerate searching in very very large subspace.

# Constructors
    HilbertSubspace(sorted_list::Vector{MBS64{bits}}; index_type::Type = Int64) where {bits}

Constructs a `HilbertSubspace` from a sorted list of MBS64 states.
If `dict` is true, a state-to-index dictionary is pre-built.
"""
mutable struct HilbertSubspace{bits}
    list::Vector{MBS64{bits}}
    length::Int64

    function HilbertSubspace(list::Vector{MBS64{bits}}) where {bits}
        if !issorted(list)
            @warn "The input list for HilbertSubspace is not sorted. It will be sorted now."
            sort!(list)
        end
        new{bits}(list, length(list))
    end
end


"""
    get_bits(space::HilbertSubspace{bits})::Int64

Returns the number of bits used to represent the MBS64 states in the Hilbert subspace.
"""
function get_bits(::HilbertSubspace{bits})::Int64 where{bits}
    bits
end

import Base: length, show, get
"""
    length(space::HilbertSubspace)

Returns the number of states in the Hilbert subspace (space dimension).
"""
length(space::HilbertSubspace) = space.length

"""
    Base.show(io::IO, mime::MIME"text/plain", space::HilbertSubspace{bits}) where {bits}

Displays a human-readable representation of the HilbertSubspace, including its bit size, dimension, and whether a dictionary is present.
"""
function Base.show(io::IO, ::MIME"text/plain", space::HilbertSubspace{bits}) where {bits}
    println(io, "Hilbert subspace {bits = $bits, dim = $(space.length)}")
    show(io, MIME("text/plain"), space.list)
end

"""
    Base.get(space::HilbertSubspace{bits}, mbs::MBS64{bits})::Int64

Retrieves the index of a given many-body state within the Hilbert subspace.
If a dictionary is available, it uses it for O(1) lookup; otherwise, it falls back to sorted list search.
Returns 0 if the state is not found.
"""
function Base.get(space::HilbertSubspace{bits}, mbs::MBS64{bits}) where {bits}
    searchsortedfirst(space.list, mbs)
end
@inline function index_fit(index::Integer, space::HilbertSubspace{bits}, mbs::MBS64{bits})::Bool where {bits}
    index <= space.length && index > 0 && space.list[index] == mbs
end