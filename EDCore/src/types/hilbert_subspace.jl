

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
    create_state_mapping(mbs_list, index_type = Int64)

Create a dictionary mapping from MBS64 states to their indices for efficient lookup.

# Arguments
- `mbs_list::Vector{MBS64{bits}}`: Sorted list of MBS64 basis states.
- `index_type::Type`: The integer type to use for the indices in the mapping (e.g., `Int64`, `Int32`).

# Returns
- `Dict{MBS64{bits}, index_type}`: Mapping from state integer representation to matrix index.
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
    HilbertSubspace{bits}

Represents a Hilbert subspace defined by a list of many-body states (MBS64).
Optionally includes a dictionary for efficient state-to-index lookup.

# Fields
- `list::Vector{MBS64{bits}}`: A sorted list of MBS64 states forming the basis of the subspace.
- `dict::Dict{MBS64{bits}, <: Integer}`: An optional dictionary mapping MBS64 states to their 1-based indices in `list`.

Notice that dict takes lots of memory and can only accelerate searching in very very large subspace.

# Constructors
    HilbertSubspace(sorted_list::Vector{MBS64{bits}}; dict::Bool = false, index_type::Type = Int64) where {bits}

Constructs a `HilbertSubspace` from a sorted list of MBS64 states.
If `dict` is true, a state-to-index dictionary is pre-built.
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


"""
    idtype(space::HilbertSubspace)::Type

Returns the type of the index used in the HilbertSubspace's dictionary.
"""
function idtype(space::HilbertSubspace)::Type
    valtype(space.dict)
end

"""
    get_bits(space::HilbertSubspace{bits})::Integer

Returns the number of bits used to represent the MBS64 states in the Hilbert subspace.
"""
function get_bits(::HilbertSubspace{bits})::Integer where{bits}
    bits
end

import Base: length, show, get
"""
    length(space::HilbertSubspace)

Returns the number of states in the Hilbert subspace (space dimension).
"""
length(space::HilbertSubspace) = length(space.list)

"""
    Base.show(io::IO, mime::MIME"text/plain", space::HilbertSubspace{bits}) where {bits}

Displays a human-readable representation of the HilbertSubspace, including its bit size, dimension, and whether a dictionary is present.
"""
function Base.show(io::IO, ::MIME"text/plain", space::HilbertSubspace{bits}) where {bits}
    println(io, "Hilbert subspace {bits = $bits, dim = $(length(space.list))}, dict = $(!isempty(space.dict))")
    show(io, MIME("text/plain"), space.list)
end

"""
    make_dict!(space::HilbertSubspace; index_type::Type = idtype(space))

Creates or recreates the state-to-index dictionary for the given HilbertSubspace.
This can be useful if the `dict` field was not initialized or was emptied.
"""
function make_dict!(space::HilbertSubspace; index_type::Type = idtype(space))
    space.dict = create_state_mapping(space.list, index_type)
end

"""
    delete_dict!(space::HilbertSubspace)

Deletes (empties) the state-to-index dictionary from the HilbertSubspace to free up memory.
"""
function delete_dict!(space::HilbertSubspace)
    empty!(space.dict)
end

"""
    Base.get(space::HilbertSubspace{bits}, mbs::MBS64{bits})::Int64

Retrieves the index of a given many-body state within the Hilbert subspace.
If a dictionary is available, it uses it for O(1) lookup; otherwise, it falls back to sorted list search.
Returns 0 if the state is not found.
"""
function Base.get(space::HilbertSubspace{bits}, mbs::MBS64{bits})::Int64 where {bits}
    if length(space.dict) != 0
        return get(space.dict, mbs, 0)
    else
        return my_searchsortedfirst(space.list, mbs)
    end
end

"""
    get_from_list(space::HilbertSubspace{bits}, mbs::MBS64{bits})::Int64

Retrieves the index of a given many-body state within the Hilbert subspace by searching the state list.
This method does not use the dictionary, even if it exists.
Returns 0 if the state is not found.
"""
function get_from_list(space::HilbertSubspace{bits}, mbs::MBS64{bits})::Int64 where {bits}
    my_searchsortedfirst(space.list, mbs)
end

"""
    get_from_dict(space::HilbertSubspace{bits}, mbs::MBS64{bits})::Int64

Retrieves the index of a given many-body state within the Hilbert subspace using the pre-built dictionary.
This method requires the dictionary to be non-empty.
Returns 0 if the state is not found.
"""
function get_from_dict(space::HilbertSubspace{bits}, mbs::MBS64{bits})::Int64 where {bits}
    @boundscheck @assert length(space.dict) != 0 "Trying to find index from empty dictionary of a Hilbert space."
    get(space.dict, mbs, 0)
end