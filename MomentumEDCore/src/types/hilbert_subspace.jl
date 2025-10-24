

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
function get_bits(::HilbertSubspace{bits})::Integer where{bits}
    bits
end

import Base: length, show, get
length(space::HilbertSubspace) = length(space.list)

function Base.show(io::IO, ::MIME"text/plain", space::HilbertSubspace{bits}) where {bits}
    println(io, "Hilbert subspace {bits = $bits, dim = $(length(space.list))}, dict = $(!isempty(space.dict))")
    show(io, MIME("text/plain"), space.list)
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