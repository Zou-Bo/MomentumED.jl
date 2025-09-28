"""
    my_searchsortedfirst(list, i)

Search for the index of the first occurrence of element i in sorted list.
Returns 0 if element is not found.
"""
function my_searchsortedfirst(list, i)
    index = searchsortedfirst(list, i)
    if index > lastindex(list) || list[index] != i
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
- `Dict{Int, Int}`: Mapping from state integer representation to matrix index
"""
function create_state_mapping(sorted_mbs_block_list::Vector{MBS64{bits}}) where {bits}
    mapping = Dict{Int, Int}()
    for (i, state) in enumerate(sorted_mbs_block_list)
        mapping[state.n] = i
    end
    return mapping
end