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

