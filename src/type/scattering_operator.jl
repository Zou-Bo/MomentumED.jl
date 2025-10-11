
# abstract type AbstractOperator end

"""
    struct MBOperator{F <: Real} <: AbstractOperator
        scats::Vector{<:Scatter}
        upper_triangular::Bool
    end

Expand an operator in a list of Scatter terms.
When upper_triangular is true, the list contain only upper-triangular terms.

In construction of operators, all the Scatter terms automatically passed checking isnormal() or isnormalupper().
Terms of the same scatter-in and -out states are merged. 
"""
struct MBOperator{F <: AbstractFloat}
    scats::Vector{<:Scatter}
    upper_hermitian::Bool

    function MBOperator{F}(scats::Vector{<:Scatter}...; 
        upper_hermitian::Bool) where {F<:AbstractFloat}
        allscats = reduce(vcat, scats)
        if upper_hermitian
            @assert all(isnormalupper,  allscats) "Scatter terms should all in normal order and in the upper triangular."
        else
            @assert all(isnormal, allscats) "Scatter terms should all in normal order."
        end
        sort_merge_scats = sort_merge_scatlist(allscats; check_normal=false, check_normalupper=false)
        return new{F}(sort_merge_scats, upper_hermitian)
    end

end



isupper(op::MBOperator)::Bool = op.upper_hermitian
function adjoint!(op::MBOperator{F})::MBOperator{F} where {F}
    if !isupper(op)
        for i in eachindex(op.scats)
            op.scats[i] = adjoint(op.scats[i])
        end
    end
    return op
end
function adjoint(op::MBOperator{F})::MBOperator{F} where {F}
    op_new = deepcopy(op)
    return adjoint!(op_new)
end