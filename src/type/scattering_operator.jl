
using LinearAlgebra

"""
    struct MBOperator{F <: Real}
        scats::Vector{<:Scatter}
        upper_triangular::Bool
    end

Expand an operator in a list of Scatter terms.
When upper_triangular is true, the list contain only upper-triangular terms.

In construction of operators, all the Scatter terms automatically passed checking isnormal() or isnormalupper().
Terms of the same scatter-in and -out states are merged. 
"""
struct MBOperator
    scats::Vector{Scatter}
    upper_hermitian::Bool

    function MBOperator(scats::Vector{<: Scatter}...; upper_hermitian::Bool)
        allscats::Vector{Scatter} = reduce(vcat, scats)
        if upper_hermitian
            @assert all(isnormalupper, allscats) "Scatter terms aren't all in normal order or in the upper triangular or not real in diagonal line."
        else
            @assert all(isnormal, allscats) "Scatter terms aren't all in normal order."
        end
        sort_merge_scats = sort_merge_scatlist(allscats; check_normal=false, check_normalupper=false)
        return new(sort_merge_scats, upper_hermitian)
    end

end

isupper(op::MBOperator)::Bool = op.upper_hermitian

import Base: adjoint, show
import LinearAlgebra: adjoint!
function Base.show(io::IO, op::MBOperator)

end

function adjoint!(op::MBOperator)::MBOperator
    if isupper(op)
        for i in eachindex(op.scats)
            if isdiagonal(op.scats[i])
                op.scats[i] = adjoint(op.scats[i])
            end
        end
    else
        for i in eachindex(op.scats)
            op.scats[i] = adjoint(op.scats[i])
        end
    end
    return op
end
function adjoint(op::MBOperator)::MBOperator
    op_new = deepcopy(op)
    return adjoint!(op_new)
end