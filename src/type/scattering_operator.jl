
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
    scats::Vector{Vector{<:Scatter}}
    upper_hermitian::Bool

    function MBOperator(scats::Vector{Vector{<:Scatter}}; upper_hermitian::Bool)
        sort_merge_scats = sort_merge_scatlist(scats; 
            check_normal =! upper_hermitian, 
            check_normalupper = upper_hermitian
        )
        return new(sort_merge_scats, upper_hermitian)
    end
    function MBOperator(scats::Vector{<: Scatter}...; upper_hermitian::Bool)
        allscats = Vector{Vector{<:Scatter}}(undef, 0)
        for lists in scats
            push!(allscats, lists)
        end
        return MBOperator(allscats; upper_hermitian)
    end
end

isupper(op::MBOperator)::Bool = op.upper_hermitian

import Base: adjoint, show
import LinearAlgebra: adjoint!
function Base.show(io::IO, op::MBOperator)
    println(io, "Many Body Operator (upper_hermitian=$(op.upper_hermitian)) :")
    for scat_list in op.scats
        println(io, "\tVector{$(eltype(scat_list))} with $(length(scat_list)) scattering terms")
    end
end

function adjoint!(op::MBOperator)::MBOperator
    if !isupper(op)
        for scat_list in eachindex(op.scats)
            for i in eachindex(scat_list)
                op.scats[i] = adjoint(op.scats[i])
            end
        end
    end
    return op
end
function adjoint(op::MBOperator)::MBOperator
    op_new = deepcopy(op)
    return adjoint!(op_new)
end