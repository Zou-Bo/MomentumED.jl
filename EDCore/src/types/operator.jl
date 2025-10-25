
using LinearAlgebra

"""
    struct MBOperator
        scats::Vector{Vector{<:Scatter}}
        upper_hermitian::Bool
    end

Represents a many-body operator as a list of lists of `Scatter` terms.

# Fields
- `scats::Vector{Vector{<:Scatter}}`: A vector of vectors, where each inner vector contains `Scatter{N}` terms.
- `upper_hermitian::Bool`: Indicates whether the operator is upper Hermitian, affecting how terms are handled.

In the construction of operators, all `Scatter` terms are automatically reorganized by their body number.
Terms of the same scatter-in and -out states are merged.
Terms of the same body number are sorted
"""
struct MBOperator
    scats::Vector{Vector{<:Scatter}}
    upper_hermitian::Bool

    """
        MBOperator(scats::Vector{<: Scatter}...; upper_hermitian::Bool)
        MBOperator(scats::Vector{Vector{<:Scatter}}; upper_hermitian::Bool)

    Constructs an `MBOperator` from input `Scatter` terms.
    The `upper_hermitian` flag determines how scattering terms are sorted and merged.
    """
    function MBOperator(scats::Vector{Vector{<:Scatter}}; upper_hermitian::Bool)
        sort_merge_scats = sort_merge_scatlist(scats; 
            check_normal = !upper_hermitian, 
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

"""
    isupper(op::MBOperator)::Bool

Returns `true` if the operator is upper Hermitian, `false` otherwise.
"""
isupper(op::MBOperator)::Bool = op.upper_hermitian

import Base: adjoint, show
import LinearAlgebra: adjoint!
"""
    Base.show(io::IO, op::MBOperator)

Displays a human-readable representation of the `MBOperator`, including its `upper_hermitian` status and the number of scattering terms.
"""
function Base.show(io::IO, op::MBOperator)
    println(io, "Many Body Operator (upper_hermitian=$(op.upper_hermitian)) :")
    for scat_list in op.scats
        println(io, "\tVector{$(eltype(scat_list))} with $(length(scat_list)) scattering terms")
    end
end

"""
    adjoint!(op::MBOperator)::MBOperator

Computes the adjoint of the operator `op` in-place.
If the operator is not upper Hermitian, it adjoints each scattering term.
"""
function adjoint!(op::MBOperator)::MBOperator
    if !isupper(op)
        for scat_list in op.scats
            for i in eachindex(scat_list)
                scat_list[i] = adjoint(scat_list[i])
            end
        end
    end
    return op
end

"""
    adjoint(op::MBOperator)::MBOperator

Computes the adjoint of the operator `op` by creating a deep copy and then applying `adjoint!`.
"""
function adjoint(op::MBOperator)::MBOperator
    op_new = deepcopy(op)
    return adjoint!(op_new)
end