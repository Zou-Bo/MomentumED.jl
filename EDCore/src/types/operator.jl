using LinearAlgebra

"""
    struct MBOperator{T <: Tuple{Vararg{Vector{<:Scatter}}}}
        scats::T
        upper_hermitian::Bool
    end

Represents a many-body operator as a list of lists of `Scatter` terms.

# Fields
- `scats::Tuple{Vararg{Vector{<:Scatter}}}`: A Tuple of vectors, where each inner vector contains `Scatter{N}` terms.
- `upper_hermitian::Bool`: Indicates whether the operator is upper Hermitian, affecting how terms are handled.

In the construction of operators, all `Scatter` terms are automatically reorganized by their body number.
Terms of the same scatter-in and -out states are merged.
Terms of the same body number are sorted

# Constructor
    MBOperator(scats::Vector{<: Scatter}...; upper_hermitian::Bool)
    MBOperator(scats::Vector{Vector{<:Scatter}}; upper_hermitian::Bool)

Constructs an `MBOperator` from input `Scatter` terms.
The `upper_hermitian` flag determines how scattering terms are sorted and merged.
"""
struct MBOperator{C <: Complex, MBS <: MBS64}
    scats::Vector{Scatter{C, MBS}}
    upper_hermitian::Bool

    function MBOperator(scats::Vector{Scatter{C, MBS}}...; upper_hermitian::Bool) where {C, MBS}
        sort_merge_scats = sort_merge_scatlist(scats...; check_upper = upper_hermitian)
        return new{C, MBS}(sort_merge_scats, upper_hermitian)
    end
end

"""
    isupper(op::MBOperator)::Bool

Returns `true` if the operator is upper Hermitian, `false` otherwise.
"""
isupper(op::MBOperator)::Bool = op.upper_hermitian

import Base.show
"""
    Base.show(io::IO, op::MBOperator)

Displays a human-readable representation of the `MBOperator`, including its `upper_hermitian` status and the number of scattering terms.
"""
function Base.show(io::IO, op::MBOperator)
    println(io, "Many Body Operator " * "(Hermitian, upper triangle entries only) "^op.upper_hermitian, ":")
    println(io, "\tVector{$(eltype(op.scats))} with $(length(op.scats)) scattering terms")
end

import Base.adjoint, LinearAlgebra.adjoint!
"""
    adjoint!(op::MBOperator{C, MBS})::MBOperator{C, MBS}

Computes the adjoint of the operator `op` in-place.
If the operator is not upper Hermitian, it adjoints each scattering term.
"""
function adjoint!(op::MBOperator{C, MBS})::MBOperator{C, MBS} where {C, MBS}
    if !isupper(op)
        for (i, scat) in enumerate(op.scats)
            op.scats[i] = adjoint(scat)
        end
        sort!(op.scats)
    end
    return op
end

"""
    adjoint(op::MBOperator{C, MBS})::MBOperator{C, MBS}

Computes the adjoint of the operator `op` by creating a deep copy and then applying `adjoint!`.
"""
function adjoint(op::MBOperator{C, MBS})::MBOperator{C, MBS} where {C, MBS}
    op_new = deepcopy(op)
    return adjoint!(op_new)
end