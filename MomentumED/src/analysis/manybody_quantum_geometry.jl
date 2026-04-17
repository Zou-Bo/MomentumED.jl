"""
Many-body Berry connection calculation functions for momentum-conserved exact diagonalization.

This file implements the calculation of Berry connections and geometric phases between
different kshift points in momentum space, which is essential for computing topological
invariants like Chern numbers.
"""

using LinearAlgebra

"""
Modify the overall phase of eigenvectors. This is a preparation for many-body connection Wilson loop integral.
"""
function ED_connection_gaugefixing!(ψ::MBS64Vector, suggest_dims::Int64=1;
    warning_tol = 1e-8)

    @assert 1 <= suggest_dims <= length(ψ) "suggested vector locates outside the Hilbert space."

    x = ψ.vec[suggest_dims]
    if abs(x) < warning_tol
        @warn "Gauge fixing may be unstable. Element at position $suggest_dims has amplitude $(abs(x))."
    end
    ψ.vec .*= cis(-angle(x))
end

function ED_connection_gaugefixing!(ψ::Vector{<:MBS64Vector}, suggest_dims::Vector{Int64} = collect(1:length(ψ));
    warning_tol = 1e-8)

    @assert all(==(ψ[1].space).(getfield.(ψ, :space))) "Vectors are not in the same Hilbert subspace."
    @assert 1 <= minimum(suggest_dims) && maximum(suggest_dims) <= length(ψ[1].vec) "suggested vectors locate outside the Hilbert space."
    @assert length(suggest_dims) == length(ψ) "number of suggested vectors mismatches number of eigenvectors."

    matrix = [ ψ[j].vec[i] for i in suggest_dims, j in eachindex(ψ)]
    x = det(matrix)
    if abs(x) < warning_tol
        @warn "Gauge fixing may be unstable. Determinant of elements at position $suggest_dims has amplitude $(abs(x))."
    end
    ψ[1].vec .*= cis(-angle(x))
end



function ED_step_inner_prod(ψ_f::MbsVec, ψ_i::MbsVec,
    orbital_inner_prod::Vector{Complex{F}}
    )::Complex{F} where {bits, F <: AbstractFloat, 
    MbsVec <: MBS64Vector{bits, F}}

    @assert ψ_f.space == ψ_i.space || ψ_f.space.list == ψ_i.space.list "The Hilbert subspaces of eigenvectors must match."
    @assert length(orbital_inner_prod) == bits "orbital inner product length should be $bits."

    inner_prod = zero(Complex{F})  # many-body connection step integral
    for x in eachindex(ψ_f.vec)
        coeffi::Complex{F} = ψ_f.vec[x]' * ψ_i.vec[x]
        product::Complex{F} = prod(
            orbital_inner_prod[occ_list(ψ_f.space.list[x])];
            init = one(Complex{F})
        )
        inner_prod += coeffi * product
    end
    return inner_prod
end
function ED_step_inner_prod(ψ_f::Vector{MbsVec}, ψ_i::Vector{MbsVec},
    orbital_inner_prod::Vector{Complex{F}}
    )::Matrix{Complex{F}} where {bits, F <: AbstractFloat, MbsVec <: MBS64Vector{bits, F}}

    @assert all(==(ψ_f[1].space).(getfield.(ψ_f, :space))) "The Hilbert subspaces of eigenvectors must match."
    @assert all(==(ψ_i[1].space).(getfield.(ψ_i, :space))) "The Hilbert subspaces of eigenvectors must match."
    @assert ψ_f[1].space == ψ_i[1].space || ψ_f[1].space.list == ψ_i[1].space.list "The Hilbert subspaces of eigenvectors must match."
    @assert length(ψ_f) == length(ψ_i) "Number of eigenvectors must match in non-Abelian connection"
    @assert length(orbital_inner_prod) == bits "orbital inner product length should be $bits."

    g = length(ψ_f)  # degeneracy
    inner_prod_matrix = zeros(Complex{F}, g, g)  # collect many-body connection step integral
    # reuse temporary matrices/vectors avoid repeated allocations
    coeffi = similar(inner_prod_matrix)
    psi_i = Vector{Complex{F}}(undef, g)
    psi_f = Vector{Complex{F}}(undef, g)
    for x in eachindex(ψ_f[1].vec)
        for j in eachindex(psi_i, psi_f)
            psi_i[j] = ψ_i[j].vec[x]
            psi_f[j] = conj(ψ_f[j].vec[x])
        end
        mul!(coeffi, psi_f, transpose(psi_i))
        product::Complex{F} = prod(
            orbital_inner_prod[occ_list(ψ_f[1].space.list[x])];
            init = one(Complex{F})
        )
        inner_prod_matrix .+= coeffi .* product
    end

    return inner_prod_matrix
end
