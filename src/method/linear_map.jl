"""
This file provides optimized structure than MBOperator for doing Krylov-Schur method
"""

# todo list
# non-hermitian map and adjoint things

using LinearAlgebra

abstract type AbstractLinearMap{bits, F <: AbstractFloat} end

mutable struct LinearMap{bits, F <: AbstractFloat} <: AbstractLinearMap{bits, F}
    scat_lists::Vector{Vector{<:Scatter}}
    space::HilbertSubspace{bits}

    function LinearMap(op::MBOperator, space::HilbertSubspace{bits}, element_type::Type) where {bits}
        @assert element_type ∈ (Float64, Float32) "element_type=$element_type. Use element_type Float64, Float32."
        if op.upper_hermitian
            scats = deepcopy(op.scats)
            for i in eachindex(op.scats)
                for s in op.scats[i]
                    if !isdiagonal(s)
                        push!(scats[i], s')
                    end
                end
            end
            return new{bits, element_type}(scats, space)
        else
            error("Try to create linear map for a Hamiltonian MBOperator that is not upper_hermitian.")
        end
    end
end

mutable struct AdjointLinearMap{bits, F <: AbstractFloat} <: AbstractLinearMap{bits, F}
    scat_lists::Vector{Vector{<:Scatter}}
    space::HilbertSubspace{bits}

    function AdjointLinearMap(A::LinearMap{bits, F}) where {bits, F <: AbstractFloat}
        new{bits, F}(A.scat_lists, A.space)
    end
end

import Base: size, adjoint, eltype
import LinearAlgebra: adjoint!

function size(A::AbstractLinearMap)
    n = length(A.space)
    (n, n)
end

function adjoint!(A_adj::Adjoint{LinearMap{bits, F}})::LinearMap{bits, F} where {bits, F <: AbstractFloat}
    reinterpret(LinearMap{bits, F}, A_adj)
end

function adjoint!(A::LinearMap{bits, F})::Adjoint{LinearMap{bits, F}} where {bits, F <: AbstractFloat}
    reinterpret(Adjoint{LinearMap{bits, F}}, A)
end

function adjoint(A_adj::AdjointLinearMap{bits, F}
    )::LinearMap{bits, F} where {bits, F <: AbstractFloat}
    A = deepcopy(A_adj)
    adjoint!(A)
end
adjoint(A::LinearMap{bits, F}) where {bits, F <: AbstractFloat} = AdjointLinearMap(A)

eltype(::AbstractLinearMap{bits, F}) where {bits, F<:AbstractFloat} = Complex{F}

# multiplication

function (A::LinearMap{bits, F})(y::Vector{Complex{F}}, x::Vector{Complex{F}}) where {bits, F}
    n = length(A.space)
    if length(y) != n && length(x) != n
        throw(DimensionMismatch("Dimension of Hamiltonian linear map mismatches vector length."))
    end

    y .= zero(Complex{F})
    list::Vector{MBS64{bits}} = A.space.list
    for scat_list in A.scat_lists
        Threads.@threads for i in eachindex(list)
            for scat in scat_list
                amp, mbs_in = list[i] * scat
                if amp != 0.0
                    j = get(A.space, mbs_in)
                    @assert i != 0 "H is not momentum- or component-conserving."
                    y[i] += amp * x[j]
                end
            end
        end
    end

end

function (A::AdjointLinearMap{bits, F})(y::Vector{Complex{F}}, x::Vector{Complex{F}}) where {bits, F}
    n = length(A.space)
    if length(y) == n && length(x) == n
        throw(DimensionMismatch("Dimension of Hamiltonian linear map mismatches vector length."))
    end

    y .= zero(Complex{F})
    list::Vector{MBS64{bits}} = A.space.list
    for scat_list in A.scat_lists
        Threads.@threads for i in eachindex(list)
            for scat in scat_list
                amp, mbs_in = scat * list[i] # adjoint operator: inversely find the mbs_in 
                if amp != 0.0
                    j = get(A.space, mbs_in)
                    @assert i != 0 "H is not momentum- or component-conserving."
                    y[i] += conj(amp) * x[j] # adjoint operator: conj(amplitute)
                end
            end
        end
    end

end

# function (A::LinearMap{bits, F})(x::Vector{Complex{F}})::Vector{Complex{F}} where {bits, F}
#     y = similar(x)
#     A(y, x)
#     return y
# end

# function (A::AdjointLinearMap{bits, F})(x::Vector{Complex{F}})::Vector{Complex{F}} where {bits, F}
#     y = similar(x)
#     A(y, x)
#     return y
# end





"""
    krylov_map_solve(H::SparseMatrixCSC{ComplexF64, Int64}, N_eigen::Int64=6; 
        converge_warning::Bool=false, krylovkit_kwargs...) -> (vals, vecs)

Solve the sparse Hamiltonian matrix using KrylovKit's eigsolve function for the lowest n eigenvalues and eigenvectors.

# Arguments
- `H::SparseMatrixCSC{Complex{eltype}, idtype}`: Sparse Hamiltonian matrix to diagonalize
- `N_eigen::Int64=6`: Number of eigenvalues/eigenvectors to compute (default: 6)

# Keywords
- `vec0::Vector{Complex{eltype}}=rand(Complex{eltype}, H.m)`: Initial guess vector for Krylov iteration
- `ishermitian::Bool=true`: Whether the matrix is Hermitian (default: true)
- `krylovkit_kwargs...`: Additional keyword arguments to pass to KrylovKit.eigsolve

# Returns
- `vals::Vector{eltype}`: Eigenvalues (energies) in ascending order
- `vecs::Vector{Vector{Complex{eltype}}}`: Corresponding eigenvectors
- `info`: Convergence information from KrylovKit

# Examples
```julia
# Solve for 3 lowest eigenstates
vals, vecs, info = matrix_solve(H_matrix, 3)
println("Ground state energy: ", vals[1])
```

# Notes
- Uses KrylovKit's eigsolve with :SR (smallest real) eigenvalue selection
- Assumes Hermitian matrix (standard for quantum Hamiltonians)
- Random initial vector ensures good convergence properties
- Automatically handles convergence warnings from KrylovKit
- For better control over convergence, consider using KrylovKit directly
"""
function krylov_map_solve(
    H::LinearMap{bits, eltype}, N_eigen::Int64;
    ishermitian::Bool = true, krylovkit_kwargs...
)::Tuple{Vector{eltype}, Vector{Vector{Complex{eltype}}}, Any} where {bits, eltype<:AbstractFloat}

    m = length(space)
    vec0 = rand(Complex{eltype}, m)
    N_eigen = min(N_eigen, m)
    eigsolve(H, vec0, N_eigen, :SR; ishermitian, krylovkit_kwargs...)
end
