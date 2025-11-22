"""
This file provides optimized structure than MBOperator for doing Krylov-Schur method
"""

# todo list
# non-hermitian map and adjoint things

using LinearAlgebra

abstract type AbstractLinearMap{bits, F <: AbstractFloat, T <: Tuple{Vararg{Vector{<:Scatter}}}} end

mutable struct LinearMap{bits, F <: AbstractFloat, T <: Tuple{Vararg{Vector{<:Scatter}}}} <: AbstractLinearMap{bits, F, T}
    scat_lists::T
    space::HilbertSubspace{bits}

    function LinearMap(op::MBOperator{T}, space::HilbertSubspace{bits}, element_type::Type) where {bits, T}
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
            return new{bits, element_type, T}(scats, space)
        else
            error("Try to create linear map for a Hamiltonian MBOperator that is not upper_hermitian.")
        end
    end
end

mutable struct AdjointLinearMap{bits, F <: AbstractFloat, T <: Tuple{Vararg{Vector{<:Scatter}}}} <: AbstractLinearMap{bits, F, T}
    scat_lists::T
    space::HilbertSubspace{bits}

    function AdjointLinearMap(A::LinearMap{bits, F, T}) where {bits, F <: AbstractFloat, T <: Tuple{Vararg{Vector{<:Scatter}}}}
        new{bits, F, T}(A.scat_lists, A.space)
    end
end

import Base: size, adjoint, eltype
import LinearAlgebra: adjoint!

function size(A::AbstractLinearMap)
    n = length(A.space)
    (n, n)
end

function adjoint!(A_adj::Adjoint{LinearMap{bits, F, T}})::LinearMap{bits, F, T} where {bits, F <: AbstractFloat, T <: Tuple{Vararg{Vector{<:Scatter}}}}
    reinterpret(LinearMap{bits, F, T}, A_adj)
end

function adjoint!(A::LinearMap{bits, F, T})::Adjoint{LinearMap{bits, F, T}} where {bits, F <: AbstractFloat, T <: Tuple{Vararg{Vector{<:Scatter}}}}
    reinterpret(Adjoint{LinearMap{bits, F, T}}, A)
end

function adjoint(A_adj::AdjointLinearMap{bits, F, T}
    )::LinearMap{bits, F, T} where {bits, F <: AbstractFloat, T <: Tuple{Vararg{Vector{<:Scatter}}}}
    A = deepcopy(A_adj)
    adjoint!(A)
end
adjoint(A::LinearMap{bits, F, T}) where {bits, F <: AbstractFloat, T <: Tuple{Vararg{Vector{<:Scatter}}}} = AdjointLinearMap(A)

eltype(::AbstractLinearMap{bits, F, T}) where {bits, F<:AbstractFloat, T} = Complex{F}

# multiplication

function (A::LinearMap{bits, F, T})(y::Vector{Complex{F}}, x::Vector{Complex{F}}) where {bits, F, T}
    n = length(A.space)
    if length(y) != n && length(x) != n
        throw(DimensionMismatch("Dimension of Hamiltonian linear map mismatches vector length."))
    end

    y .= zero(Complex{F})
    foreach(A.scat_lists) do scat_list
        Threads.@threads for i in eachindex(A.space.list)
            for scat in scat_list
                amp, mbs_in = A.space.list[i] * scat
                if amp != 0.0
                    j = get(A.space, mbs_in)
                    @boundscheck @assert j != 0 "H is not momentum- or component-conserving."
                    y[i] += amp * x[j]
                end
            end
        end
    end

end

function (A::AdjointLinearMap{bits, F, T})(y::Vector{Complex{F}}, x::Vector{Complex{F}}) where {bits, F, T}
    n = length(A.space)
    if length(y) == n && length(x) == n
        throw(DimensionMismatch("Dimension of Hamiltonian linear map mismatches vector length."))
    end

    y .= zero(Complex{F})
    foreach(A.scat_lists) do scat_list
        Threads.@threads for i in eachindex(A.space.list)
            for scat in scat_list
                amp, mbs_in = scat * A.space.list[i] # adjoint operator: inversely find the mbs_in 
                if amp != 0.0
                    j = get(A.space, mbs_in)
                    @boundscheck @assert j != 0 "H is not momentum- or component-conserving."
                    y[i] += conj(amp) * x[j] # adjoint operator: conj(amplitute)
                end
            end
        end
    end

end

function (A::LinearMap{bits, F, T})(x::Vector{Complex{F}})::Vector{Complex{F}} where {bits, F, T}
    y = similar(x)
    A(y, x)
    return y
end

function (A::AdjointLinearMap{bits, F, T})(x::Vector{Complex{F}})::Vector{Complex{F}} where {bits, F, T}
    y = similar(x)
    A(y, x)
    return y
end





"""
    krylov_map_solve(H::SparseMatrixCSC{ComplexF64, Int64}, N_eigen::Int64=6; 
        converge_warning::Bool=false, krylovkit_kwargs...) -> (vals, vecs)

Solve the sparse Hamiltonian matrix using KrylovKit's eigsolve function for the lowest `N_eigen` eigenvalues and eigenvectors.

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
vals, vecs, info = krylov_map_solve(H_map, 3)
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

    m = length(H.space)
    vec0 = rand(Complex{eltype}, m)
    N_eigen = min(N_eigen, m)
    eigsolve(H, vec0, N_eigen, :SR; ishermitian, krylovkit_kwargs...)
end
