"""
This file provides matrix-free LinearMap structures for Krylov-Schur diagonalization.
"""

# todo list
# non-hermitian map and adjoint things

using LinearAlgebra

abstract type AbstractLinearMap{bits, F <: AbstractFloat} end
abstract type AbstractCPULinearMap{bits, F <: AbstractFloat} <: AbstractLinearMap{bits, F} end

"""
(docstring needed)
when constructing LinearMap from scat_list, the list is always copied as a distict Vector.
when constructing from adjointing a AdjointLinearMap, the scat_list is shared.
"""
mutable struct LinearMap{bits, F <: AbstractFloat} <: AbstractCPULinearMap{bits, F}
    scat_list::Vector{Scatter{Complex{F}, MBS64{bits}}}
    space::HilbertSubspace{bits}

    function LinearMap(op::MBOperator{Complex{F}, MBS64{bits}}, space::HilbertSubspace{bits}) where {bits, F <: AbstractFloat}
        @assert F ∈ (Float64, Float32) "element_type=$(Complex{F}). Use Complex of Float64, Float32."
        if op.upper_hermitian
            new_scats = copy(op.scats)
            for s in op.scats
                if !isdiagonal(s)
                    push!(new_scats, s')
                end
            end
            sort!(new_scats)
            return new{bits, F}(new_scats, space)
        else
            # error("Try to create linear map for a Hamiltonian MBOperator that is not upper_hermitian.")
            return new{bits, F}(copy(op.scats), space)
        end
    end
    # function LinearMap(scat_list::Vector{Scatter{Complex{F}, MBS64{bits}}}, 
    #     subspace::HilbertSubspace{bits}) where {bits, F<:AbstractFloat}
    #     new{bits, F}(copy(scat_list), subspace)
    # end
end

"""
(docstring needed)
The AdjointLinearMap can only be constructed by adjointing a LinearMap, sharing the scat_list.
"""
mutable struct AdjointLinearMap{bits, F <: AbstractFloat} <: AbstractCPULinearMap{bits, F}
    scat_list::Vector{Scatter{Complex{F}, MBS64{bits}}}
    space::HilbertSubspace{bits}

    function AdjointLinearMap(A::LinearMap{bits, F}) where {bits, F <: AbstractFloat}
        new{bits, F}(A.scat_list, A.space)
    end
end

import Base: size, adjoint, eltype
# import LinearAlgebra: adjoint!

"""
(docstring needed)
"""
function size(A::AbstractLinearMap)
    n = length(A.space)
    return (n, n)
end
"""
(docstring needed)
"""
eltype(::AbstractLinearMap{bits, F}) where {bits, F <: AbstractFloat} = Complex{F}

# """
# (docstring needed)
# """
# function adjoint!(A_adj::AdjointLinearMap{bits, F})::LinearMap{bits, F} where {bits, F <: AbstractFloat}
#     A_adj = reinterpret(LinearMap{bits, F}, A_adj)
# end
# function adjoint!(A::LinearMap{bits, F})::AdjointLinearMap{bits, F} where {bits, F <: AbstractFloat}
#     A = reinterpret(AdjointLinearMap{bits, F}, A)
# end

"""
(docstring needed)
"""
function adjoint(A_adj::AdjointLinearMap{bits, F})::LinearMap{bits, F} where {bits, F <: AbstractFloat}
    reinterpret(LinearMap{bits, F}, A_adj)
end
function adjoint(A::LinearMap{bits, F})::AdjointLinearMap{bits, F} where {bits, F <: AbstractFloat}
    AdjointLinearMap(A)
end

# multiplication
@inline function _check_linearmap_dims(y, x, n::Int)
    if length(y) != n || length(x) != n
        throw(DimensionMismatch("Dimension of Hamiltonian linear map mismatches vector length."))
    end
    return nothing
end

function (A::LinearMap{bits, F})(
    y::AbstractVector{Complex{F}}, x::AbstractVector{Complex{F}}
    )::AbstractVector{Complex{F}} where {bits, F}
    
    n = length(A.space)
    _check_linearmap_dims(y, x, n)

    y .= zero(Complex{F})
    Threads.@threads for i in eachindex(A.space.list)
        for scat in A.scat_list
            amp, mbs_in = A.space.list[i] * scat
            if !iszero(amp)
                j = get(A.space, mbs_in)
                index_fit(j, A.space, mbs_in) && (y[i] += amp * x[j])
            end
        end
    end
    return y
end

function (A::AdjointLinearMap{bits, F})(
    y::AbstractVector{Complex{F}}, x::AbstractVector{Complex{F}}
    )::AbstractVector{Complex{F}} where {bits, F}

    n = length(A.space)
    _check_linearmap_dims(y, x, n)

    y .= zero(Complex{F})
    Threads.@threads for i in eachindex(A.space.list)
        for scat in A.scat_list
            amp, mbs_in = scat * A.space.list[i] # adjoint operator: inversely scatter to find the mbs_in 
            if !iszero(amp)
                j = get(A.space, mbs_in)
                # @boundscheck @assert j != 0 "H is not momentum- or component-conserving."
                index_fit(j, A.space, mbs_in) && (y[i] += conj(amp) * x[j]) # adjoint operator: conj(amplitute)
            end
        end
    end
    return y
end

function (A::LinearMap{bits, F})(x::AbstractVector{Complex{F}})::AbstractVector{Complex{F}} where {bits, F}
    y = similar(x)
    A(y, x)
    return y
end

function (A::AdjointLinearMap{bits, F})(x::AbstractVector{Complex{F}})::AbstractVector{Complex{F}} where {bits, F}
    y = similar(x)
    A(y, x)
    return y
end





"""
    krylov_map_solve(H::AbstractLinearMap, N_eigen::Int64; 
        ishermitian::Bool=true, vec0 = nothing, krylovkit_kwargs...) -> (vals, vecs)

Solve the matrix-free Hamiltonian map using KrylovKit's `eigsolve` function for the
lowest `N_eigen` eigenvalues and eigenvectors.

# Arguments
- `H::AbstractLinearMap{bits, eltype}`: Matrix-free Hamiltonian map to diagonalize.
- `N_eigen::Int64`: Number of eigenvalues/eigenvectors to compute.

# Keywords
- `ishermitian::Bool=true`: Whether the map is Hermitian.
- `vec0::Union{Nothing, AbstractVector{Complex{eltype}}}=nothing`: Optional starting vector.
  If `H isa CuLinearMap`, a random `CuArray` is generated on the active CUDA device.
- `krylovkit_kwargs...`: Additional keyword arguments passed to `KrylovKit.eigsolve`.

# Returns
- `vals`: Eigenvalues.
- `vecs`: Corresponding eigenvectors. Their container type follows the starting vector type.
- `info`: Convergence information from KrylovKit.
"""
function krylov_map_solve(
    H::AbstractCPULinearMap{bits, eltype}, N_eigen::Int64;
    ishermitian::Bool = true,
    vec0::Union{Nothing, AbstractVector{Complex{eltype}}} = nothing,
    krylovkit_kwargs...) where {bits, eltype <: AbstractFloat}

    m = length(H.space)
    if isnothing(vec0)
        vec0 = rand(Complex{eltype}, m)
    end
    N_eigen = min(N_eigen, m)
    eigsolve(H, vec0, N_eigen, :SR; ishermitian, krylovkit_kwargs...)
end
