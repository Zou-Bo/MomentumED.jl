
using Combinatorics
using LinearAlgebra

"""
    Scattering{N} - Represents an N-body scattering term in the Hamiltonian
    
    Fields:
    - Amp::ComplexF64: Scattering amplitude
    - out::NTuple{N, Int64}: Output orbital indices (creation operators)
    - in::NTuple{N, Int64}: Input orbital indices (annihilation operators)
    
    Example:
    ```julia
    # One-body scattering: V * c†_i c_j
    s1 = Scattering(V, i, j)
    
    # Two-body scattering: V * c†_i1 c†_i2 c_j2 c_j1  
    s2 = Scattering(V, i1, i2, j2, j1)
    ```
"""
struct Scattering{N}
    Amp::ComplexF64
    out::NTuple{N, Int64}
    in::NTuple{N, Int64}
end

"""
    Scattering(V, outin::Int64...)

Construct a scattering term from amplitude and orbital indices.

The constructor expects an even number of indices, where the first half are output
indices (creation operators) and the second half are input indices (annihilation
operators), in reverse order for proper normal ordering.

# Arguments
- `V::Number`: Scattering amplitude (converted to ComplexF64)
- `outin::Int64...`: Variable number of orbital indices (must be even)

# Examples
```julia
# One-body: V * c†_i c_j
s1 = Scattering(1.0, 1, 2)  # Creates c†_1 c_2 term

# Two-body: V * c†_i1 c†_i2 c_j2 c_j1  
s2 = Scattering(0.5, 1, 2, 4, 3)  # Creates c†_1 c†_2 c_4 c_3 term
```
"""
function Scattering(V, outin::Int64...)
    @assert iseven(length(outin)) "Number of indices must be even"
    N = length(outin) ÷ 2
    outstates = outin[begin:N]
    instates = reverse(outin[N + 1:end])
    return Scattering{N}(ComplexF64(V), outstates, instates)
end

"""
    Base.show(io::IO, st::Scattering{N})

Display a scattering term in a readable format.
"""
function Base.show(io::IO, st::Scattering{N}) where {N}
    print(io, "$N-body scattering: c†_out ", st.out, " c_in ", reverse(st.in), ", Amp = ", st.Amp)
end

"""
Generate a scattering term with normal ordering (now working for N = 1, 2)
term: V * c†_i1 c†_i2 ... c†_iN c_jN ... c_j2 c_j1 (j-in, i-out )
(1) j1 > j2 > ... > jN (no equality)
(2) i1 > i2 > ... > iN (no equality)
Hermitian Upper Triangular:
(3) j1 > i1 or j1 = i1 && j2 > i2 or j1,j2 = i1,i2 && j3 > i3 or ... or j1,...,jN-1 = i1,...,iN-1 && jN >= iN
or equlivantly in Julia's grammer
(3') (j1,...,jN) >= (i1,...,iN)
"""
function NormalScattering(V::ComplexF64, ij::Int64...; hermitian_upper_triangular::Bool = true)::Scattering
    @assert iseven(length(ij)) "number conservation requires annihilated and created indices number being even"
    N = length(ij) ÷ 2
    @assert N >= 3

    is = collect(ij[1:N])
    js = collect(ij[2N:-1:N+1])

    i_sort = sortperm(is, rev = true)
    j_sort = sortperm(js, rev = true)
    i_sorted = is[i_sort]
    j_sorted = js[j_sort]
    if isodd(parity(i_sort) + parity(j_sort))
        V = -V
    end

    if j_sorted < i_sorted && hermitian_upper_triangular
        i_sorted, j_sorted = j_sorted, i_sorted
        V = conj(V)
    end

    return Scattering{N}(V, tuple(i_sorted), tuple(j_sorted))
end
function NormalScattering(V::ComplexF64, i::Int64, j::Int64; hermitian_upper_triangular::Bool = true)::Scattering{1}
    # N = 1
    
    # Apply normal ordering rules

    # j >= i
    if j < i && hermitian_upper_triangular
        j, i = i, j
        V = conj(V)
    end

    if j == i
        V = real(V) + 0im
    end

    return Scattering{1}(V, (i,), (j,))
end
function NormalScattering(V::ComplexF64, i1::Int64, i2::Int64, j2::Int64, j1::Int64; hermitian_upper_triangular::Bool = true)::Scattering{2}
    # N = 2
    
    # Apply normal ordering rules

    # Skip if indices are invalid
    if j1 == j2 || i1 == i2
        @warn "Skipping invalid interaction term: $S"
        return Scattering(0, i1, i2, j2, j1)
    end
    
    # i1 > i2
    if i1 < i2
        i1, i2 = i2, i1
        V = -V
    end
    
    # j1 > j2
    if j1 < j2
        j1, j2 = j2, j1
        V = -V
    end
    
    # j1 > i1 or j1 = i1 && j2 > i2
    if (j1, j2) < (i1, i2) && hermitian_upper_triangular
        j1, i1 = i1, j1
        j2, i2 = i2, j2
        V = conj(V)
    end

    if (j1, j2) == (i1, i2)
        V = real(V) + 0im
    end
    return Scattering{2}(V, (i1, i2), (j1, j2))
end
isnormal(x::Scattering)::Bool = issorted(x.in; rev=true) && issorted(x.out; rev=true)
isnormalupper(x::Scattering)::Bool = isnormal(x) && x.in >= x.out


import Base: adjoint
function adjoint(s::Scattering{N})::Scattering{N} where {N}
    Scattering{N}(conj(s.Amp), s.in, s.out)
end
import LinearAlgebra: ishermitian
ishermitian(s::Scattering)::Bool = s.in == s.out



import Base: isless, ==, +, *
isless(x::Scattering{N}, y::Scattering{N}) where {N} = x.in < y.in || x.in == y.in && x.out < y.out
==(x::Scattering{N}, y::Scattering{N}) where {N} = x.in == y.in && x.out == y.out
function +(x::Scattering{N}, y::Scattering{N})::Scattering{N} where {N}
    @assert x == y "Can only add identical Scattering terms"
    return Scattering{N}(x.Amp + y.Amp, x.out, x.in)
end
function *(x::T, scat::Scattering{N})::Scattering{N} where {N, T <: Number}
    Scattering{N}(ComplexF64(x)*scat.Amp, scat.out, scat.in)
end




"""
Sort and merge a list of normalized scattering terms
"""
function sort_merge_scatlist(normal_sct_list::Vector{Scattering{N}};
    check_normal::Bool = true, check_normalupper::Bool = true
    )::Vector{Scattering{N}} where {N}

    if check_normalupper
        @assert all(isnormalupper, normal_sct_list) "All scattering terms must be in normal order and in the upper triangular."
    elseif check_normal
        @assert all(isnorm, normal_sct_list) "All scattering terms must be in normal order."
    end

       sorted_list = sort(normal_sct_list)
    merged_list = Vector{Scattering{N}}()
    for sct in sorted_list
        if !isempty(merged_list) && sct == merged_list[end]
            merged_list[end] += sct
        else
            push!(merged_list, sct)
        end
    end
    return merged_list
end




"""
    struct MBSOperator{F <: Real, I <: Integer} <: AbstractMatrix{Complex{F}}
        scats::Vector{<:Scattering}
        upper_triangular::Bool
    end

Expand an operator in a list of Scattering terms.
When upper_triangular is true, the list contain only upper-triangular terms.

In construction of operators, all the scattering terms automatically passed checking isnormal() or isnormalupper().
Terms of the same scatter-in and -out states are merged. 
"""
struct MBSOperator{F <: Real} <: AbstractMatrix{Complex{F}}
    scats::Vector{<:Scattering}
    upper_triangular::Bool

    function MBSOperator{F}(scats::Vector{<:Scattering}...; upper_triangular::Bool) where {F<:Real}
        allscats = reduce(vcat, scats)
        if upper_triangular
            @assert all(isnormalupper,  allscats) "Scattering terms should all in normal order and in the upper triangular."
        else
            @assert all(isnormal, allscats) "Scattering terms should all in normal order."
        end
        sort_merge_scats = sort_merge_scatlist(allscats; check_normal=false, check_normalupper=false)
        return new{F}(sort_merge_scats, upper_triangular)
    end

end


ishermitian(op::MBSOperator)::Bool = op.upper_triangular
function adjoint!(op::MBSOperator{F})::MBSOperator{F} where {F}
    if !ishermitian(op)
        for i in eachindex(op.scats)
            op.scats[i] = adjoint(op.scats[i])
        end
    end
    return op
end
function adjoint(op::MBSOperator{F})::MBSOperator{F} where {F}
    op_new = deepcopy(op)
    return adjoint!(op_new)
end