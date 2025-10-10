
using Combinatorics
using LinearAlgebra

"""
    Scatter{N} - Represents an N-body Scatter term in the Hamiltonian
    
    Fields:
    - Amp::ComplexF64: Scatter amplitude
    - out::NTuple{N, Int64}: Output orbital indices (creation operators)
    - in::NTuple{N, Int64}: Input orbital indices (annihilation operators)
    
    Example:
    ```julia
    # One-body Scatter: V * c†_i c_j
    s1 = Scatter(V, i, j)
    
    # Two-body Scatter: V * c†_i1 c†_i2 c_j2 c_j1  
    s2 = Scatter(V, i1, i2, j2, j1)
    ```
"""
struct Scatter{N}
    Amp::ComplexF64
    out::NTuple{N, Int64}
    in::NTuple{N, Int64}
end

"""
    Scatter(V, outin::Int64...)

Construct a Scatter term from amplitude and orbital indices.

The constructor expects an even number of indices, where the first half are output
indices (creation operators) and the second half are input indices (annihilation
operators), in reverse order for proper normal ordering.

# Arguments
- `V::Number`: Scatter amplitude (converted to ComplexF64)
- `outin::Int64...`: Variable number of orbital indices (must be even)

# Examples
```julia
# One-body: V * c†_i c_j
s1 = Scatter(1.0, 1, 2)  # Creates c†_1 c_2 term

# Two-body: V * c†_i1 c†_i2 c_j2 c_j1  
s2 = Scatter(0.5, 1, 2, 4, 3)  # Creates c†_1 c†_2 c_4 c_3 term
```
"""
function Scatter(V, outin::Int64...)
    @assert iseven(length(outin)) "Number of indices must be even"
    N = length(outin) ÷ 2
    outstates = outin[begin:N]
    instates = reverse(outin[N + 1:end])
    return Scatter{N}(ComplexF64(V), outstates, instates)
end

"""
    Base.show(io::IO, st::Scatter{N})

Display a scattering term in a readable format.
"""
function Base.show(io::IO, st::Scatter{N}) where {N}
    print(io, "$N-body Scatter: c†_out ", st.out, " c_in ", reverse(st.in), ", Amp = ", st.Amp)
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
function NormalScatter(V::ComplexF64, ij::Int64...; upper_hermitian::Bool = true)::Scatter
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

    if j_sorted < i_sorted && upper_hermitian
        i_sorted, j_sorted = j_sorted, i_sorted
        V = conj(V)
    end

    return Scatter{N}(V, tuple(i_sorted), tuple(j_sorted))
end
function NormalScatter(V::ComplexF64, i::Int64, j::Int64; upper_hermitian::Bool = true)::Scatter{1}
    # N = 1
    
    # Apply normal ordering rules

    # j >= i
    if j < i && upper_hermitian
        j, i = i, j
        V = conj(V)
    end

    if j == i
        V = real(V) + 0im
    end

    return Scatter{1}(V, (i,), (j,))
end
function NormalScatter(V::ComplexF64, i1::Int64, i2::Int64, j2::Int64, j1::Int64; upper_hermitian::Bool = true)::Scatter{2}
    # N = 2
    
    # Apply normal ordering rules

    # Skip if indices are invalid
    if j1 == j2 || i1 == i2
        @warn "Skipping invalid interaction term: $S"
        return Scatter(0, i1, i2, j2, j1)
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
    if (j1, j2) < (i1, i2) && upper_hermitian
        j1, i1 = i1, j1
        j2, i2 = i2, j2
        V = conj(V)
    end

    if (j1, j2) == (i1, i2)
        V = real(V) + 0im
    end
    return Scatter{2}(V, (i1, i2), (j1, j2))
end
isnormal(x::Scatter)::Bool = issorted(x.in; rev=true) && issorted(x.out; rev=true)
isnormalupper(x::Scatter)::Bool = isnormal(x) && x.in >= x.out


import Base: adjoint
function adjoint(s::Scatter{N})::Scatter{N} where {N}
    Scatter{N}(conj(s.Amp), s.in, s.out)
end
isdiagonal(s::Scatter)::Bool = s.in == s.out



import Base: isless, ==, +, *
isless(x::Scatter{N}, y::Scatter{N}) where {N} = x.in < y.in || x.in == y.in && x.out < y.out
==(x::Scatter{N}, y::Scatter{N}) where {N} = x.in == y.in && x.out == y.out
function +(x::Scatter{N}, y::Scatter{N})::Scatter{N} where {N}
    @assert x == y "Can only add identical Scatter terms"
    return Scatter{N}(x.Amp + y.Amp, x.out, x.in)
end
function *(x::T, scat::Scatter{N})::Scatter{N} where {N, T <: Number}
    Scatter{N}(ComplexF64(x)*scat.Amp, scat.out, scat.in)
end




"""
Sort and merge a list of normalized Scatter terms
"""
function sort_merge_scatlist(normal_sct_list::Vector{Scatter{N}};
    check_normal::Bool = true, check_normalupper::Bool = true
    )::Vector{Scatter{N}} where {N}

    if check_normalupper
        @assert all(isnormalupper, normal_sct_list) "All Scatter terms must be in normal order and in the upper triangular."
    elseif check_normal
        @assert all(isnorm, normal_sct_list) "All Scatter terms must be in normal order."
    end

    sorted_list = sort(normal_sct_list)
    merged_list = Vector{Scatter{N}}()
    for sct in sorted_list
        if !isempty(merged_list) && sct == merged_list[end]
            merged_list[end] += sct
        else
            push!(merged_list, sct)
        end
    end
    return merged_list
end

