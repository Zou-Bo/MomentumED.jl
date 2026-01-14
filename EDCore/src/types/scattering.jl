# todo list:
# constructor for number nonconserved scatter

using LinearAlgebra
using Combinatorics

"""
    Scatter{C <: Complex, MBS <: MBS64}
    
Represents a `Scatter`ing term in an operator:
V * c†_i1 c†_i2 ... c†_iN c_jN ... c_j2 c_j1 (i-out, j-in)
The order of creation and annihilated operators are interpreted in the order
- (1) j1 > j2 > ... > jN (no equality)
- (2) i1 > i2 > ... > iN (no equality)
    
Fields:
- `Amp::C` : Scattering amplitude
- `out::MBS` : Output orbital indices (creation operators)
- `in::MBS` : Insident orbital indices (annihilation operators)

Constructor:
    Scatter(V::Number, out_in::Int64...; bits::Int64, upper_hermitian::Bool = false)

Generate a scattering term with input orbitals. Optimized for N=1,2.
Input orders are the operator order: c†_i1 c†_i2 ... c†_iN c_jN ... c_j2 c_j1 (i-out, j-in)
The sign of amplitute may be flipped when transformed to the sorted order.

if upper_hermitian is true, the in and out states are interchanged such that:\n
(3) j1 > i1 or j1 = i1 && j2 > i2 or j1,j2 = i1,i2 && j3 > i3 or ... or j1,...,jN-1 = i1,...,iN-1 && jN >= iN\n
or equlivently\n
(3') (j1,...,jN) >= (i1,...,iN)\n
(3'') field in >= out\n

Example:
```julia
Scatter(1.0+0.5im, 5, 3; bits=10, upper_hermitian=true) == Scatter(1.0-0.5im, 3, 5; bits=10)
```
"""
struct Scatter{C <: Complex, MBS <: MBS64}
    Amp::C
    out::MBS
    in::MBS
end
function Scatter(V::C, out_in::Int64...; 
    bits::Int64, upper_hermitian::Bool = false
) where{C<:Complex}

    @assert iseven(length(out_in)) "number conservation requires annihilated and created indices number being even"
    N = length(out_in) ÷ 2
    # @assert N >= 3

    is = collect(out_in[1:N])
    js = collect(out_in[2N:-1:N+1])

    i_mask = make_mask64(is)
    j_mask = make_mask64(js)

    # no repetition
    if count_ones(i_mask) < N || count_ones(j_mask) < N
        @info "scattering term with repeated orbitals is zero (Pauli exclusion)"
        return Scatter{C, MBS64{bits}}(zero(V), MBS64{bits}(i_mask), MBS64{bits}(j_mask))
    end

    # normal ordering
    i_sort = sortperm(is, rev = true)
    j_sort = sortperm(js, rev = true)
    if isodd(parity(i_sort) + parity(j_sort))
        V = -V
    end

    if upper_hermitian
        if j_mask < i_mask
            i_mask, j_mask = j_mask, i_mask
            V = conj(V)
        elseif j_mask == i_mask
            V = Complex(real(V))
        end
    end

    return Scatter{C, MBS64{bits}}(V, MBS64{bits}(i_mask), MBS64{bits}(j_mask))
end
function Scatter(V::C, i::Int64, j::Int64;
    bits::Int64, upper_hermitian::Bool = false
) where{C<:Complex}

    # N = 1
    i_mask = make_mask64((i,))
    j_mask = make_mask64((j,))

    upper_hermitian || 
    return Scatter{C, MBS64{bits}}(V, MBS64{bits}(i_mask), MBS64{bits}(j_mask))

    # for upper_hermitian
    if j_mask < i_mask
        j_mask, i_mask = i_mask, j_mask
        V = conj(V)
    elseif j_mask == i_mask
        V = Complex(real(V))
    end
    return Scatter{C, MBS64{bits}}(V, MBS64{bits}(i_mask), MBS64{bits}(j_mask))
end
function Scatter(V::C, i1::Int64, i2::Int64, j2::Int64, j1::Int64;
    bits::Int64, upper_hermitian::Bool = false
) where{C<:Complex}
    
    # N = 2
    
    i_mask = make_mask64((i1, i2))
    j_mask = make_mask64((j1, j2))

    if j1 == j2 || i1 == i2
        @info "scattering term with repeated orbitals is zero (Pauli exclusion)"
        return Scatter{C, MBS64{bits}}(zero(V), MBS64{bits}(i_mask), MBS64{bits}(j_mask))
    end
    
    # Apply normal ordering rules
    if i1 < i2
        V = -V
    end
    if j1 < j2
        V = -V
    end

    if upper_hermitian
        if j_mask < i_mask
            j_mask, i_mask = i_mask, j_mask
            V = conj(V)
        elseif j_mask == i_mask
            V = Complex(real(V))
        end
    end

    return Scatter{C, MBS64{bits}}(V, MBS64{bits}(i_mask), MBS64{bits}(j_mask))
end
function Scatter(V::Real, out_in::Int64...; bits::Int64, upper_hermitian::Bool = false)
    return Scatter(Complex(V), out_in...; bits, upper_hermitian)
end

"""
    Base.show(io::IO, st::Scatter{C, MBS})

Display a scattering term in a readable format.
"""
function Base.show(io::IO, st::Scatter{C, MBS64{bits}}) where {C, bits}
    print(io, typeof(st), ": c†_out ", reverse(occ_list(st.out)), " c_in ", occ_list(st.in), ", Amp = ", st.Amp)
end

"""
docstring needed
"""
isupper(x::Scatter)::Bool = x.in > x.out || x.in == x.out && iszero(imag(x.Amp))

"""
    get_body(::Scatter)

Return number of body.
"""
get_body(s::Scatter) = (count_ones(s.out), count_ones(s.in))

import Base: adjoint
"""
    adjoint(s::Scatter{C, MBS})::Scatter{C, MBS}

Create a reverse scattering term: exchange incident and output orbitals and conjugate amplitude.
"""
function adjoint(s::Scatter{C, MBS})::Scatter{C, MBS} where {C,MBS}
    Scatter{C, MBS}(conj(s.Amp), s.in, s.out)
end
"""
    isdiagonal(s::Scatter)::Bool

s.in == s.out
"""
isdiagonal(s::Scatter)::Bool = s.in == s.out



import Base: isless, ==, +, *
"""
    isless(s1::Scatter, s2::Scatter)::Bool

Irrelevant to amplitute, compare scattering types.

s1.in < s2.in; if equal, s1.out < s2.out.
"""
function isless(s1::Scatter{C, MBS}, s2::Scatter{C, MBS}) where {C, MBS} 
    s1.in < s2.in || s1.in == s2.in && s1.out < s2.out
end
"""
    ==(s1::Scatter1, s2::Scatter2)::Bool

Irrelevant to amplitute, check if the scattering types are the same.

Scatter1 == Scatter2, and s1.in == s2.in, and s1.out == s2.out.
"""
function ==(x::S, y::S)::Bool where{S <: Scatter}
    x.in == y.in && x.out == y.out
end
function ==(x::S1, y::S2)::Bool where {S1 <: Scatter, S2 <: Scatter}
    false
end
"""
    s1::Scatter{C, MBS} + s2::Scatter{C, MBS} -> sum::Scatter{N}

Combine the amplitutes of the two terms of the same type (s1==s2). 

Return a scattering term with summing amplitute.
"""
function +(x::Scatter{C, MBS}, y::Scatter{C, MBS})::Scatter{C, MBS} where {C, MBS}
    @assert x == y "Can only add identical Scatter terms"
    return Scatter{C, MBS}(x.Amp + y.Amp, x.out, x.in)
end
"""
    a::Number * s::Scatter{C, MBS} -> s'::Scatter{C, MBS}

Return a scattering term with amplitute multiplied a number.

The number `a` will be converted as C.
"""
function *(a::T, scat::Scatter{C, MBS})::Scatter{C, MBS} where {C, MBS, T <: Number}
    Scatter{C, MBS}(C(a)*scat.Amp, scat.out, scat.in)
end




"""
    sort_merge_scatlist(lists; keywords)

Sort and merge a list of Scatter terms.

# Input
- `lists`::Vector{Scatter{C, MBS}}:
Return a sorted and merged list: Vector{Scatter{N}}

# Keywords
- `check_upper::Bool` = true: checking all the scatting terms are in the upper triangular of a Hermitian operator.
"""
function sort_merge_scatlist(sct_list::Vector{Scatter{C, MBS}};
    check_upper::Bool = true)::Vector{Scatter{C, MBS}} where {C, MBS}
    
    if isempty(sct_list) return similar(sct_list) end

    if check_upper
        @assert all(isupper, sct_list) "All Scatter terms must be in the upper triangular."
    end

    sorted_list = sort(sct_list)
    merged_list = [0.0 * sorted_list[1]; ] # starting point
    for sct in sorted_list
        if sct == merged_list[end]
            merged_list[end] += sct
        else
            push!(merged_list, sct)
        end
    end
    return merged_list
end
function sort_merge_scatlist(sct_list::Vector{Scatter{C, MBS}}...;
    check_upper::Bool = true)::Vector{Scatter{C, MBS}} where {C, MBS}
    
    return sort_merge_scatlist(vcat(sct_list...); check_upper)
end