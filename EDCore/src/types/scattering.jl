# to-do list:
# Scattering{F <: AbstractFloat, N}

using LinearAlgebra

"""
    Scatter{N} - Represents an N-body Scatter term in the Hamiltonian
    
    Fields:
    - Amp::ComplexF64: Scatter amplitude
    - out::NTuple{N, Int64}: Output orbital indices (creation operators)
    - in::NTuple{N, Int64}: Input orbital indices (annihilation operators)
"""
struct Scatter{N}
    Amp::ComplexF64
    out::NTuple{N, Int64}
    in::NTuple{N, Int64}
end

"""
    Scatter(V, outin::Int64...)

Construct a Scatter term from amplitude and orbital indices. 
However, it's recommemded to use `NormalScatter` for construction.


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
    NormalScatter(V::ComplexF64, ij::Int64...; upper_hermitian::Bool = false)::Scatter

Generate a scattering term with normal ordering. Optimized for N=1,2.

term: V * c†_i1 c†_i2 ... c†_iN c_jN ... c_j2 c_j1 (j-in, i-out )
- (1) j1 > j2 > ... > jN (no equality)
- (2) i1 > i2 > ... > iN (no equality)
- Hermitian Upper Triangular:\n
(3) j1 > i1 or j1 = i1 && j2 > i2 or j1,j2 = i1,i2 && j3 > i3 or ... or j1,...,jN-1 = i1,...,iN-1 && jN >= iN\n
or equlivently in Julia's grammer\n
(3') (j1,...,jN) >= (i1,...,iN)

Example:
```julia
NormalScatter(1.0+0.0im, 5, 3) == NormalScatter(-1.0+0.0im, 3, 5)
```
"""
function NormalScatter(V::ComplexF64, ij::Int64...; upper_hermitian::Bool = false)::Scatter
    @assert iseven(length(ij)) "number conservation requires annihilated and created indices number being even"
    N = length(ij) ÷ 2
    @assert N >= 3

    is = collect(ij[1:N])
    js = collect(ij[2N:-1:N+1])

    i_sort = sortperm(is, rev = true)
    j_sort = sortperm(js, rev = true)
    i_sorted = is[i_sort]
    j_sorted = js[j_sort]

    # no repetition
    if reduce(|, [diff(i_sorted); diff(j_sorted)] .== 0; init = false)
        return Scatter{N}(0.0, i_sorted, j_sorted)
    end
    if isodd(parity(i_sort) + parity(j_sort))
        V = -V
    end

    if upper_hermitian
        if j_sorted < i_sorted
            i_sorted, j_sorted = j_sorted, i_sorted
            V = conj(V)
        elseif j_sorted == i_sort
            V = real(V) + 0.0im
        end
    end

    return Scatter{N}(V, i_sorted, j_sorted)
end
function NormalScatter(V::ComplexF64, i::Int64, j::Int64; upper_hermitian::Bool = false)::Scatter{1}
    # N = 1
    
    upper_hermitian || return Scatter{1}(V, (i,), (j,))

    # Apply normal ordering rules only when upper_hermitian is required

    # j >= i
    if j < i
        j, i = i, j
        V = conj(V)
    elseif j == i
        V = real(V) + 0.0im
    end

    return Scatter{1}(V, (i,), (j,))
end
function NormalScatter(V::ComplexF64, i1::Int64, i2::Int64, j2::Int64, j1::Int64; upper_hermitian::Bool = false)::Scatter{2}
    # N = 2
    
    # Apply normal ordering rules
    
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

    # Skip if indices are invalid
    if j1 == j2 || i1 == i2
        @warn "Skipping invalid interaction term: $S"
        return Scatter(0.0, i1, i2, j2, j1)
    end
    
    # j1 > i1 or j1 = i1 && j2 > i2
    if upper_hermitian
        if (j1, j2) < (i1, i2) 
            j1, i1 = i1, j1
            j2, i2 = i2, j2
            V = conj(V)
        elseif (j1, j2) == (i1, i2)
            V = real(V) + 0.0im
        end
    end

    return Scatter{2}(V, (i1, i2), (j1, j2))
end
"""
docstring needed
"""
isnormal(x::Scatter)::Bool = issorted(x.in; rev=true) && issorted(x.out; rev=true)
"""
docstring needed
"""
isnormalupper(x::Scatter)::Bool = isnormal(x) && (x.in > x.out || x.in == x.out && iszero(imag(x.Amp)))

"""
    get_body(::Scatter{N}) = N

Return number of body.
"""
get_body(::Scatter{N}) where {N} = N

import Base: adjoint
"""
    adjoint(s::Scatter{N})::Scatter{N}

Create a reverse scattering term: exchange incident and output orbitals and conjugate amplitude.
"""
function adjoint(s::Scatter{N})::Scatter{N} where {N}
    Scatter{N}(conj(s.Amp), s.in, s.out)
end
"""
    isdiagonal(s::Scatter)::Bool

s.in == s.out
"""
isdiagonal(s::Scatter)::Bool = s.in == s.out



import Base: isless, ==, +, *
"""
    isless(s1::Scatter{N1}, s2::Scatter{N2})::Bool

Irrelevant to  amplitute, compare scattering types.

N1 < N2; if equal, s1.in < s2.in; if equal, s1.out < s2.out.
"""
function isless(s1::Scatter{N1}, s2::Scatter{N2}) where {N1, N2} 
    N1 < N2 || N1 == N2 && ( s1.in < s2.in || s1.in == s2.in && s1.out < s2.out )
end
"""
    ==(s1::Scatter{N1}, s2::Scatter{N2})::Bool

Irrelevant to  amplitute, check if the scattering types are the same.

N1 == N2, and s1.in == s2.in, and s1.out == s2.out.
"""
function ==(x::Scatter{N1}, y::Scatter{N2}) where {N1, N2}
    N1 == N2 && x.in == y.in && x.out == y.out
end
"""
    s1::Scatter{N} + s2::Scatter{N} -> sum::Scatter{N}

Combine the amplitutes of the two terms of the same type. 

(N1 == N2) and s1.in == s2.in and s1.out == s2.out.

Return a scattering term with summing amplitute.
"""
function +(x::Scatter{N}, y::Scatter{N})::Scatter{N} where {N}
    @assert x == y "Can only add identical Scatter terms"
    return Scatter{N}(x.Amp + y.Amp, x.out, x.in)
end
"""
    a::Number * s::Scatter{N} -> s'::Scatter{N}

Return a scattering term with amplitute multiplied a number.

The number should be able to converted as ComplexF64.
"""
function *(a::T, scat::Scatter{N})::Scatter{N} where {N, T <: Number}
    Scatter{N}(ComplexF64(a)*scat.Amp, scat.out, scat.in)
end




"""
    sort_merge_scatlist(lists; keywords)

Sort and merge a list (lists) of normalized Scatter terms.

# Input
- (1) lists::Vector{Scatter{N}}:
Return a sorted and merged list: Vector{Scatter{N}}
- (2) lists::Vector{Scatter}:
Grouping scattering terms by their N, then sorting and merging each group. 
Return a list of sorted and merged lists: Vector{Vector{<: Scatter}}, each inner list has a specified N.
- (3) lists::Vector{Vector{<:Scatter}}:
Similar to (2).
Return a list of sorted and merged lists: Vector{Vector{<: Scatter}}, each inner list has a specified N.

# Keywords
- `check_normal::Bool` = true: checking all the scatting terms are in nomal order
- `check_normalupper::Bool` = true: checking all the scatting terms are in nomal order and is in the upper triangular position.
"""
function sort_merge_scatlist(normal_sct_list::Vector{<: Scatter};
    check_normal::Bool = true, check_normalupper::Bool = true
    )::Vector{<:Scatter}

    if check_normalupper
        @assert all(isnormalupper, normal_sct_list) "All Scatter terms must be in normal order and in the upper triangular."
    elseif check_normal
        @assert all(isnormal, normal_sct_list) "All Scatter terms must be in normal order."
    end

    sorted_list = sort(normal_sct_list)
    merged_list = eltype(sorted_list)[]
    for sct in sorted_list
        if !isempty(merged_list) && sct == merged_list[end]
            merged_list[end] += sct
        else
            push!(merged_list, sct)
        end
    end
    return merged_list
end
function sort_merge_scatlist(normal_sct_list::Vector{Scatter};
    check_normal::Bool = true, check_normalupper::Bool = true
    )::Vector{Vector{<:Scatter}}

    if check_normalupper
        @assert all(isnormalupper, normal_sct_list) "All Scatter terms must be in normal order and in the upper triangular."
    elseif check_normal
        @assert all(isnormal, normal_sct_list) "All Scatter terms must be in normal order."
    end

    body_count = Int64[]
    merged_list = Vector{<:Scatter}[]
    for s in normal_sct_list
        N = get_body(s)
        i = findfirst(body_count .== N)
        if isnothing(i)
            push!(body_count, N)
            push!(merged_list, Scatter{N}[])
            i = length(body_count)
        end
        push!(merged_list[i], s)
    end
    for i in eachindex(merged_list)
        merged_list[i] = sort_merge_scatlist(merged_list[i]; 
            check_normal = false, check_normalupper = false)
    end
    return merged_list
end
function sort_merge_scatlist(normal_sct_lists::Vector{Vector{<:Scatter}};
    check_normal::Bool = true, check_normalupper::Bool = true
    )::Vector{Vector{<:Scatter}}

    body_count = Int64[]
    merged_lists = Vector{<:Scatter}[]
    for scat_list in normal_sct_lists

        if length(scat_list) == 0
            continue
        end

        if eltype(scat_list) == Scatter
            div_lists = sort_merge_scatlist(scat_list; check_normal = false, check_normalupper = false)
            for div_list in div_lists
                N = get_body(div_list[1])
                i = findfirst(body_count .== N)
                if isnothing(i)
                    push!(body_count, N)
                    push!(merged_lists, Scatter{N}[])
                    i = length(body_count)
                end
                append!(merged_lists[i], div_list)
            end
        else
            N = get_body(scat_list[1])
            i = findfirst(body_count .== N)
            if isnothing(i)
                push!(body_count, N)
                push!(merged_lists, Scatter{N}[])
                i = length(body_count)
            end
            append!(merged_lists[i], scat_list)
        end
    end
    for i in eachindex(merged_lists)
        merged_lists[i] = sort_merge_scatlist(merged_lists[i]; check_normal, check_normalupper)
    end
    return merged_lists
end