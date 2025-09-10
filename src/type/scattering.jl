

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
(3) j1 > i1 or j1 = i1 && j2 > i2 or j1,j2 = i1,i2 && j3 > i3 or ... or j1,...,jN-1 = i1,...,iN-1 && jN >= iN
"""
function NormalScattering(V, i, j)
    # N = 1
    
    # Apply normal ordering rules

    # j >= i
    if j < i
        j, i = i, j
        V = conj(V)
    end

    if j == i
        V = real(V) + 0im
    end

    return Scattering(V, i, j)
end
function NormalScattering(V, i1, i2, j2, j1)
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
    if j1 < i1 || (j1 == i1 && j2 < i2)
        j1, i1 = i1, j1
        j2, i2 = i2, j2
        V = conj(V)
    end

    if j1 == i1 && j2 == i2
        V = real(V) + 0im
    end
    # @show V, i1, i2, j2, j1
    return Scattering(V, i1, i2, j2, j1)
end
isnormal(x::Scattering{1}) = x.in >= x.out
isnormal(x::Scattering{2}) = x.in[1] > x.in[2] && x.out[1] > x.out[2] && x.in >= x.out




import Base: isless, ==, +
isless(x::Scattering{N}, y::Scattering{N}) where {N} = x.in < y.in || x.in == y.in && x.out < y.out
==(x::Scattering{N}, y::Scattering{N}) where {N} = x.in == y.in && x.out == y.out
function +(x::Scattering{N}, y::Scattering{N})::Scattering{N} where {N}
    @assert x == y "Can only add identical Scattering terms"
    return Scattering{N}(x.Amp + y.Amp, x.out, x.in)
end
"""
Sort and merge a list of normalized scattering terms
"""
function sortMergeScatteringList(normal_sct_list::Vector{Scattering{N}})::Vector{Scattering{N}} where {N}
    @assert N == 1 || N == 2 "Only normal ordering of 1-body and 2-body scattering terms are supported"
    @assert all(isnormal, normal_sct_list) "All scattering terms must be normalized"
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

