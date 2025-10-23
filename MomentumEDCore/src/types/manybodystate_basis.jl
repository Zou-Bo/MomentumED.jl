
# using Combinatorics

"""
    MBS64{bits} <: Integer

Many-body state representation using 64-bit unsigned integers.

Each bit represents the occupation of an orbital (1 = occupied, 0 = empty).
The `bits` parameter specifies how many orbitals are physically meaningful.

# Fields
- `state::UInt64`: The bit representation of occupied orbitals

# Constructor
    MBS64{bits}(state::UInt64) where {bits}

Creates a new MBS64 with the given bit state, validating that the state
fits within the specified number of bits.

# Reinterpret Construction Example
    reinterpret(MBS64{20}, UInt(250))

This may generate unphysical state.
"""
struct MBS64{bits}
    n::UInt64
    
    function MBS64{bits}(state::UInt64) where {bits}
        @assert bits isa Integer && 0 <= bits <= 64 "The number of bits must be an integer between 0 and 64."
        @assert state <= (UInt64(1) << bits - UInt64(1)) "State representation out of range for given bits"
        new{bits}(state)
    end
end

"""
    Base.show(io::IO, mbs::MBS64{bits}) where bits

Display the MBS64 state in a human-readable format showing the bit pattern.
"""
function Base.show(io::IO, mbs::MBS64{bits}) where bits
    bs = reverse(bitstring(mbs.n))
    print(io, "MBS64: ", mbs.n, " = ", view(bs, 1:bits), " ($bits bits)")
    if !isempty(findall(==('1'), view(bs, bits+1:64)))
        println(io, " (Unphysical bits are occupied in MBS64.)")
        @warn "Unphysical bits are occupied in MBS64."
    end
end

"""
    get_bits(mbs::MBS64{bits})

Return the bits of the type of input mbs state.
"""
function get_bits(::MBS64{bits})::Integer where{bits}
    bits
end

"""
    isphysical(mbs::MBS64{bits})::Bool

Check if all indices of occupied orbitals are smaller than or eaual to bits.
"""
function isphysical(mbs::MBS64{bits})::Bool where{bits}
    if bits < 64
        mbs.n < (UInt64(1) << bits) && return true
    else
        mbs.n <= typemax(UInt64) && return true
    end
    return false
end

# Basic operations
import Base: *, +, ==, hash, isless

"""
    *(mbs1::MBS64{b1}, mbs2::MBS64{b2}) where {b1, b2}

Combine(concatenate) two MBS64 states from two orthogonal Hilbert spaces.
The first state accounts smaller bit positions and the second greater positions.
Used for combining states from different components.
"""
function *(mbs1::MBS64{b1}, mbs2::MBS64{b2}) where {b1, b2}
    MBS64{b1+b2}(mbs1.n << b2 | mbs2.n)
end

"""
    +(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}

Combine(plus) two MBS64 states in the same(combined) Hilbert space with mbs1.n | mbs2.n.
Used for combining states generated with complete masks.
Repeated occupations on the same bit, if exist, count once.
"""
function +(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}
    MBS64{b}(mbs1.n | mbs2.n)
end

"""
    isless(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}

Comparison operators for sorting MBS64 states.
"""
Base.isless(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b} = mbs1.n < mbs2.n


"""
    ==(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}

Check equality of two MBS64 states. Different bit sizes are never equal.
"""
function ==(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}
    mbs1.n == mbs2.n
end

==(mbs::MBS64, i::Integer) = mbs.n == i
==(i::Integer, mbs::MBS64) = mbs.n == i

hash(mbs::MBS64) = hash(mbs.n)

# Occupation and state manipulation

"""
    occ_list(mbs::MBS64{bits}) where {bits}

Return the list of occupied orbital indices (1-based) in the many-body state.
"""
function occ_list(mbs::MBS64{bits}) where {bits}
    return findall(==('1'), view(reverse(bitstring(mbs.n)), 1:bits))
end

"""
    make_mask64(occ_list::Vector{Int64})::UInt64
    make_mask64(occ_list)::UInt64 = make_mask64(collect(occ_list))
    make_mask64(occ_list::Tuple{Int64})::UInt64
    make_mask64(occ_list::Tuple{Int64, Int64})::UInt64

Create mask with ones on the assigned bit positions.
Bit positions should be >= 1 and <= 64.
Repeating positions only take effect once.

Optimized for one and two occupations.
"""
function make_mask64(occ_list::Vector{Int64})::UInt64
    mask = UInt64(0)
    for i in occ_list
        @assert 1 <= i <= 64
        mask |= UInt64(1) << (i - 1)
    end
    return mask
end
@inline function make_mask64(occ_list)::UInt64 
    @assert typeof(occ_list) != Tuple{Int64}
    @assert typeof(occ_list) != Tuple{Int64, Int64}
    make_mask64(collect(occ_list))
end
function make_mask64(occ_list::Tuple{Int64})::UInt64
    @assert 1 <= occ_list[1] <= 64
    return UInt64(1) << (occ_list[1] - 1)
end
function make_mask64(occ_list::Tuple{Int64, Int64})::UInt64
    @assert 1 <= occ_list[1] <= 64
    @assert 1 <= occ_list[2] <= 64
    return UInt64(1) << (occ_list[1] - 1) | UInt64(1) << (occ_list[2] - 1)
end

"""
    MBS64(bits, occ_list[, mask])

Construct an MBS64 from a iteratable list of occupied orbital. 
No repetition allowed.

If a mask is specified, occ_list refers to the positions in the mask.
"""
function MBS64(bits, occ_list)
    sort!(occ_list)
    @assert allunique(occ_list)
    MBS64{bits}(make_mask64(occ_list))
end
MBS64(bits, occ_list, mask) = MBS64(bits, mask[occ_list])

"""
    Input a MBS64{bits} mask; return its complete mask.
"""
function MBS64_complete(mask::MBS64{bits})::MBS64{bits} where {bits}
    MBS64{bits}(UInt64(1) << bits -1 - mask.n)
end

"""
    isoccupied(mbs::MBS64{bits}, i_list::Vector{Int64}) where {bits}
    isoccupied(mbs::MBS64{bits}, i_list::Tuple{Vararg{Int64}}) where {bits}

Check if the specified orbital(s) are all occupied in the many-body state.
Returns true if all specified orbitals are occupied.
"""
function isoccupied(mbs::MBS64{bits}, i_list::Vector{Int64}) where {bits}
    mask = make_mask64(i_list)
    return mbs.n & mask == mask
end
function isoccupied(mbs::MBS64{bits}, i_list::Tuple{Vararg{Int64}}) where {bits}
    mask = make_mask64(i_list)
    return mbs.n & mask == mask
end
function isoccupied(mbs::MBS64{bits}, i_list::Tuple{Int64}) where {bits}
    mask = make_mask64(i_list)
    return mbs.n & mask == mask
end
function isoccupied(mbs::MBS64{bits}, i_list::Tuple{Int64, Int64}) where {bits}
    mask = make_mask64(i_list)
    return mbs.n & mask == mask
end

"""
    isempty(mbs::MBS64{bits}, i_list::Vector{Int64}) where {bits}
    isempty(mbs::MBS64{bits}, i_list::Tuple{Vararg{Int64}}) where {bits}

Check if the specified orbital(s) are all empty in the many-body state.
Returns true if all specified orbitals are empty.
"""
function Base.isempty(mbs::MBS64{bits}, i_list::Vector{Int64}) where {bits}
    mask = make_mask64(i_list)
    return mbs.n & mask == 0
end
function Base.isempty(mbs::MBS64{bits}, i_list::Tuple{Vararg{Int64}}) where {bits}
    mask = make_mask64(i_list)
    return mbs.n & mask == 0
end
function Base.isempty(mbs::MBS64{bits}, i_list::Tuple{Int64}) where {bits}
    mask = make_mask64(i_list)
    return mbs.n & mask == 0
end
function Base.isempty(mbs::MBS64{bits}, i_list::Tuple{Int64, Int64}) where {bits}
    mask = make_mask64(i_list)
    return mbs.n & mask == 0
end

"""
    occupy!(mbs::MBS64{bits}, i_list::Vector{Int64}; check::Bool=true) where {bits}
    occupy!(mbs::MBS64{bits}, i_list::Tuple{Vararg{Int64}}; check::Bool=true) where {bits}

Create a new MBS64 with the specified orbital(s) occupied.
If check=true, verifies that the orbitals were originally empty.
"""
function occupy!(mbs::MBS64{bits}, i_list::Vector{Int64}; check::Bool=true) where {bits}
    mask = make_mask64(i_list)
    if check
        @assert mbs.n & mask == 0 "Some orbitals are already occupied."
    end
    return MBS64{bits}(mbs.n | mask)
end
function occupy!(mbs::MBS64{bits}, i_list::Tuple{Vararg{Int64}}; check::Bool=true) where {bits}
    mask = make_mask64(i_list)
    if check
        @assert mbs.n & mask == 0 "Some orbitals are already occupied."
    end
    return MBS64{bits}(mbs.n | mask)
end
function occupy!(mbs::MBS64{bits}, i_list::Tuple{Int64}; check::Bool=true) where {bits}
    mask = make_mask64(i_list)
    if check
        @assert mbs.n & mask == 0 "Some orbitals are already occupied."
    end
    return MBS64{bits}(mbs.n | mask)
end
function occupy!(mbs::MBS64{bits}, i_list::Tuple{Int64, Int64}; check::Bool=true) where {bits}
    mask = make_mask64(i_list)
    if check
        @assert mbs.n & mask == 0 "Some orbitals are already occupied."
    end
    return MBS64{bits}(mbs.n | mask)
end

"""
    empty!(mbs::MBS64{bits}, i_list::Vector{Int64}; check::Bool=true) where {bits}
    empty!(mbs::MBS64{bits}, i_list::Tuple{Vararg{Int64}}; check::Bool=true) where {bits}

Create a new MBS64 with the specified orbital(s) emptied.
If check=true, verifies that the orbitals were originally occupied.
"""
function Base.empty!(mbs::MBS64{bits}, i_list::Vector{Int64}; check::Bool=true) where {bits}
    mask = make_mask64(i_list)
    if check
        @assert mbs.n & mask == mask "Some orbitals are already empty."
    end
    return MBS64{bits}(mbs.n & ~mask)
end
function Base.empty!(mbs::MBS64{bits}, i_list::Tuple{Vararg{Int64}}; check::Bool=true) where {bits}
    mask = make_mask64(i_list)
    if check
        @assert mbs.n & mask == mask "Some orbitals are already empty."
    end
    return MBS64{bits}(mbs.n & ~mask)
end
function Base.empty!(mbs::MBS64{bits}, i_list::Tuple{Int64}; check::Bool=true) where {bits}
    mask = make_mask64(i_list)
    if check
        @assert mbs.n & mask == mask "Some orbitals are already empty."
    end
    return MBS64{bits}(mbs.n & ~mask)
end
function Base.empty!(mbs::MBS64{bits}, i_list::Tuple{Int64, Int64}; check::Bool=true) where {bits}
    mask = make_mask64(i_list)
    if check
        @assert mbs.n & mask == mask "Some orbitals are already empty."
    end
    return MBS64{bits}(mbs.n & ~mask)
end

"""
    scat_occ_number(mbs::MBS64{bits}, i_list::Vector{Int64})::Int64 where {bits}
    scat_occ_number(mbs::MBS64{bits}, i_list::Tuple{Vararg{Int64}})::Int64 where {bits}
    scat_occ_number(mbs::MBS64{bits}, i_list::Tuple{Int64})::Int64 where {bits}
    scat_occ_number(mbs::MBS64{bits}, i_list::Tuple{Int64, Int64}})::Int64 where {bits}

Count the total number of occupied orbitals that contribute to the sign flip when applying a series of creation/annihilation operators.
When a Scatter{N} object is applied, the number of sign flips should be the sum of 
applying the creation i_list and annihilation i_list on the middle state.

Optimized for one-body and two-body Scatter with Tuple input.
"""
function scat_occ_number(mbs::MBS64{bits}, i_list::Vector{Int64})::Int64 where {bits}

    i_list = sort(i_list) # sort from small to large
    N = length(i_list) # number of operators
    if N == 0
        return 0
    end

    @boundscheck @assert 1 <= i_list[1] && i_list[end] <= bits "Invalid bit positions"

    mask = zero(UInt64)
    if isodd(N)
        push!(i_list, bits + 1) # the last segment goes to the end
    end
    for x in 1:2:N
        # count the occupation number in segment i_list[x]+1 : i_list[x+1]-1
        for i in i_list[x]+1:i_list[x+1]-1
            mask |= UInt64(1) << (i - 1)
        end
    end
    return count_ones(mbs.n & mask)

end
@inline function scat_occ_number(mbs::MBS64{bits}, i_list::Tuple{Vararg{Int64}})::Int64 where {bits}
    @assert typeof(i_list) != Tuple{Int64}
    @assert typeof(i_list) != Tuple{Int64, Int64}
    scat_occ_number(mbs, collect(i_list))
end
function scat_occ_number(mbs::MBS64{bits}, i_list::Tuple{Int64})::Int64 where {bits}

    i = i_list[1]
    @boundscheck @assert 1 <= i <= bits "Invalid bit positions"

    i == bits && return 0

    mask = typemax(UInt64) - UInt64(1) << (i) + UInt64(1)
    return count_ones(mbs.n & mask)

end
function scat_occ_number(mbs::MBS64{bits}, i_list::Tuple{Int64, Int64})::Int64 where {bits}

    imin, imax = extrema(i_list)
    @boundscheck @assert 1 <= imin && imax <= bits "Invalid bit positions"

    mask = UInt64(1) << (imax-1) - UInt64(1) << (imin)
    return count_ones(mbs.n & mask)

end

