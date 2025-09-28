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
"""
struct MBS64{bits} <: Integer
    n::UInt64
    
    function MBS64{bits}(state::UInt64) where {bits}
        @assert bits isa Integer && 0 < bits <= 64 "The number of bits must be an integer between 1 and 64."
        if bits < 64
            @assert state < (UInt64(1) << bits) "State representation out of range for given bits"
        else
            # For 64 bits, any UInt64 value is valid
            @assert state <= typemax(UInt64) "State representation out of range for given bits"
        end
        new{bits}(state)
    end
end

"""
    Base.show(io::IO, mbs::MBS64{bits}) where bits

Display the MBS64 state in a human-readable format showing the bit pattern.
"""
function Base.show(io::IO, mbs::MBS64{bits}) where bits
    print(io, "MBS64: ", mbs.n, " = ", view(reverse(bitstring(mbs.n)), 1:bits), " ($bits bits)")
    if !isempty(findall(==('1'), view(reverse(bitstring(mbs.n)), bits+1:64)))
        println(io, " (Unphysical bits are occupied in MBS64.)")
        @warn "Unphysical bits are occupied in MBS64."
    end
end

# Basic operations
import Base: *, <, ==

"""
    *(mbs1::MBS64{b1}, mbs2::MBS64{b2}) where {b1, b2}

Concatenate two MBS64 states by shifting the first state and ORing with the second.
Used for combining states from different components.
"""
function *(mbs1::MBS64{b1}, mbs2::MBS64{b2}) where {b1, b2}
    MBS64{b1+b2}(mbs1.n << b2 | mbs2.n)
end

"""
    <(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}

Comparison operators for sorting MBS64 states.
"""
<(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b} = mbs1.n < mbs2.n


"""
    ==(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}

Check equality of two MBS64 states. Different bit sizes are never equal.
"""
function ==(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}
    mbs1.n == mbs2.n
end

# Occupation and state manipulation

"""
    occ_list(mbs::MBS64{bits}) where {bits}

Return the list of occupied orbital indices (1-based) in the many-body state.
"""
function occ_list(mbs::MBS64{bits}) where {bits}
    return findall(==('1'), view(reverse(bitstring(mbs.n)), 1:bits))
end

"""
    MBS64(bits, occ_list::Int64...)

Construct an MBS64 from a list of occupied orbital indices (1-based).
"""
function MBS64(bits, occ_list::Int64...)
    state = UInt64(0)
    for i in occ_list
        @assert 1 <= i <= bits "Occupied state index out of bounds"
        state |= UInt64(1) << (i - 1)
    end
    return MBS64{bits}(state)
end

"""
    isoccupied(mbs::MBS64{bits}, i_list::Int64...) where {bits}

Check if the specified orbital(s) are all occupied in the many-body state.
Returns true if all specified orbitals are occupied.
"""
function isoccupied(mbs::MBS64{bits}, i_list::Int64...) where {bits}
    mask = MBS64(bits, i_list...)
    return mbs.n & mask.n == mask.n
end

"""
    isempty(mbs::MBS64{bits}, i_list::Int64...) where {bits}

Check if the specified orbital(s) are all empty in the many-body state.
Returns true if all specified orbitals are empty.
"""
function Base.isempty(mbs::MBS64{bits}, i_list::Int64...) where {bits}
    mask = MBS64(bits, i_list...)
    return mbs.n & mask.n == 0
end

"""
    occupy!(mbs::MBS64{bits}, i_list::Int64...; check::Bool=true) where {bits}

Create a new MBS64 with the specified orbital(s) occupied.
If check=true, verifies that the orbitals were originally empty.
"""
function occupy!(mbs::MBS64{bits}, i_list::Int64...; check::Bool=true) where {bits}
    mask = MBS64(bits, i_list...)
    if check
        @assert mbs.n & mask.n == 0 "Some orbitals are already occupied."
    end
    return MBS64{bits}(mbs.n | mask.n)
end

"""
    empty!(mbs::MBS64{bits}, i_list::Int64...; check::Bool=true) where {bits}

Create a new MBS64 with the specified orbital(s) emptied.
If check=true, verifies that the orbitals were originally occupied.
"""
function Base.empty!(mbs::MBS64{bits}, i_list::Int64...; check::Bool=true) where {bits}
    mask = MBS64(bits, i_list...)
    if check
        @assert mbs.n & mask.n == mask.n "Some orbitals are already empty."
    end
    return MBS64{bits}(mbs.n & ~mask.n)
end

"""
    scat_occ_number(mbs::MBS64{bits}, i_list::Union{Vector{Int64}, NTuple{N, Int64}}) where {bits}

Count the total number of occupied orbitals that contribute to the sign flip when applying a series of creation/annihilation operators.
When a Scattering{N} object is applied, the number of sign flips should be the sum of 
applying the creation i_list and annihilation i_list on the middle state.
"""
function scat_occ_number(mbs::MBS64{bits}, i_list::Vector{Int64}) where {bits}

    i_list = sort(i_list) # sort from small to large
    N = length(i_list) # number of operators
    if N == 0
        return 0
    end

    @assert i_list[end] <= bits "Invalid bit positions"
    @assert 1 <= i_list[1] "Invalid bit positions"

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
scat_occ_number(mbs::MBS64, i_list::Tuple{Vararg{Int64}}) = scat_occ_number(mbs, collect(i_list))

