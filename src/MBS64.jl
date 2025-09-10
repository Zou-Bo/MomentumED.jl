"""
MBS64 - Many-Body State 64-bit representation

This module provides the MBS64 type and associated functions for representing
many-body states using 64-bit unsigned integers where each bit represents
orbital occupation.
"""

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
    state::UInt64
    
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
    print(io, "MBS64: ", mbs.state, " = ", view(reverse(bitstring(mbs.state)), 1:bits), " ($bits bits)")
    if !isempty(findall(==('1'), view(reverse(bitstring(mbs.state)), bits+1:64)))
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
    MBS64{b1+b2}(mbs1.state << b2 | mbs2.state)
end

"""
    <(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}

Comparison operators for sorting MBS64 states.
"""
<(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b} = mbs1.state < mbs2.state


"""
    ==(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}

Check equality of two MBS64 states. Different bit sizes are never equal.
"""
function ==(mbs1::MBS64{b}, mbs2::MBS64{b}) where {b}
    mbs1.state == mbs2.state
end

# Occupation and state manipulation

"""
    occ_list(mbs::MBS64{bits}) where {bits}

Return the list of occupied orbital indices (1-based) in the many-body state.
"""
function occ_list(mbs::MBS64{bits}) where {bits}
    return findall(==('1'), view(reverse(bitstring(mbs.state)), 1:bits))
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
    return mbs.state & mask.state == mask.state
end

"""
    isempty(mbs::MBS64{bits}, i_list::Int64...) where {bits}

Check if the specified orbital(s) are all empty in the many-body state.
Returns true if all specified orbitals are empty.
"""
function Base.isempty(mbs::MBS64{bits}, i_list::Int64...) where {bits}
    mask = MBS64(bits, i_list...)
    return mbs.state & mask.state == 0
end

"""
    occupy!(mbs::MBS64{bits}, i_list::Int64...; check::Bool=true) where {bits}

Create a new MBS64 with the specified orbital(s) occupied.
If check=true, verifies that the orbitals were originally empty.
"""
function occupy!(mbs::MBS64{bits}, i_list::Int64...; check::Bool=true) where {bits}
    mask = MBS64(bits, i_list...)
    if check
        @assert mbs.state & mask.state == 0 "Some orbitals are already occupied."
    end
    return MBS64{bits}(mbs.state | mask.state)
end

"""
    empty!(mbs::MBS64{bits}, i_list::Int64...; check::Bool=true) where {bits}

Create a new MBS64 with the specified orbital(s) emptied.
If check=true, verifies that the orbitals were originally occupied.
"""
function Base.empty!(mbs::MBS64{bits}, i_list::Int64...; check::Bool=true) where {bits}
    mask = MBS64(bits, i_list...)
    if check
        @assert mbs.state & mask.state == mask.state "Some orbitals are already empty."
    end
    return MBS64{bits}(mbs.state & ~mask.state)
end

# These functions may be useless
#=
"""
    flip_orbital(mbs::MBS64{bits}, i::Int64) where {bits}

Flip the occupation of a single orbital (occupied -> empty, empty -> occupied).
"""
function flip_orbital(mbs::MBS64{bits}, i::Int64) where {bits}
    @assert 1 <= i <= bits "Orbital index out of bounds"
    return MBS64{bits}(xor(mbs.state, UInt64(1) << (i - 1)))
end

# Particle counting and statistics

"""
    count_particles(mbs::MBS64{bits}) where {bits}
    get_particle_number(mbs::MBS64{bits}) where {bits}

Count the total number of occupied orbitals (particles) in the state.
"""
function count_particles(mbs::MBS64{bits}) where {bits}
    count_ones(mbs.state)
end

const get_particle_number = count_particles


"""
    get_occupied_orbitals(mbs::MBS64{bits}) where {bits}

Return a vector of all occupied orbital indices (1-based).
"""
function get_occupied_orbitals(mbs::MBS64{bits}) where {bits}
    return occ_list(mbs)
end

"""
    get_empty_state(bits::Int)

Create an MBS64 representing the vacuum state (no particles).
"""
function get_empty_state(bits::Int)
    return MBS64{bits}(UInt64(0))
end

"""
    get_full_state(bits::Int)

Create an MBS64 representing the fully occupied state (all orbitals filled).
"""
function get_full_state(bits::Int)
    return MBS64{bits}((UInt64(1) << bits) - 1)
end


# State generation utilities

"""
    get_state_index(mbs::MBS64{bits}) where {bits}

Get the raw UInt64 representation of the state.
"""
function get_state_index(mbs::MBS64{bits}) where {bits}
    return mbs.state
end

"""
    set_state(mbs::MBS64{bits}, new_state::UInt64) where {bits}

Create a new MBS64 with the specified raw state value.
"""
function set_state(mbs::MBS64{bits}, new_state::UInt64) where {bits}
    return MBS64{bits}(new_state)
end


=#

"""
    occ_num_between(mbs::MBS64{bits}, i_start::Int64, i_end::Int64) where {bits}

Count the number of occupied orbitals between two bit positions (exclusive).
The range can be specified in either order.
"""
function occ_num_between(mbs::MBS64{bits}, i_start::Int64, i_end::Int64) where {bits}
    @assert 1 <= i_start <= bits "Invalid bit positions"
    @assert 1 <= i_end <= bits "Invalid bit positions"
    i_start, i_end = minmax(i_start, i_end) # allow inversely-ordered inputs
    mask = zero(UInt64)
    for i in i_start+1:i_end-1
        mask |= UInt64(1) << (i - 1)
    end
    return count_ones(mbs.state & mask)
end


# Momentum calculations with EDPara integration

"""
    MBS64_totalmomentum(para::EDPara, mbs::MBS64)

Calculate the total momentum (K1, K2) of a many-body state.
The momentum is mod G if G is nonzero (from para.Gk).
"""
function MBS64_totalmomentum(para::EDPara, mbs::MBS64)
    # momentum are integers
    k1 = 0; k2 = 0; Gk = para.Gk
    for i in occ_list(mbs)
        momentum = @view para.k_list[:, mod1(i, para.Nk)]
        k1 += momentum[1]
        k2 += momentum[2]
    end
    iszero(Gk[1]) || (k1 = mod(k1, Gk[1]))
    iszero(Gk[2]) || (k2 = mod(k2, Gk[2]))
    return k1, k2
end

"""
    MBS64_totalmomentum(para::EDPara, i_list::Int64...)

Calculate the total momentum (K1, K2) from a list of occupied orbital indices.
"""
function MBS64_totalmomentum(para::EDPara, i_list::Int64...)
    # momentum are integers
    k1 = 0; k2 = 0; Gk = para.Gk
    for i in i_list
        momentum = @view para.k_list[:, mod1(i, para.Nk)]
        k1 += momentum[1]
        k2 += momentum[2]
    end
    iszero(Gk[1]) || (k1 = mod(k1, Gk[1]))
    iszero(Gk[2]) || (k2 = mod(k2, Gk[2]))
    return k1, k2
end


# Export all public functions
export MBS64
export occ_list, isoccupied, isempty, occupy!, empty!, flip_orbital
export count_particles, get_particle_number, occ_num_between, get_occupied_orbitals
export get_empty_state, get_full_state
export MBS64_totalmomentum
export create_particle, annihilate_particle, c_dagger, c
export get_state_index, set_state