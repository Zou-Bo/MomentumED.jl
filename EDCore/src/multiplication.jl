using LinearAlgebra

import Base: *

"""
    scat * (amp_in::ComplexF64, mbs_in::MBS64) -> (amp_out::ComplexF64, mbs_out::MBS64)
    scat * mbs_in::MBS64 = scat * (1.0, mbs_in)

Applying a scatter operator on a many-body basis. Return the amplitute and the output many-body basis.

The amplitute is zero if no output state.

"""
function *(scat::Scatter{C, MBS64{bits}}, mbs_in::MBS64{bits})::Tuple{C, MBS64{bits}} where {C, bits} 
    if isoccupied(mbs_in, scat.in.n)
        if scat.in == scat.out
            return scat.Amp, mbs_in
        else
            mbs_mid = empty!(mbs_in, scat.in.n; check = false)
            if isempty(mbs_mid, scat.out.n)
                mbs_out = occupy!(mbs_mid, scat.out.n; check = false)
                if isodd(scat_occ_number(mbs_mid, scat.in.n) + scat_occ_number(mbs_mid, scat.out.n))
                    return -scat.Amp, mbs_out
                else
                    return scat.Amp, mbs_out
                end
            end
        end
    end
    return zero(C), mbs_in
end
function *(scat::Scatter{C, MBS64{bits}}, tuple_mbs_in::Tuple{C, MBS64{bits}})::Tuple{C, MBS64{bits}} where {C, bits}
    iszero(tuple_mbs_in[1]) && return tuple_mbs_in

    amp_in, mbs_in = tuple_mbs_in
    amp_out, mbs_out = scat * mbs_in
    return amp_in * amp_out, mbs_out
end

"""
    (amp_out::ComplexF64, mbs_out::MBS64) * scat -> (amp_in::ComplexF64, mbs_in::MBS64)
    mbs_out::MBS64 * scat = (1.0, mbs_out) * scat

Applying a scatter operator on a output many-body basis from the right.
Return the amplitute and the incident many-body basis.

The amplitute is zero if no incident state.

if (amp, mbs_in) == mbs_out * scat && !iszero(amp),  (amp, mbs_out) == scat * mbs_in.
if (amp_in, mbs_in) == (amp_out, mbs_out) * scat && !iszero(amp_in),  (amp_out, mbs_out) == scat * (amp_in, mbs_in).

"""
function *(mbs_out::MBS64{bits}, scat::Scatter{C, MBS64{bits}})::Tuple{C, MBS64{bits}} where {C, bits} 
    if isoccupied(mbs_out, scat.out.n)
        if scat.in == scat.out
            return scat.Amp, mbs_out
        else
            mbs_mid = empty!(mbs_out, scat.out.n; check = false)
            if isempty(mbs_mid, scat.in.n)
                mbs_in = occupy!(mbs_mid, scat.in.n; check = false)
                if isodd(scat_occ_number(mbs_mid, scat.in.n) + scat_occ_number(mbs_mid, scat.out.n))
                    return -scat.Amp, mbs_in
                else
                    return scat.Amp, mbs_in
                end
            end
        end
    end
    return zero(C), mbs_out
end
function *(tuple_mbs_out::Tuple{C, MBS64{bits}}, scat::Scatter{C, MBS64{bits}})::Tuple{C, MBS64{bits}} where {C, bits}
    iszero(amp_out) && return tuple_mbs_out
    
    amp_out, mbs_out = tuple_mbs_out
    amp_in, mbs_in = mbs_out * scat
    return amp_in * amp_out, mbs_in
end


"""
Multiply a `Vector{Scatter{N}}` list on MBS64Vector and add on the collect_result. 
N is known by compiler.

Core function for MBOperator * MBS64Vector.
It convinces the Julia compiler to use stack instead of heap.

Notice: no checking whether collect_result and mbs_vec are in the same Hilbert space
"""
function mul_add!(collect_result::MBS64Vector{bits, eltype},
    scat_list::Vector{Scatter{Complex{eltype}, MBS64{bits}}}, mbs_vec::MBS64Vector{bits, eltype},
    upper_hermitian::Bool) where{bits, eltype <: AbstractFloat}

    for scat in scat_list
        for (j, mbs_in) in enumerate(mbs_vec.space.list)
            amp, mbs_out = scat * mbs_in
            iszero(amp) && continue
            i = get(collect_result.space, mbs_out)
            if i != 0
                collect_result.vec[i] += amp * mbs_vec.vec[j]
                upper_hermitian && i != j && (collect_result.vec[j] += conj(amp) * mbs_vec.vec[i])
            else
                # @boundscheck throw(DimensionMismatch("The operator scatters the state out of its Hilbert subspace."))
            end
        end
    end
end

"""
Use mul_add!() to avoid heap allocation.
"""
function mul!(mbs_vec_result::MBS64Vector{bits, eltype}, 
    op::MBOperator{Complex{eltype}, MBS64{bits}}, mbs_vec::MBS64Vector{bits, eltype}
    ) where {bits, eltype <: AbstractFloat}
    
    mbs_vec_result.vec .= zero(Complex{eltype})
    mul_add!(mbs_vec_result, op.scats, mbs_vec, op.upper_hermitian)
end

"""
    op::MBOperator * mbs_vec::MBS64Vector -> mbs_vec_result::MBS64Vector
    op2::MBOperator * op1::MBOperator * mbs_vec::MBS64Vector -> mbs_vec_result::MBS64Vector

Applying one or two operators on a many-body state.
More number of operators is not supported and should be implemented in steps.
"""
function *(op::MBOperator{Complex{eltype}, MBS64{bits}}, mbs_vec_in::MBS64Vector{bits, eltype}
    )::MBS64Vector{bits, eltype} where {bits, eltype <: AbstractFloat}
    
    mbs_vec_out = similar(mbs_vec_in)
    mul!(mbs_vec_out, op, mbs_vec_in)
    return mbs_vec_out

end
function *(op2::MBOperator{Complex{eltype}, MBS64{bits}}, op1::MBOperator, mbs_vec_in::MBS64Vector{bits, eltype}
    )::MBS64Vector{bits, eltype} where {bits, eltype <: AbstractFloat}

    mbs_vec_out = similar(mbs_vec_in)
    mbs_vec_mid = similar(mbs_vec_in)
    mul!(mbs_vec_mid, op1, mbs_vec_in)
    mul!(mbs_vec_out, op2, mbs_vec_mid)
    return mbs_vec_out

end


"""
Bracket value of one or two Scatter{N} terms between two MBS64Vectors. 

Core function for ED_bracket(::MBS64Vector, [::MBOperator,] ::MBOperator, ::MBS64Vector).
It convinces the Julia compiler to use stack instead of heap even in multi-threading.

The two term version does not take in upper_hermitian keywords (must be op.upper_hermitian = false).

Notice: no checking whether mbs_bra and mbs_ket are in the same Hilbert space
"""
function mul_add_bracket(mbs_vec_bra::MBS64Vector{bits, eltype},
    scat::Scatter{Complex{eltype}, MBS64{bits}}, mbs_vec_ket::MBS64Vector{bits, eltype}, upper_hermitian::Bool
    )::Complex{eltype} where{bits, eltype <: AbstractFloat}

    collect_result = zero(Complex{eltype})
    for (j, mbs_in) in enumerate(mbs_vec_ket.space.list)
        amp, mbs_out = scat * mbs_in
        iszero(amp) && continue
        i = get(mbs_vec_bra.space, mbs_out)
        if iszero(i)
            # @boundscheck throw(DimensionMismatch("The operator does not scatter the incident state to the output state's Hilbert subspace."))
        else
            collect_result += conj(mbs_vec_bra.vec[i]) * amp * mbs_vec_ket.vec[j]
            if upper_hermitian && i != j
                collect_result += conj(mbs_vec_bra.vec[j]) * conj(amp) * mbs_vec_ket.vec[i]
            end
        end
    end
    return collect_result
end
function mul_add_bracket(mbs_vec_bra::MBS64Vector{bits, eltype},
    scat2::Scatter{Complex{eltype}, MBS64{bits}}, scat1::Scatter{Complex{eltype}, MBS64{bits}}, 
    mbs_vec_ket::MBS64Vector{bits, eltype})::Complex{eltype} where{bits, eltype <: AbstractFloat}

    collect_result = zero(Complex{eltype})
    for (j, mbs_in) in enumerate(mbs_vec_ket.space.list)
        amp, mbs_out = scat2 * (scat1 * mbs_in)
        iszero(amp) && continue         # necessary for multiple Scatter without middle dictionary
        i = get(mbs_vec_bra.space, mbs_out)
        if iszero(i)
            # @boundscheck throw(DimensionMismatch("The operators do not scatter the incident state to the output state's Hilbert subspace."))
        else
            collect_result += conj(mbs_vec_bra.vec[i]) * amp * mbs_vec_ket.vec[j]
        end
    end
    return collect_result
end

"""
Bracket value of a Vector of Scatter{N}, summing over bracket value of each Scatter{N} term.

Can be converted to the multi-threaded version.
"""
function mul_add_bracket_scatlist(mbs_vec_bra::MBS64Vector{bits, eltype},
    scat_list::Vector{Scatter{Complex{eltype}, MBS64{bits}}}, mbs_vec_ket::MBS64Vector{bits, eltype}, 
    upper_hermitian::Bool)::Complex{eltype} where{bits, eltype <: AbstractFloat}

    sum(scat_list) do scat
        mul_add_bracket(mbs_vec_bra, scat, mbs_vec_ket, upper_hermitian)
    end

end
function mul_add_bracket_scatlist_threaded(mbs_vec_bra::MBS64Vector{bits, eltype},
    scat_list::Vector{Scatter{Complex{eltype}, MBS64{bits}}}, mbs_vec_ket::MBS64Vector{bits, eltype},
    upper_hermitian::Bool)::Complex{eltype} where{bits, eltype <: AbstractFloat}

    # A vector to store the partial result from each thread
    thread_brackets = zeros(Complex{eltype}, Threads.nthreads())

    Threads.@threads for scat in scat_list
        tid = Threads.threadid() - Threads.nthreads(:interactive)
        thread_brackets[tid] += mul_add_bracket(mbs_vec_bra, scat, mbs_vec_ket, upper_hermitian)
    end

    return sum(thread_brackets, init = zero(Complex{eltype}))
end
function mul_add_bracket_scatlist(mbs_vec_bra::MBS64Vector{bits, eltype},
    scat_list2::Vector{Scatter{Complex{eltype}, MBS64{bits}}}, scat_list1::Vector{Scatter{Complex{eltype}, MBS64{bits}}},
    mbs_vec_ket::MBS64Vector{bits, eltype})::Complex{eltype} where{bits, eltype <: AbstractFloat}

    collect_result = zero(Complex{eltype})
    for scat2 in scat_list2, scat1 in scat_list1
        collect_result += mul_add_bracket(mbs_vec_bra, scat2, scat1, mbs_vec_ket)
    end
    return collect_result
end
function mul_add_bracket_scatlist_threaded(mbs_vec_bra::MBS64Vector{bits, eltype},
    scat_list2::Vector{Scatter{Complex{eltype}, MBS64{bits}}}, scat_list1::Vector{Scatter{Complex{eltype}, MBS64{bits}}},
    mbs_vec_ket::MBS64Vector{bits, eltype})::Complex{eltype} where{bits, eltype <: AbstractFloat}

    # A vector to store the partial result from each thread
    thread_brackets = zeros(Complex{eltype}, Threads.nthreads())

    Threads.@threads for scat2 in scat_list2
        tid = Threads.threadid() - Threads.nthreads(:interactive)
        for scat1 in scat_list1
            thread_brackets[tid] += mul_add_bracket(mbs_vec_bra, scat2, scat1, mbs_vec_ket)
        end
    end
    
    return sum(thread_brackets, init = zero(Complex{eltype}))

end


"""
    ED_bracket(mbs_vec_bra::MBS64Vector{bits, eltype}, 
        op::MBOperator, mbs_vec_ket::MBS64Vector{bits, eltype}
        )::Complex{eltype} where {bits, eltype <: AbstractFloat}
    ED_bracket(mbs_vec_bra::MBS64Vector{bits, eltype}, 
        op2::MBOperator, op1::MBOperator, mbs_vec_ket::MBS64Vector{bits, eltype}
        )::Complex{eltype} where {bits, eltype <: AbstractFloat}

Compute the bracket <bra::MBS64Vector | op::MBOperator | ket::MBS64Vector>\n
or the bracket <bra::MBS64Vector | op2::MBOperator * op1::MBOperator | ket::MBS64Vector>
"""
function ED_bracket(mbs_vec_bra::MBS64Vector{bits, eltype}, 
    op::MBOperator{Complex{eltype}, MBS64{bits}}, mbs_vec_ket::MBS64Vector{bits, eltype}
    )::Complex{eltype} where {bits, eltype <: AbstractFloat}

    bracket = mul_add_bracket_scatlist(mbs_vec_bra, op.scats, mbs_vec_ket, op.upper_hermitian)
    return bracket
end
function ED_bracket(mbs_vec_bra::MBS64Vector{bits, eltype}, 
    op2::MBOperator{Complex{eltype}, MBS64{bits}}, op1::MBOperator{Complex{eltype}, MBS64{bits}}, 
    mbs_vec_ket::MBS64Vector{bits, eltype})::Complex{eltype} where {bits, eltype <: AbstractFloat}

    if op1.upper_hermitian || op2.upper_hermitian
        println("one-by-one multiplication")
        return ED_bracket(mbs_vec_bra, op2, op1*mbs_vec_ket)
    end 

    bracket = mul_add_bracket_scatlist(mbs_vec_bra, op2.scats, op1.scats, mbs_vec_ket)
    return bracket
end
"""
    ED_bracket_threaded(mbs_vec_bra::MBS64Vector{bits, eltype}, 
        op::MBOperator, mbs_vec_ket::MBS64Vector{bits, eltype}
        )::Complex{eltype} where {bits, eltype <: AbstractFloat}
    ED_bracket_threaded(mbs_vec_bra::MBS64Vector{bits, eltype}, 
        op2::MBOperator, op1::MBOperator, mbs_vec_ket::MBS64Vector{bits, eltype}
        )::Complex{eltype} where {bits, eltype <: AbstractFloat}

(Multithread version.) Compute the bracket <bra::MBS64Vector | op::MBOperator | ket::MBS64Vector>\n
or the bracket <bra::MBS64Vector | op2::MBOperator * op1::MBOperator | ket::MBS64Vector>
"""
function ED_bracket_threaded(mbs_vec_bra::MBS64Vector{bits, eltype},
    op::MBOperator{Complex{eltype}, MBS64{bits}}, mbs_vec_ket::MBS64Vector{bits, eltype}
    )::Complex{eltype} where {bits, eltype <: AbstractFloat}

    bracket = mul_add_bracket_scatlist_threaded(mbs_vec_bra, op.scats, mbs_vec_ket, op.upper_hermitian)
    return bracket
end
function ED_bracket_threaded(mbs_vec_bra::MBS64Vector{bits, eltype},
    op2::MBOperator{Complex{eltype}, MBS64{bits}}, op1::MBOperator{Complex{eltype}, MBS64{bits}}, 
    mbs_vec_ket::MBS64Vector{bits, eltype})::Complex{eltype} where {bits, eltype <: AbstractFloat}
    
    if op1.upper_hermitian || op2.upper_hermitian
        println("one-by-one multiplication")
        return ED_bracket(mbs_vec_bra, op2, op1*mbs_vec_ket)
    end
    
    bracket = mul_add_bracket_scatlist_threaded(mbs_vec_bra, op2.scats, op1.scats, mbs_vec_ket)
    return bracket
end
