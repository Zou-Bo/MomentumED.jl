using LinearAlgebra

import Base: *

"""
    scat * (amp_in::ComplexF64, mbs_in::MBS64) -> (amp_out::ComplexF64, mbs_out::MBS64)
    scat * mbs_in::MBS64 = scat * (1.0, mbs_in)

Applying a scatter operator on a many-body basis. Return the amplitute and the output many-body basis.

The amplitute is zero if no output state.

Notice: this multiplication assumes `scat` is normal.
Using abnormal `Scatter` term will generate same result as if its indices are sorted,
which may cause a sign error.
It is recommended to always use `NormalScatter` to create `Scatter` terms.
"""
function *(scat::Scatter{N}, mbs_in::MBS64{bits})::Tuple{ComplexF64, MBS64{bits}} where {N, bits} 
    
    @boundscheck @assert scat.in[1] <= bits && scat.out[1] <= bits "The scat operator change mbs bits beyond physical limit."

    if isoccupied(mbs_in, scat.in)
        if scat.in == scat.out
            return scat.Amp, mbs_in
        else
            mbs_mid = empty!(mbs_in, scat.in; check = false)
            if isempty(mbs_mid, scat.out)
                mbs_out = occupy!(mbs_mid, scat.out; check = false)
                amp = scat.Amp
                if isodd(scat_occ_number(mbs_mid, scat.in) + scat_occ_number(mbs_mid, scat.out))
                    amp = -amp
                end
                return amp, mbs_out
            end
        end
    end
    return ComplexF64(0.0), mbs_in
end
function *(scat::Scatter{N}, tuple_mbs_in::Tuple{ComplexF64, MBS64{bits}})::Tuple{ComplexF64, MBS64{bits}} where {N, bits}
    
    @boundscheck @assert scat.in[1] <= bits && scat.out[1] <= bits "The scat operator change mbs bits beyond physical limit."

    amp_in, mbs_in = tuple_mbs_in
    iszero(amp_in) && return tuple_mbs_in

    @inbounds amp_out, mbs_out = scat * mbs_in
    return amp_in * amp_out, mbs_out
end

"""
    (amp_out::ComplexF64, mbs_out::MBS64) * scat -> (amp_in::ComplexF64, mbs_in::MBS64)
    mbs_out::MBS64 * scat = (1.0, mbs_out) * scat

Applying a scatter operator on a output many-body basis from the right.
Return the amplitute and the incident many-body basis.

The amplitute is zero if no incident state.

Notice: this multiplication assumes `scat` is normal.
Using abnormal `Scatter` term will generate same result as if its indices are sorted,
which may cause a sign error.
It is recommended to always use `NormalScatter` to create `Scatter` terms.
"""
function *(mbs_out::MBS64{bits}, scat::Scatter{N})::Tuple{ComplexF64, MBS64{bits}} where {N, bits} 
    
    @boundscheck @assert scat.in[1] <= bits && scat.out[1] <= bits "The scat operator change mbs bits beyond physical limit."

    if isoccupied(mbs_out, scat.out)
        if scat.in == scat.out
            return scat.Amp, mbs_out
        else
            mbs_mid = empty!(mbs_out, scat.out; check = false)
            if isempty(mbs_mid, scat.in)
                mbs_in = occupy!(mbs_mid, scat.in; check = false)
                amp = scat.Amp
                if isodd(scat_occ_number(mbs_mid, scat.in) + scat_occ_number(mbs_mid, scat.out))
                    amp = -amp
                end
                return amp, mbs_in
            end
        end
    end
    return ComplexF64(0.0), mbs_out
end
function *(tuple_mbs_out::Tuple{ComplexF64, MBS64{bits}}, scat::Scatter{N})::Tuple{ComplexF64, MBS64{bits}} where {N, bits}
    
    @boundscheck @assert scat.in[1] <= bits && scat.out[1] <= bits "The scat operator change mbs bits beyond physical limit."

    amp_out, mbs_out = tuple_mbs_out
    iszero(amp_out) && return tuple_mbs_out

    @inbounds amp_in, mbs_in = mbs_out * scat
    return amp_in * amp_out, mbs_in
end


"""
Multiply a Scatter{N} term on MBS64Vector and add on the collect_result. 

Core function for MBOperator * MBS64Vector.
It convinces the Julia compiler to use stack instead of heap.

Notice: no checking whether collect_result and mbs_vec are in the same Hilbert space
"""
function mul_add!(collect_result::MBS64Vector{bits, eltype},
    scat::Scatter{N}, mbs_vec::MBS64Vector{bits, eltype},
    upper_hermitian::Bool) where{N, bits, eltype <: AbstractFloat}

    # for (mbs_in, j) in mbs_vec.space.dict
    for (j, mbs_in) in enumerate(mbs_vec.space.list)
        amp, mbs_out = scat * mbs_in
        iszero(amp) && continue
        i = get(collect_result.space, mbs_out)
        @boundscheck if iszero(i)
            throw(DimensionMismatch("The operator scatters the state out of its Hilbert subspace."))
        end
        collect_result.vec[i] += amp * mbs_vec.vec[j]
        upper_hermitian && i != j && (collect_result.vec[j] += conj(amp) * mbs_vec.vec[i])
    end
end

"""
Use mul_add!() to avoid heap allocation.
"""
function mul!(mbs_vec_result::MBS64Vector{bits, eltype}, 
    op::MBOperator, mbs_vec::MBS64Vector{bits, eltype}
    ) where {bits, eltype <: AbstractFloat}
    
    mbs_vec_result.vec .= 0.0
    for scat_list in op.scats
        for scat in scat_list
            mul_add!(mbs_vec_result, scat, mbs_vec, op.upper_hermitian)
        end
    end
end

"""
    op::MBOperator * mbs_vec::MBS64Vector -> mbs_vec_result::MBS64Vector
    op2::MBOperator * op1::MBOperator * mbs_vec::MBS64Vector -> mbs_vec_result::MBS64Vector

Applying one or two operators on a many-body state.
More number of operators is not supported and should be implemented in steps.
"""
function *(op::MBOperator, mbs_vec_in::MBS64Vector{bits, eltype}
    )::MBS64Vector{bits, eltype} where {bits, eltype <: AbstractFloat}
    
    mbs_vec_out = similar(mbs_vec_in)
    mul!(mbs_vec_out, op, mbs_vec_in)
    return mbs_vec_out

end
function *(op2::MBOperator, op1::MBOperator, mbs_vec_in::MBS64Vector{bits, eltype}
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
    scat::T, mbs_vec_ket::MBS64Vector{bits, eltype}, upper_hermitian::Bool
    )::Complex{eltype} where{bits, eltype <: AbstractFloat, T <: Scatter}

    collect_result::Complex{eltype} = 0.0
    for (j, mbs_in) in enumerate(mbs_vec_ket.space.list)
        amp, mbs_out = scat * mbs_in
        iszero(amp) && continue
        i = get(mbs_vec_bra.space, mbs_out)
        if iszero(i)
            @boundscheck throw(DimensionMismatch("The operator does not scatter the incident state to the output state's Hilbert subspace."))
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
    scat2::T2, scat1::T1, mbs_vec_ket::MBS64Vector{bits, eltype},
    )::Complex{eltype} where{bits, eltype <: AbstractFloat, T2 <: Scatter, T1 <:Scatter}

    collect_result = Complex{eltype}(0.0)
    for (j, mbs_in) in enumerate(mbs_vec_ket.space.list)
        amp, mbs_out = scat2 * (scat1 * mbs_in)
        iszero(amp) && continue         # necessary for multiple Scatter without middle dictionary
        i = get(mbs_vec_bra.space, mbs_out)
        if iszero(i)
            @boundscheck throw(DimensionMismatch("The operators do not scatter the incident state to the output state's Hilbert subspace."))
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
    scat_list::Vector{T}, mbs_vec_ket::MBS64Vector{bits, eltype}, upper_hermitian::Bool
    )::Complex{eltype} where{bits, eltype <: AbstractFloat, T <: Scatter}

    sum(scat_list) do scat
        mul_add_bracket(mbs_vec_bra, scat, mbs_vec_ket, upper_hermitian)
    end

end
function mul_add_bracket_scatlist_threaded(mbs_vec_bra::MBS64Vector{bits, eltype},
    scat_list::Vector{T}, mbs_vec_ket::MBS64Vector{bits, eltype}, upper_hermitian::Bool
    )::Complex{eltype} where{bits, eltype <: AbstractFloat, T <: Scatter}

    # A vector to store the partial result from each thread
    thread_brackets = zeros(Complex{eltype}, Threads.nthreads())

    Threads.@threads for scat in scat_list
        tid = Threads.threadid() - Threads.nthreads(:interactive)
        thread_brackets[tid] += mul_add_bracket(mbs_vec_bra, scat, mbs_vec_ket, upper_hermitian)
    end

    return sum(thread_brackets, init = Complex{eltype}(0.0))
end
function mul_add_bracket_scatlist(mbs_vec_bra::MBS64Vector{bits, eltype},
    scat_list2::Vector{T2}, scat_list1::Vector{T1}, mbs_vec_ket::MBS64Vector{bits, eltype},
    )::Complex{eltype} where{bits, eltype <: AbstractFloat, T2 <: Scatter, T1 <:Scatter}

    collect_result = Complex{eltype}(0.0)
    for scat2 in scat_list2, scat1 in scat_list1
        collect_result += mul_add_bracket(mbs_vec_bra, scat2, scat1, mbs_vec_ket)
    end
    return collect_result
end
function mul_add_bracket_scatlist_threaded(mbs_vec_bra::MBS64Vector{bits, eltype},
    scat_list2::Vector{T2}, scat_list1::Vector{T1}, mbs_vec_ket::MBS64Vector{bits, eltype},
    )::Complex{eltype} where{bits, eltype <: AbstractFloat, T2 <: Scatter, T1 <:Scatter}

    # A vector to store the partial result from each thread
    thread_brackets = zeros(Complex{eltype}, Threads.nthreads())

    Threads.@threads for scat2 in scat_list2
        tid = Threads.threadid() - Threads.nthreads(:interactive)
        for scat1 in scat_list1
            thread_brackets[tid] += mul_add_bracket(mbs_vec_bra, scat2, scat1, mbs_vec_ket)
        end
    end
    
    return sum(thread_brackets, init = Complex{eltype}(0.0))

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
    op::MBOperator, mbs_vec_ket::MBS64Vector{bits, eltype}
    )::Complex{eltype} where {bits, eltype <: AbstractFloat}

    bracket = Complex{eltype}(0.0)
    for scat_list in op.scats
        bracket += mul_add_bracket_scatlist(mbs_vec_bra, scat_list, mbs_vec_ket, op.upper_hermitian)
    end
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
    op::MBOperator, mbs_vec_ket::MBS64Vector{bits, eltype}
    )::Complex{eltype} where {bits, eltype <: AbstractFloat}

    bracket = Complex{eltype}(0.0)
    for scat_list in op.scats
        bracket += mul_add_bracket_scatlist_threaded(mbs_vec_bra, scat_list, mbs_vec_ket, op.upper_hermitian)
    end
    return bracket
end
function ED_bracket(mbs_vec_bra::MBS64Vector{bits, eltype}, 
    op2::MBOperator, op1::MBOperator, mbs_vec_ket::MBS64Vector{bits, eltype}
    )::Complex{eltype} where {bits, eltype <: AbstractFloat}

    if op1.upper_hermitian || op2.upper_hermitian
        println("one-by-one multiplication")
        return ED_bracket(mbs_vec_bra, op2, op1*mbs_vec_ket)
    end 

    bracket = Complex{eltype}(0.0)
    for scat_list1 in op1.scats, scat_list2 in op2.scats
        bracket += mul_add_bracket_scatlist(mbs_vec_bra, scat_list2, scat_list1, mbs_vec_ket)
    end
    return bracket
end
function ED_bracket_threaded(mbs_vec_bra::MBS64Vector{bits, eltype},
    op2::MBOperator, op1::MBOperator, mbs_vec_ket::MBS64Vector{bits, eltype}
    )::Complex{eltype} where {bits, eltype <: AbstractFloat}
    
    if op1.upper_hermitian || op2.upper_hermitian
        println("one-by-one multiplication")
        return ED_bracket(mbs_vec_bra, op2, op1*mbs_vec_ket)
    end
    
    bracket = Complex{eltype}(0.0)
    for scat_list1 in op1.scats, scat_list2 in op2.scats
        bracket += mul_add_bracket_scatlist_threaded(mbs_vec_bra, scat_list2, scat_list1, mbs_vec_ket)
    end
    return bracket
end
