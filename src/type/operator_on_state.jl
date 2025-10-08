using LinearAlgebra

# function search_vector_component(mbs::MBS64{bits}, mbs_vec::MBS64Vector{bits, eltype, idtype}
#     )::Complex{eltype} where {bits, eltype <: AbstractFloat, idtype <: Integer} 

#     idx = get(mbs_vec.space, mbs, zero(idtype))
#     @boundscheck if iszero(idx)
#         throw(DimensionMismatch("The operator scatters the state out of its Hilbert subspace."))
#     end
#     mbs_vec.vec[idx]
# end

"""
    scat * (amp_in::ComplexF64, mbs_in::MBS64) -> (amp_out::ComplexF64, mbs_out::MBS64)
    scat * mbs_in::MBS64 = scat * (1.0, mbs_in)

Applying a scatter operator on a many-body basis. Return the amplitute and the output many-body basis.
The amplitute is zero if no output state.
"""
function *(scat::Scattering, mbs_in::MBS64{bits})::Tuple{ComplexF64, MBS64{bits}} where {bits} 
    
    @boundscheck @assert scat.in[1] <= bits && scat.out[1] <= bits "The scat operator change mbs bits beyond physical limit."

    if isoccupied(mbs_in, scat.in...)
        if scat.in == scat.out
            return scat.Amp, mbs_in
        else
            mbs_mid = empty!(mbs_in, scat.in...; check = false)
            if isempty(mbs_mid, scat.out...)
                mbs_out = occupy!(mbs_mid, scat.out...; check = false)
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
function *(scat::Scattering, tuple_mbs_in::Tuple{ComplexF64, MBS64{bits}})::Tuple{ComplexF64, MBS64{bits}} where {bits}
    
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
"""
function *(mbs_out::MBS64{bits}, scat::Scattering, )::Tuple{ComplexF64, MBS64{bits}} where {bits} 
    
    @boundscheck @assert scat.in[1] <= bits && scat.out[1] <= bits "The scat operator change mbs bits beyond physical limit."

    if isoccupied(mbs_out, scat.out...)
        if scat.in == scat.out
            return scat.Amp, mbs_out
        else
            mbs_mid = empty!(mbs_out, scat.out...; check = false)
            if isempty(mbs_mid, scat.in...)
                mbs_in = occupy!(mbs_mid, scat.in...; check = false)
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
function *(tuple_mbs_out::Tuple{ComplexF64, MBS64{bits}}, scat::Scattering)::Tuple{ComplexF64, MBS64{bits}} where {bits}
    
    @boundscheck @assert scat.in[1] <= bits && scat.out[1] <= bits "The scat operator change mbs bits beyond physical limit."

    amp_out, mbs_out = tuple_mbs_out
    iszero(amp_out) && return tuple_mbs_out

    @inbounds amp_in, mbs_in = mbs_out * scat
    return amp_in * amp_out, mbs_in
end








"""
Multiply a Scattering{N} term on MBS64Vector and add on the collect_result. 

Core function for MBSOperator * MBS64Vector.
It convinces the Julia compiler to use stack instead of heap.

Notice: no checking whether collect_result and mbs_vec are in the same Hilbert space
"""
function mul_add!(collect_result::MBS64Vector{bits, eltype, idtype},
    scat::Scattering{N}, mbs_vec::MBS64Vector{bits, eltype, idtype},
    upper_triangular::Bool) where{N, bits, eltype <: AbstractFloat, idtype <: Integer}

    for (mbs_in, j) in mbs_vec.space
        amp, mbs_out = scat * mbs_in
        # iszero(amp) && continue
        i = get(mbs_vec.space, mbs_out, zero(idtype))
        @boundscheck if iszero(i)
            throw(DimensionMismatch("The operator scatters the state out of its Hilbert subspace."))
        end
        collect_result.vec[i] += amp * mbs_vec.vec[j]
        upper_triangular && i != j && (collect_result.vec[j] += conj(amp) * mbs_vec.vec[i])
    end
end

"""
Use mul_add!() to avoid heap allocation.
"""
function mul!(mbs_vec_result::MBS64Vector{bits, eltype, idtype}, 
    op::MBSOperator{eltype}, mbs_vec::MBS64Vector{bits, eltype, idtype}
    ) where {bits, eltype <: AbstractFloat, idtype <: Integer}
    
    @boundscheck mbs_vec_result.space == mbs_vec.space || throw(DimensionMismatch("mul! shouldn't change MBSVector Hilbert subspace."))

    mbs_vec_result.vec .= 0.0
    for scat in op.scats
        mul_add!(mbs_vec_result, scat, mbs_vec, op.upper_triangular)
    end
end

"""
    op::MBSOperator * mbs_vec::MBS64Vector -> mbs_vec_result::MBS64Vector
    op2::MBSOperator * op1::MBSOperator * mbs_vec::MBS64Vector -> mbs_vec_result::MBS64Vector

Applying one or two operators on a many-body state.
More number of operators is not supported and should be implemented in steps.
"""
function *(op::MBSOperator{eltype}, mbs_vec_in::MBS64Vector{bits, eltype, idtype}
    )::MBS64Vector{bits, eltype, idtype} where {bits, eltype <: AbstractFloat, idtype <: Integer}
    
    mbs_vec_out = similar(mbs_vec_in)
    mul!(mbs_vec_out, op, mbs_vec_in)
    return mbs_vec_out

end
function *(op2::MBSOperator{eltype}, op1::MBSOperator{eltype}, mbs_vec_in::MBS64Vector{bits, eltype, idtype}
    )::MBS64Vector{bits, eltype, idtype} where {bits, eltype <: AbstractFloat, idtype <: Integer}

    mbs_vec_out = similar(mbs_vec_in)
    mbs_vec_mid = similar(mbs_vec_in)
    mul!(mbs_vec_mid, op1, mbs_vec_in)
    mul!(mbs_vec_out, op2, mbs_vec_mid)
    return mbs_vec_out

end






"""
Bracket value of a Scattering{N} term between two MBS64Vectors. 

Core function for ED_bracket(::MBS64Vector, ::MBSOperator, ::MBS64Vector).
It convinces the Julia compiler to use stack instead of heap.

Notice: no checking whether mbs_bra and mbs_ket are in the same Hilbert space
"""
function mul_add_bracket!(mbs_vec_bra::MBS64Vector{bits, eltype, idtype},
    scat::Scattering{N}, mbs_vec_ket::MBS64Vector{bits, eltype, idtype},
    upper_triangular::Bool)::Complex{eltype} where{N, bits, eltype <: AbstractFloat, idtype <: Integer}

    collect_result::Complex{eltype} = 0.0
    for (mbs_in, j) in mbs_vec_ket.space
        amp, mbs_out = scat * mbs_in
        i = get(mbs_vec_ket.space, mbs_out, zero(idtype))
        if iszero(i)
            @boundscheck throw(DimensionMismatch("The operator scatters the state out of its Hilbert subspace."))
        else
            collect_result += conj(mbs_vec_bra.vec[i]) * amp * mbs_vec_ket.vec[j]
            if upper_triangular && i != j
                collect_result += conj(mbs_vec_bra.vec[j]) * conj(amp) * mbs_vec_ket.vec[i]
            end
        end
    end
    return collect_result
end


"""
(need more doc)

compute the bracket <bra::MBS64Vector | op::MBSOperator | ket::MBS64Vector>
"""
function ED_bracket(mbs_vec_bra::MBS64Vector{bits, eltype, idtype}, 
    op::MBSOperator{eltype}, mbs_vec_ket::MBS64Vector{bits, eltype, idtype}
    )::Complex{eltype} where {bits, eltype <: AbstractFloat, idtype <: Integer}

    @boundscheck mbs_vec_bra.space == mbs_vec_ket.space || throw(DimensionMismatch(
        """The "bra" and "ket" are not in the same Hilbert subspace."""
    ))

    bracket = Complex{eltype}(0.0)
    for scat in op.scats
        bracket += mul_add_bracket!(mbs_vec_bra, scat, mbs_vec_ket, op.upper_triangular)
    end
    return bracket
end

function ED_bracket_threaded(mbs_vec_bra::MBS64Vector, op::MBSOperator, mbs_vec_ket::MBS64Vector)
    # A vector to store the partial result from each thread
    thread_brackets = zeros(eltype(mbs_vec_bra.vec), Threads.nthreads())

    Threads.@threads for scat in op.scats
        tid = Threads.threadid()
        thread_brackets[tid] += mul_add_bracket!(mbs_vec_bra, scat, mbs_vec_ket, op.upper_triangular)
    end

    # Final reduction
    return sum(thread_brackets)
end

"""
    state |> scat_or_operator = scat_or_operator(state) = scat_or_operator * state
"""
(scat::Scattering)(state) = scat * state
(op::MBSOperator)(state) = op * state






# Below are tests for multi-threaded





"""
threaded version
use reverse Scattering search.
"""
function mul_add_reverse!(collect_result::MBS64Vector{bits, eltype, idtype},
    mbs_out::MBS64{bits}, i::idtype,
    op::MBSOperator{eltype}, mbs_vec_in::MBS64Vector{bits, eltype, idtype}
    )where{bits, eltype <: AbstractFloat, idtype <: Integer}

    for scat in op.scats
        amp, mbs_in = mbs_out * scat
        if op.upper_triangular && iszero(amp)
            amp, mbs_in = mbs_out * scat'
        end
        j = get(mbs_vec_in.space, mbs_in, zero(idtype))
        @boundscheck if iszero(i)
            throw(DimensionMismatch("The operator scatters the state out of its Hilbert subspace."))
        end
        collect_result.vec[i] += amp * mbs_vec_in.vec[j]

    end
end

"""
Use reverse search.
"""
function mul_reverse!(mbs_vec_result::MBS64Vector{bits, eltype, idtype}, 
    op::MBSOperator{eltype}, mbs_vec::MBS64Vector{bits, eltype, idtype};
    ) where {bits, eltype <: AbstractFloat, idtype <: Integer}
    
    @boundscheck mbs_vec_result.space == mbs_vec.space || throw(DimensionMismatch("mul! shouldn't change MBSVector Hilbert subspace."))

    for (mbs_out, i) in mbs_vec_result.space
        mul_add_reverse!(mbs_vec_result, mbs_out, i, op, mbs_vec)
    end

end

function mul_reverse_threaded!(mbs_vec_result::MBS64Vector{bits, eltype, idtype}, 
    op::MBSOperator{eltype}, mbs_vec::MBS64Vector{bits, eltype, idtype};
    ) where {bits, eltype <: AbstractFloat, idtype <: Integer}
    
    @boundscheck mbs_vec_result.space == mbs_vec.space || throw(DimensionMismatch("mul! shouldn't change MBSVector Hilbert subspace."))

    Threads.@threads :greedy for (mbs_out, i) in mbs_vec_result.space
        mul_add_reverse!(mbs_vec_result, mbs_out, i, op, mbs_vec)
    end

end


function multiplication_threaded(op::MBSOperator{eltype}, mbs_vec_in::MBS64Vector{bits, eltype, idtype};
    multi_thread::Bool = true)::MBS64Vector{bits, eltype, idtype} where {bits, eltype <: AbstractFloat, idtype <: Integer}
    
    mbs_vec_out = similar(mbs_vec_in)
    if multi_thread
        mul_reverse_threaded!(mbs_vec_out, op, mbs_vec_in)
    else
        mul_reverse!(mbs_vec_out, op, mbs_vec_in)
    end
    return mbs_vec_out
end

