

using LinearAlgebra

"""
    scat * (amp::ComplexF64, mbs::MBS64) -> (amp_out::ComplexF64, mbs_out::MBS64)
    scat * mbs::MBS64 = scat * (1.0, mbs)

Applying a scatter operator on a many-body basis. Return the amplitute and the output many-body basis.
The amplitute is zero if no output state.
"""
function *(scat::Scattering, tuple_mbs_in::Tuple{ComplexF64, MBS64{bits}})::Tuple{ComplexF64, MBS64{bits}} where {bits}
    
    @assert scat.in[1] <= bits && scat.out[1] <= bits "The scat operator change mbs bits beyond physical limit."

    amp_in, mbs_in = tuple_mbs_in
    iszero(amp_in) && return 0.0, mbs_in
    if isoccupied(mbs_in, scat.in...)
        if scat.in == scat.out
            mbs_out = mbs_in
            return scat.Amp * amp_in, mbs_out
        else
            mbs_mid = empty!(mbs_in, scat.in...; check = false)
            if isempty(mbs_mid, scat.out...)
                mbs_out = occupy!(mbs_mid, scat.out...; check = false)
                amp = scat.Amp
                if isodd(scat_occ_number(mbs_mid, scat.in) + scat_occ_number(mbs_mid, scat.out))
                    amp = -amp
                end
                return amp * amp_in, mbs_out
            end
        end
    end
    return 0.0, mbs_in
end
function *(scat::Scattering, mbs_in::MBS64{bits})::Tuple{ComplexF64, MBS64{bits}} where {bits} 
    scat * (ComplexF64(1.0), mbs_in) 
end





function mul!(mbs_vec_out::MBS64Vector{bits, eltype, idtype}, 
    op::MBSOperator{eltype}, mbs_vec_in::MBS64Vector{bits, eltype, idtype}
    )::MBS64Vector{bits, eltype, idtype} where {bits, eltype, idtype}
    
    mbs_vec_out.space = mbs_vec_in.space || throw(DimensionMismatch("mul! shouldn't change MBSVector Hilbert subspace."))

    mbs_vec_out.vec .= complex(0.0)
    hermitian = op.upper_triangular
    for (mbs_in, j) in mbs_vec_in.space
        for scat in op.scats
            amp, mbs_out = scat * mbs_in
            i = get(mbs_vec_in.space, mbs_out, 0)
            if i == 0
                throw(BoundsError("The operator does not preserve the assigned Hilbert subspace."))
            else
                mbs_vec_out.vec[i] += amp * mbs_vec_in.vec[j]
                i == j || hermitian && (mbs_vec_out.vec[j] += conj(amp) * mbs_vec_in.vec[i])
            end
        end
    end
end


"""
    op::MBSOperator * mbs_vec::MBS64Vector -> mbs_vec_out::MBS64Vector
    op2::MBSOperator * op1::MBSOperator * mbs_vec::MBS64Vector -> mbs_vec_out::MBS64Vector

Applying one or two operators on a many-body state.
More number of operators is not supported and should be implemented in steps.
"""
function *(op::MBSOperator{eltype}, mbs_vec_in::MBS64Vector{bits, eltype, idtype}
    )::MBS64Vector{bits, eltype, idtype} where {bits, eltype, idtype}
    
    mbs_vec_out = similar(mbs_vec_in)
    mul!(mbs_vec_out, op, mbs_vec_in)
    return mbs_vec_out

end
function *(op2::MBSOperator{eltype}, op1::MBSOperator{eltype}, mbs_vec_in::MBS64Vector{bits, eltype, idtype}
    )::MBS64Vector{bits, eltype, idtype} where {bits, eltype, idtype}

    mbs_vec_out = similar(mbs_vec_in)
    mbs_vec_mid = similar(mbs_vec_in)
    mul!(mbs_vec_mid, op1, mbs_vec_in)
    mul!(mbs_vec_out, op2, mbs_vec_mid)
    return mbs_vec_out

end




"""
    state |> scat_or_operator = scat_or_operator(state) = scat_or_operator * state
"""
(scat::Scattering)(state) = scat * state
(op::MBSOperator)(state) = op * state

