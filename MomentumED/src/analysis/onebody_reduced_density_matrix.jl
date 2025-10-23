function RDM_OneBody(ψ::MBS64Vector{bits, F}) where{bits, F <: AbstractFloat}
    rdm = zeros(Complex{F}, bits, bits)
    for i in 1:bits, j in 1:bits
        if i <= j
            for x_in in eachindex(ψ.vec)
                mbs_in = ψ.space.list[x_in]
                if isoccupied(mbs_in, (j,))
                    mbs_mid = empty!(mbs_in, (j,))
                    if isempty(mbs_mid, (i,))
                        mbs_out = occupy!(mbs_mid, (i,))
                        x_out = get(ψ.space, mbs_out)
                        if x_out != 0
                            rdm[i,j] += conj(ψ.vec[x_out]) * ψ.vec[x_in]
                        end
                    end
                end
            end
        end
    end
    for i in 1:bits, j in 1:bits
        if i > j
            rdm[i,j] = conj(rdm[j,i])
        end
    end
    return rdm
end