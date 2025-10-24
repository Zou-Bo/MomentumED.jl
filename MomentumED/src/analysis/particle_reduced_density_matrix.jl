function PES_1rdm(ψ::MBS64Vector{bits, F}) where{bits, F <: AbstractFloat}
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

function PES_MomtBlocks(para, Ne_in_A, Ne)
    subspacesA = HilbertSubspace{bits}[]
    subspacesB = HilbertSubspace{bits}[]
    momentumA = Tuple{Int64, Int64}[]
    momentumB = Tuple{Int64, Int64}[]

    return subspacesA, subspacesB, momentumA, momentumB
end

function PES_MomtBlock_coefficients()
    @info "Coefficient matrix is not encouraged. Try generate the density matrix directly."

end