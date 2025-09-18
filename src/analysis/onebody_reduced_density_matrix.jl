function ED_onebody_rdm(mbs_list::Vector{MBS64{bits}}, ψ::Vector{ComplexF64}) where{bits}
    rdm = zeros(ComplexF64, bits, bits)
    for i in 1:bits, j in 1:bits
        if i <= j
            for x_in in eachindex(ψ)
                mbs_in = mbs_list[x_in]
                if isoccupied(mbs_in, j)
                    mbs_mid = empty!(mbs_in, j)
                    if isempty(mbs_mid, i)
                        mbs_out = occupy!(mbs_mid, i)
                        x_out = my_searchsortedfirst(mbs_list, mbs_out)
                        if x_out != 0
                            rdm[i,j] += conj(ψ[x_out]) * ψ[x_in]
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