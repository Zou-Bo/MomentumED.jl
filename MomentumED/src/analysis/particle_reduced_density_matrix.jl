"""
Simple way of generating one-body reduced density.
"""
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
                            if iseven(scat_occ_number(mbs_mid, (i,)) + scat_occ_number(mbs_mid, (j,)))
                                rdm[i,j] += conj(ψ.vec[x_out]) * ψ.vec[x_in]
                            else
                                rdm[i,j] -= conj(ψ.vec[x_out]) * ψ.vec[x_in]
                            end
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

"""
"""
function PES_MomtBlock_rdm(para::EDPara, ψ::MBS64Vector{bits, F}, 
    part_subspace::HilbertSubspace{bits},
    ph_transform::BitVector = falses(para.Nc_conserve)) where {bits, F<:AbstractFloat}
    
    @assert length(ph_transform) == para.Nc_conserve "lengths of subspaces and particle_hole should == para.Nc_conserve."
    @assert bits == para.Nk * para.Nc_hopping * para.Nc_conserve

    bit_c = para.Nk * para.Nc_hopping
    component_width = UInt64(1) << bit_c - 1
    ph_trans_mask = UInt64(0)
    for c in 1:para.Nc_conserve;
        if ph_transform[c] 
            ph_trans_mask |= component_width << ((c-1)*bit_c)
        end
    end
    # println("ph_trans_mask= $ph_trans_mask")

    rdm = zeros(Complex{F}, length(part_subspace), length(part_subspace))
    for (i, anni) in enumerate(part_subspace.list)
        for (j, crea) in enumerate(part_subspace.list)
            if i <= j
                # println("anni: $anni \n occ_list: ", occ_list(anni))
                # println("anni: $crea \n occ_list: ", occ_list(crea))
                # try to apply creation/annihilation; if pass, add to rdm
                for (x_in, mbs_in) in enumerate(ψ.space.list)
                    mbs_in_ph = flip!(mbs_in, ph_trans_mask)
                    # println("$mbs_in is flipped into $mbs_in_ph")
                    if isoccupied(mbs_in_ph, anni.n)
                        mbs_mid_ph = empty!(mbs_in_ph, anni.n; check = false)
                        if isempty(mbs_mid_ph, crea.n)
                            mbs_out_ph = occupy!(mbs_mid_ph, crea.n; check = false)
                            mbs_out = flip!(mbs_out_ph, ph_trans_mask)
                            x_out = get(ψ.space, mbs_out)
                            if !iszero(x_out)
                                # determine the sign using the states before ph-transform
                                if iseven(scat_occ_number(mbs_in, occ_list(anni)) + scat_occ_number(mbs_out, occ_list(crea)))
                                    # println("even")
                                    rdm[i,j] += conj(ψ.vec[x_out]) * ψ.vec[x_in]
                                else
                                    # println("odd")
                                    rdm[i,j] -= conj(ψ.vec[x_out]) * ψ.vec[x_in]
                                end
                            else
                                # println("find $mbs_out failed")
                            end
                        else
                            # println("occupy $mbs_mid_ph with $(crea.n) failed.")
                        end
                    else
                        # println("empty $mbs_in_ph with $(anni.n) failed")
                    end
                end
            end
        end
    end
    return Hermitian(rdm, :U)
end