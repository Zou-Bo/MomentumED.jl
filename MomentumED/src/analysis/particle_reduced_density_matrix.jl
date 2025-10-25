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

"""
    particle_hole_density(mask2::MBS64{bits}, mask1::MBS64{bits}, mbs_in::MBS64{bits};
        particle::Bool = true) where{bits}

Apply density operator of particles/holes on incident state: operator2 * operator1 * mbs_in.
If particle = true (particle density),
first annihilate particles in mask1, then create particles in mask2;
if particle = false (hole density),
first create particles (annihilate holes) in mask1, then annihilate particles (create holes) in mask2.

Return (true, mbs_out) if succeed. Otherwise, return (false, mbs_in).

Different to Scatter * MBS64, there's no amplitute or signs due to swapping.
"""
function particle_holt_density(mask2::MBS64{bits}, mask1::MBS64{bits}, 
    mbs_in::MBS64{bits}; particle::Bool = true
    )::Tuple{Bool, MBS64{bits}} where{bits}

    if particle
        if isoccupied(mbs_in, mask1)
            mbs_mid = empty!(mbs_in, mask1; check = false)
            if isempty(mbs_mid, mask2)
                mbs_out = occupy!(mbs_mid, mask2; check = false)
                return true, mbs_out
            end
        end
    else
        if isempty(mbs_in, mask1)
            mbs_mid = occupy!(mbs_in, mask1; check = false)
            if isoccupied(mbs_mid, mask2)
                mbs_out = empty!(mbs_mid, mask2; check = false)
                return true, mbs_out
            end
        end
    end
    return false, mbs_in
end



function PES_MomtBlock_rdm(para::EDPara, ψ::MBS64Vector{bits, F}, 
    conserved_component_subspace::HilbertSubspace{bits},
    particle_hole::BitVector = trues(para.Nc_conserve)) where {bits, F<:AbstractFloat}
    
    @assert length(particle_hole) == para.Nc_conserve "lengths of subspaces and particle_hole should == para.Nc_conserve."

    component_mask = UInt64(1) << (para.Nk * para.Nc_hopping) - 1


end