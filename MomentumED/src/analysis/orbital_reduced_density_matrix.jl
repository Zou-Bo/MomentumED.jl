function RDM_NumberBlocks(para::EDPara, orbs, 
    Ne::Union{Vector{Int64}, Tuple{Vararg{Int64}}})

    @assert length(Ne) == para.Nc_conserve

    # generate maskA and maskB
    bits = para.Nk * para.Nc
    maskA_mbs = MBS64(bits, orbs)
    maskB_mbs = MBS64_complete(maskA_mbs)
    maskA = occ_list(maskA_mbs)
    maskB = occ_list(maskB_mbs)

    # collect results of blocks with different numbers
        lengthA = zeros(Int64, para.Nc_conserve)
    for i in maskA
        c = fld1(i, bits) 
        lengthA[c] += 1
    end
    number_block_size = min.(lengthA, Ne) .+ 1
    subspacesA_list = Array{Vector{HilbertSubspace{bits}}}(undef, number_block_size...)
    subspacesB_list = Array{Vector{HilbertSubspace{bits}}}(undef, number_block_size...)
    momentumA_list = Array{Vector{Tuple{Int64, Int64}}}(undef, number_block_size...)
    momentumB_list = Array{Vector{Tuple{Int64, Int64}}}(undef, number_block_size...)

    for indexA in CartesianIndices(subspacesA_list)
        NA = Tuple(indexA) .- 1
        NB = Tuple(Ne) .- NA
        Ass, Ak1, Ak2 = ED_momentum_subspaces(para, NA; mask = maskA);
        Bss, Bk1, Bk2 = ED_momentum_subspaces(para, NB; mask = maskB);
        subspacesA_list[indexA] = Ass
        subspacesB_list[indexA] = Bss
        momentumA_list[indexA] = tuple.(Ak1, Ak2)
        momentumB_list[indexA] = tuple.(Bk1, Bk2)
    end

    return subspacesA_list, subspacesB_list, momentumA_list, momentumB_list

end

function RDM_MomentumCoefficients(para::EDPara, vector::MBS64Vector{bits}, momentum::Tuple{<:Real, <:Real},
    subspaceA::Vector{HilbertSubspace{bits}}, subspaceB::Vector{HilbertSubspace{bits}}, 
    momentumA::Vector{Tuple{Int64, Int64}}, momentumB::Vector{Tuple{Int64, Int64}};
    transpose::Bool = false) where {bits}

    if length(subspaceA) > length(subspaceB)
        return RDM_MomentumCoefficients(para, vector, momentum,
            subspaceB, subspaceA, momentumB, momentumA; transpose = true)
    else
        collect_matrices = Vector{Matrix{ComplexF64}}(undef, length(subspaceA))
        Gk = para.Gk
        for iA in eachindex(subspaceA)
            # momentum pair
            kA = momentumA[iA]
            kB = momentum .- kA
            if Gk[1] != 0
                kB = (mod(kB[1], Gk[1]), kB[2])
            end
            if Gk[2] != 0
                kB = (kB[1], mod(kB[2], Gk[2]))
            end
            iB = findfirst(==(kB), momentumB)
            
            # fill the coefficient matrices
            if isnothing(iB)
                collect_matrices[iA] = Matrix{ComplexF64}(undef, 0, 0)
            else
                if !transpose
                    collect_matrices[iA] = Matrix{ComplexF64}(undef, length(subspaceA[iA]), length(subspaceB[iB]))
                else
                    collect_matrices[iA] = Matrix{ComplexF64}(undef, length(subspaceB[iB]), length(subspaceA[iA]))
                end
                for (A, mbsA) in enumerate(subspaceA[iA].list)
                    for (B, mbsB) in enumerate(subspaceB[iB].list)
                        mbs = mbsA + mbsB
                        i = get(vector.space, mbs)
                        if !transpose
                            collect_matrices[iA][A, B] = vector.vec[i]
                        else
                            collect_matrices[iA][B, A] = vector.vec[i]
                        end
                    end
                end
            end

        end
        return collect_matrices
    end
end
