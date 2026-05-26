"""
    docstring needed
"""
function OES_NumMomtBlocks(para::EDPara{dim}, orb_list, 
    Ne::Union{Vector{Int64}, Tuple{Vararg{Int64}}}) where {dim}

    @assert length(Ne) == para.Nc_conserve

    # generate maskA and maskB
    bits = para.Nk * para.Nc
    maskA_mbs = MBS64(bits, orb_list)
    maskB_mbs = MBS64_complete(maskA_mbs)
    maskA = occ_list(maskA_mbs)
    maskB = occ_list(maskB_mbs)

    # collect results of blocks with different numbers
    lengthA = zeros(Int64, para.Nc_conserve)
    for i in maskA
        c = fld1(i, para.Nk * para.Nc_mix) 
        lengthA[c] += 1
    end
    number_block_size = min.(lengthA, Ne) .+ 1
    # println("lengthA = $lengthA, number_block_size = $number_block_size")
    subspacesA_list = Array{Vector{HilbertSubspace{bits}}}(undef, number_block_size...)
    subspacesB_list = Array{Vector{HilbertSubspace{bits}}}(undef, number_block_size...)
    momentumA_list = Array{Vector{NTuple{dim, Int64}}}(undef, number_block_size...)
    momentumB_list = Array{Vector{NTuple{dim, Int64}}}(undef, number_block_size...)

    for indexA in CartesianIndices(subspacesA_list)
        NA = Tuple(indexA) .- 1
        NB = Tuple(Ne) .- NA
        # println(NA, " ", NB)
        Asubspace, Ak = ED_momentum_subspaces(para, NA; mask = maskA);
        Bsubspace, Bk = ED_momentum_subspaces(para, NB; mask = maskB);
        subspacesA_list[indexA] = Asubspace
        subspacesB_list[indexA] = Bsubspace
        momentumA_list[indexA] = Ak
        momentumB_list[indexA] = Bk
    end

    return subspacesA_list, subspacesB_list, momentumA_list, momentumB_list

end

"""
    docstring needed.
"""
function OES_NumMomtBlock_coef(vector::MBS64Vector{bits, F}, momentum::NTuple{dim, Int64}, Gk::NTuple{dim, Int64},
    subspaceA::Vector{HilbertSubspace{bits}}, subspaceB::Vector{HilbertSubspace{bits}}, 
    momentumA::Vector{NTuple{dim, Int64}}, momentumB::Vector{NTuple{dim, Int64}}) where {bits, dim, F<:AbstractFloat}

    collect_matrices = Vector{Matrix{Complex{F}}}(undef, length(subspaceA))
    for iA in eachindex(subspaceA)
        # momentum pair
        kA = momentumA[iA]
        kB = momentum .- kA
        Preparation.momentum_residue(kB, Gk)
        iB = findfirst(==(kB), momentumB)
        
        # fill the coefficient matrices
        if isnothing(iB)
            collect_matrices[iA] = Matrix{Complex{F}}(undef, 0, 0)
        else
            collect_matrices[iA] = Matrix{Complex{F}}(undef, length(subspaceA[iA]), length(subspaceB[iB]))
            for (A, mbsA) in enumerate(subspaceA[iA].list)
                for (B, mbsB) in enumerate(subspaceB[iB].list)
                    mbs = mbsA + mbsB
                    i = get(vector.space, mbs)
                    if index_fit(i, vector.space, mbs) # must fit in the same subspace
                        # if !transpose
                            collect_matrices[iA][A, B] = vector.vec[i]
                        # else
                            # collect_matrices[iA][B, A] = vector.vec[i]
                        # end
                    else
                        println("iA = $iA, kA = $kA, kB = $kB, iB = $iB, A = $A, B = $B, mbsA = $mbsA, mbsB = $mbsB, mbs = $mbs, i = $i")
                        error()
                    end
                end
            end
        end

    end
    return collect_matrices
end

"""
    docstring needed.
"""
function OES_NumMomtBlock_spectrum(para::EDPara{dim}, vector::MBS64Vector{bits, F},
    subspacesA_list::Array{Vector{HilbertSubspace{bits}}}, subspacesB_list::Array{Vector{HilbertSubspace{bits}}}, 
    momentumA_list::Array{Vector{NTuple{dim, Int64}}}, momentumB_list::Array{Vector{NTuple{dim, Int64}}};
    eigval_cutoff::Float64 = exp(-20), print_cutoff_position::Bool = false) where {bits, dim, F<:AbstractFloat}

    @assert size(subspacesA_list) == size(subspacesB_list) == size(momentumA_list) == size(momentumB_list)

    total_momentum = MBS_totalmomentum(para, vector.space.list[1])
    Preparation.momentum_residue(total_momentum, para.Gk)

    entanglement_spectrum = similar(momentumA_list, Vector{Vector{Float64}});
    for i in eachindex(entanglement_spectrum)
        coefficient_matrices = OES_NumMomtBlock_coef(vector, total_momentum, para.Gk,
            subspacesA_list[i], subspacesB_list[i], momentumA_list[i], momentumB_list[i]
        )
        density_matrices = similar(coefficient_matrices, Matrix{ComplexF64})
        for j in eachindex(density_matrices)
            s = size(coefficient_matrices[j])
            if s[1] <= s[2]
                density_matrices[j] = coefficient_matrices[j] * coefficient_matrices[j]'
            else
                density_matrices[j] = coefficient_matrices[j]' * coefficient_matrices[j]
            end
        end
        entanglement_spectrum[i] = similar(coefficient_matrices, Vector{Float64})
        for j in eachindex(density_matrices)
            s = size(density_matrices[j], 1)
            if s == (0, 0)
                entanglement_spectrum[i][j] = Vector{Float64}()
                continue
            end
            vals = eigvals(Hermitian(density_matrices[j]))
            nonzeroposition = findfirst(>(eigval_cutoff), vals)
            if isnothing(nonzeroposition)
                nonzeroposition = length(vals) + 1
            end
            if print_cutoff_position && nonzeroposition > 1
                println("Warning: there are $(nonzeroposition-1) eigenvalues smaller than cutoff $(eigval_cutoff).")
            end
            entanglement_spectrum[i][j] = -log.(vals[nonzeroposition:end])
        end
    end
    return entanglement_spectrum
end