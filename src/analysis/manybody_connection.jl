"""
Many-body Berry connection calculation functions for momentum-conserved exact diagonalization.

This file implements the calculation of Berry connections and geometric phases between
different kshift points in momentum space, which is essential for computing topological
invariants like Chern numbers.
"""

using LinearAlgebra

"""
Modify the overall phase of eigenvectors. This is a preparation for many-body connection Wilson loop integral.
"""
function ED_connection_gaugefixing!(ψ::MBS64Vector, suggest_dims::Int64=1;
    warning_tol = 1e-8)

    @assert 1 <= suggest_dims <= length(ψ) "suggested vector locates outside the Hilbert space."

    x = ψ.vec[suggest_dims]
    if abs(x) < warning_tol
        @warn "Gauge fixing may be unstable. Element at position $suggest_dims has amplitude $(abs(x))."
    end
    ψ.vec .*= cis(-angle(x))
end

function ED_connection_gaugefixing!(ψ::Vector{<:MBS64Vector}, suggest_dims::Vector{Int64} = collect(1:length(ψ));
    warning_tol = 1e-8)

    @assert all(==(ψ[1].space).(getfield.(ψ, :space))) "Vectors are not in the same Hilbert subspace."
    @assert 1 <= minimum(suggest_dims) && maximum(suggest_dims) <= length(ψ[1].vec) "suggested vectors locate outside the Hilbert space."
    @assert length(suggest_dims) == length(ψ) "number of suggested vectors mismatches number of eigenvectors."

    matrix = [ ψ[j].vec[i] for i in suggest_dims, j in eachindex(ψ)]
    x = det(matrix)
    if abs(x) < warning_tol
        @warn "Gauge fixing may be unstable. Determinant of elements at position $suggest_dims has amplitude $(abs(x))."
    end
    ψ[1].vec .*= cis(-angle(x))
end


"""
    ED_connection_step(mbs_list::Vector{MBS64}, ψ_f::Vector{ComplexF64}, ψ_i::Vector{ComplexF64},
                       shift_f::Tuple{Float64, Float64}, shift_i::Tuple{Float64, Float64},
                       para::EDPara; wavefunction_tol::Float64=1e-8, amp_warn_tol::Float64=0.6,
                       amp_warn::Bool=true)

Compute the many-body Berry connection integral between two kshift points in momentum space.

# Arguments
- `mbs_list::Vector{MBS64}`: List of many-body states in this momentum block
- `ψ_f::Vector{ComplexF64}`: Eigenvector at shift_f
- `ψ_i::Vector{ComplexF64}`: Eigenvector at shift_i
- `shift_f::Tuple{Float64, Float64}`: Final momentum shift
- `shift_i::Tuple{Float64, Float64}`: Initial momentum shift
- `para::EDPara`: ED parameters containing momentum states and system details

# Keywords
- `wavefunction_tol::Float64=1e-8`: Minimum amplitude in eigenvectors to consider
- `amp_warn_tol::Float64=0.6`: Minimum inner product amplitude to avoid numerical instability warning
- `amp_warn::Bool=true`: Whether to issue a warning for small inner product amplitude

# Returns
- `Float64`: Step integral of many-body connection = arg( ⟨ψ2|ψ1⟩ )

# Description
The many-body Berry connection step integral is computed as the phase of the inner product
For small twist difference δshift = shift_f - shift_i, inner product ⟨ψ2|ψ1⟩ ≈ exp(i * ∫ A ⋅ δshift)
Step integral = arg(⟨ψ2|ψ1⟩)
"""
function ED_connection_step(ψ_f::MbsVec, ψ_i::MbsVec,
    shift_f::Tuple{Float64, Float64}, shift_i::Tuple{Float64, Float64}, para::EDPara;
    wavefunction_tol::Float64 = 1e-8, print_amp::Bool = false,
    amp_warn_tol::Float64 = 0.6, amp_warn::Bool = true
)::Float64 where {MbsVec <: MBS64Vector}

    @assert ψ_f.space == ψ_i.space "The Hilbert subspaces of eigenvectors must match."

    Gk1, Gk2 = para.Gk
    frac_ki = float(para.k_list) .+ collect(shift_i)
    frac_kf = float(para.k_list) .+ collect(shift_f)
    if Gk1 != 0
        frac_ki[1, :] ./= Gk1
        frac_kf[1, :] ./= Gk1
    end
    if Gk2 != 0
        frac_ki[2, :] ./= Gk2
        frac_kf[2, :] ./= Gk2
    end
    

    inner_prod = 0.0 + 0.0im  # many-body connection step integral
    for x in eachindex(ψ_f.vec)
        if abs(ψ_i.vec[x]) > wavefunction_tol && abs(ψ_f.vec[x]) > wavefunction_tol

            coeffi = ψ_f.vec[x]' * ψ_i.vec[x]

            bc = sum(occ_list(ψ_f.space.list[x]) ) do i
                c, k = fldmod1(i, para.Nk)
                ki = (frac_ki[1, k], frac_ki[2, k])
                kf = (frac_kf[1, k], frac_kf[2, k])
                para.FF_inf_angle(kf, ki, c)
            end

            inner_prod += coeffi * cis(bc)
        end
    end

    if print_amp
        println("Inner product: amp = $(abs(inner_prod)), phase = $(angle(inner_prod))")
    end
    
    if amp_warn && abs(inner_prod) < amp_warn_tol
        @warn "Small inner product amplitude: $(abs(inner_prod))"
    end
    # many-body connection step integral is the phase of the inner product
    mbc = angle(inner_prod)

    if abs(mbc) > π/2
        @warn "Accumulated connection on this segment $(round(mbc, digits = 3)) is greater than π/2. \n Consider use finer mesh."
    end

    return mbc
end
function ED_connection_step(ψ_f::Vector{MbsVec}, ψ_i::Vector{MbsVec},
    shift_f::Tuple{Float64, Float64}, shift_i::Tuple{Float64, Float64}, para::EDPara;
    wavefunction_tol::Float64 = 1e-8, print_amp::Bool = false,
    amp_warn_tol::Float64 = 0.7, amp_warn::Bool = true
)::Float64 where {MbsVec <: MBS64Vector}

    @assert all(==(ψ_f[1].space).(getfield.(ψ_f, :space))) "The Hilbert subspaces of eigenvectors must match."
    @assert all(==(ψ_f[1].space).(getfield.(ψ_i, :space))) "The Hilbert subspaces of eigenvectors must match."
    @assert length(ψ_f) == length(ψ_i) "Number of eigenvectors must match in non-Abelian connection"

    g = length(ψ_f)  # degeneracy

    Gk1, Gk2 = para.Gk
    frac_ki = float(para.k_list) .+ collect(shift_i)
    frac_kf = float(para.k_list) .+ collect(shift_f)
    if Gk1 != 0
        frac_ki[1, :] ./= Gk1
        frac_kf[1, :] ./= Gk1
    end
    if Gk2 != 0
        frac_ki[2, :] ./= Gk2
        frac_kf[2, :] ./= Gk2
    end
    

    inner_prod_matrix = zeros(ComplexF64, g, g)  # many-body connection step integral
    coeffi = similar(inner_prod_matrix)
    for x in eachindex(ψ_f[1].vec)
        psi_i = [ψ_i[j].vec[i] for i in x:x, j in eachindex(ψ_i)]
        psi_f = [ψ_f[j].vec[i] for i in x:x, j in eachindex(ψ_f)]
        if norm(psi_i) > wavefunction_tol && norm(psi_f) > wavefunction_tol

            coeffi .= psi_f' * psi_i

            bc = sum(occ_list(ψ_f[1].space.list[x]) ) do i
                c, k = fldmod1(i, para.Nk)
                ki = (frac_ki[1, k], frac_ki[2, k])
                kf = (frac_kf[1, k], frac_kf[2, k])
                para.FF_inf_angle(kf, ki, c)
            end

            inner_prod_matrix += coeffi * cis(bc)
        end
    end

    inner_prod = det(inner_prod_matrix)

    if print_amp
        println("Inner product: amp = $(abs(inner_prod)), phase = $(angle(inner_prod))")
    end
    
    if amp_warn && abs(inner_prod) < amp_warn_tol
        @warn "Small inner product amplitude: $(abs(inner_prod))"
    end
    # many-body connection step integral is the phase of the inner product
    mbc = angle(inner_prod)

    if abs(mbc) > π/2
        @warn "Accumulated connection on this segment $(round(mbc, digits = 3)) is greater than π/2. \n Consider use finer mesh."
    end

    return mbc
end

