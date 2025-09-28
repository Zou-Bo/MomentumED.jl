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
function ED_connection_gaugefixing!(ψ::Vector{ComplexF64}, suggest_dims::Int64=1;
    warning_tol = 1e-8)
    @assert 1 <= suggest_dims <= length(ψ)
    x = ψ[suggest_dims]
    if abs(x) < warning_tol
        @warn "Gauge fixing may be unstable. Element at position $suggest_dims has amplitute $(abs(x))."
    end
    ψ .*= cis(-angle(x))
end

function ED_connection_gaugefixing!(ψ::Matrix{ComplexF64}, suggest_dims::Vector{Int64} = collect(1:size(ψ, 2));
    warning_tol = 1e-8)
    @assert 1 <= minimum(suggest_dims) && maximum(suggest_dims) <= size(ψ, 1)
    @assert length(suggest_dims) == size(ψ, 2)
    x = det(ψ[suggest_dims, :])
    if abs(x) < warning_tol
        @warn "Gauge fixing may be unstable. Determinant of elements at position $suggest_dims has amplitute $(abs(x))."
    end
    ψ[:, 1] .*= cis(-angle(x))
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
function ED_connection_step(mbs_list::Vector{<: MBS64}, 
    ψ_f::Vector{ComplexF64}, ψ_i::Vector{ComplexF64},
    shift_f::Tuple{Float64, Float64}, shift_i::Tuple{Float64, Float64},
    para::EDPara;
    wavefunction_tol::Float64 = 1e-8,
    print_amp::Bool = false,
    amp_warn_tol::Float64 = 0.6, amp_warn::Bool = true
)::Float64

    @assert length(ψ_f) == length(ψ_i) == length(mbs_list) "Length of basis and eigenvectors must match."

    Nk = para.Nk
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
    for x in eachindex(mbs_list)
        if abs(ψ_i[x]) > wavefunction_tol && abs(ψ_f[x]) > wavefunction_tol
            
            coeffi = ψ_f[x]' * ψ_i[x]
            
            bc = sum(occ_list(mbs_list[x]) ) do i
                c, k = fldmod1(i, Nk)
                ki = Tuple(frac_ki[:, k])
                kf = Tuple(frac_kf[:, k])
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
        @warn "Accumulated connection on this segment $(round(mbs, digits = 3)) is greater than π/2. \n Consider use finer mesh."
    end

    return mbc
end
function ED_connection_step(mbs_list::Vector{<: MBS64}, 
    ψ_f::Matrix{ComplexF64}, ψ_i::Matrix{ComplexF64},
    shift_f::Tuple{Float64, Float64}, shift_i::Tuple{Float64, Float64},
    para::EDPara;
    wavefunction_tol::Float64 = 1e-8,
    print_amp::Bool = false,
    amp_warn_tol::Float64 = 0.7, amp_warn::Bool = true
)::Float64

    @assert size(ψ_f, 1) == size(ψ_i, 1) == length(mbs_list) "Length of basis and eigenvectors must match."
    @assert size(ψ_f, 2) == size(ψ_i, 2) "Number of states must match in non-Abelian connection"

    g = size(ψ_f, 2)  # degeneracy
    Nk = para.Nk
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
    for x in eachindex(mbs_list)
        if norm(ψ_i[x, :]) > wavefunction_tol && norm(ψ_f[x, :]) > wavefunction_tol

            coeffi .= ψ_f[x:x, :]' * ψ_i[x:x, :]

            bc = sum(occ_list(mbs_list[x]) ) do i
                c, k = fldmod1(i, Nk)
                ki = Tuple(frac_ki[:, k])
                kf = Tuple(frac_kf[:, k])
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
        @warn "Accumulated connection on this segment $(round(mbs, digits = 3)) is greater than π/2. \n Consider use finer mesh."
    end

    return mbc
end

