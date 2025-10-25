# Define k-mesh for triangular lattice
# Using 3×5 mesh (Nk=15) for accurate Laughlin state calculation
# Note: Nk must be multiple of m=3 for 1/3 filling factor
k_list = [0 1 2 0 1 2 0 1 2 0 1 2;
          0 0 0 1 1 1 2 2 2 3 3 3]
# System parameters
Nk = 12         # Total number of k-points
Gk = (3, 4)      # Grid dimensions (G1_direction, G2_direction)
# Number of electrons for 1/3 filling
Ne = 4      # N electrons for this system

# Import the momentum-conserved exact diagonalization package
using MomentumED

# Physical parameters for the FQH system
Gl = sqrt(2π/sqrt(0.75))  # Magnetic length scale from Brillouin zone area
D_l = 5.0                  # Screening length / magnetic length (D/l = 5)
W0 = 1.0                   # Interaction strength (energy unit)
G12_angle = 2π/3          # Angle between reciprocal lattice vectors (triangular lattice)

# Define the form factor for Coulomb interaction in Landau level
# This is the Fourier transform of the projected Coulomb interaction
# V(q) = W₀ * 1/|ql| * tanh(|qD|) * exp(-0.5 * q²l²)
# The exp(-0.5 * q²l²) factor comes from Landau level projection
function VFF(q1::Float64, q2::Float64)
    ql = sqrt(q1^2 + q2^2 + 2cos(G12_angle) * q1*q2) * Gl  # |q| in magnetic length units
    if ql == 0.0
        return W0 * D_l  # Regularization at q=0 (divergent part)
    end
    return W0 / ql * tanh(ql * D_l) * exp(-0.5 * ql^2)
end

# Sign function for reciprocal lattice vectors
# This implements the phase structure of the magnetic translation group
# The sign depends on the parity of the reciprocal lattice vector indices
function ita(g1::Int64, g2::Int64)
    if iseven(g1) && iseven(g2)
        return 1
    else
        return -1
    end
end

# Cross product for 2D vectors (returns scalar z-component)
# Used for computing geometric phases in the magnetic translation algebra
function ql_cross(q1_1, q1_2, q2_1, q2_2)
    return q1_1 * q2_2 - q1_2 * q2_1
end

# Two-body interaction matrix element
# This implements the full Coulomb interaction with proper magnetic translation phases
# The interaction is computed in momentum space with Landau level projection
# Momentum inputs are Tuple{Float64, Float64} representing (k1, k2) in ratio of Gk
function V_int(kf1, kf2, ki2, ki1, cf1=1, cf2=1, ci2=1, ci1=1)::ComplexF64
    
    # Calculate momentum transfer (modulo reciprocal lattice)
    q = rem.(ki1 .- kf1, 1, RoundNearest)
    G_shift1 = round.(Int64, ki1 .- kf1 .- q, RoundNearest)
    G_shift2 = round.(Int64, kf2 .- ki2 .- q, RoundNearest)

    V_total = ComplexF64(0.0)
    # Sum over reciprocal lattice vectors for convergence
    # Nshell = 2 provides good convergence for this system
    Nshell = 2
    for g1 in -Nshell:Nshell, g2 in -Nshell:Nshell
        if abs(g1-g2) > Nshell
            continue
        end

        # Construct the full momentum transfer including reciprocal lattice
        qq1 = q[1] + g1
        qq2 = q[2] + g2

        # Calculate phase factors from magnetic translation algebra
        # These phases ensure proper commutation relations and gauge invariance
        phase_angle = 0.5ql_cross(ki1[1], ki1[2], kf1[1], kf1[2])
        phase_angle += 0.5ql_cross(ki1[1]+kf1[1], ki1[2]+kf1[2], qq1, qq2)
        phase_angle += 0.5ql_cross(ki2[1], ki2[2], kf2[1], kf2[2])
        phase_angle += 0.5ql_cross(ki2[1]+kf2[1], ki2[2]+kf2[2], -qq1, -qq2)

        phase = cispi(2.0phase_angle)
        sign = ita(g1+G_shift1[1], g2+G_shift1[2]) * ita(g1+G_shift2[1], g2+G_shift2[2])

        V_total += sign * phase * VFF(Float64(qq1), Float64(qq2))
    end

    return V_total
end

# Create parameter structure for the exact diagonalization
# This contains all the system information needed for the calculation
para = EDPara(k_list=k_list, Gk=Gk, V_int=V_int);
para.momentum_coordinate

# Create momentum blocks (Hilbert subspace)
subspaces, ss_k1, ss_k2 = ED_momentum_subspaces(para, (Ne,));
display(length.(subspaces))

# one-body terms are all zero in flat Landau level
scat = ED_sortedScatterList_twobody(para);


Neigen = 10  # Number of eigenvalues to compute per subspace
energies = Vector{Vector{Float64}}(undef, length(subspaces));
vectors = Vector{Vector{<:MBS64Vector}}(undef, length(subspaces));
for i in eachindex(subspaces)
    println("Processing subspace #$i with size $(length(subspaces[i])), momentum $(ss_k1[i]), $(ss_k2[i])")
    energies[i], vectors[i] = EDsolve(subspaces[i], scat;
        N=Neigen, showtime=true, ishermitian=true
    )
end



# Define the Landau level infinitesimal form factor
function Landau_ff_inf(k_f, k_i, c=1)
    dk = k_f .- k_i
    k = 0.5 .* (k_f .+ k_i)
    return -π * (k[1]*dk[2] - k[2]*dk[1])
end
para.FF_inf_angle = Landau_ff_inf; # Update the form factor in the parameter



