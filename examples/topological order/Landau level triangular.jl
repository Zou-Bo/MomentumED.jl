"Parameters and functions for Landau levels with triangular magnetic unit cell"
module LLT

    Gl = sqrt(2π/sqrt(0.75))   # Magnetic length scale from Brillouin zone area
    D_l = 10                   # Gate distance / magnetic length (D/l)
    d_l = 1                    # Interlayer distance / magnetic length (d/l)
    W0 = 1.0                   # Interaction strength (energy unit)


    # Cross product for 2D vectors (returns scalar z-component)
    # Used for computing geometric phases in the magnetic translation algebra
    function ql_cross(q1, q2)
        return q1[1] * q2[2] - q1[2] * q2[1]
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



    # Shift a momentum to the hexagonal Brillouin zone (120 degree from G1 to G2)
    function BZ(k::Tuple{<: Real, <: Real})::Tuple{<: Real, <: Real}
        k1 = rem(k[1], 2, RoundNearest)
        k2 = rem(k[2], 2, RoundNearest)

        # 1. -1 <= 2k1 - k2 < 1
        if 2k1 - k2 < -1
            k1 += 1
        elseif 2k1 - k2 >= 1
            k1 -= 1
        end

        # 2. -1 < k1 - 2k2 <= 1
        if k1 - 2k2 <= -1
            k2 -= 1
        elseif k1 - 2k2 > 1
            k2 += 1
        end

        # redo step 1
        if 2k1 - k2 < -1
            k1 += 1
        elseif 2k1 - k2 >= 1
            k1 -= 1
        end

        # 3. -1 <= k1 + k2 < 1
        if k1 + k2 < -1
            k1 += 1
            k2 += 1
        elseif k1 + k2 >= 1
            k1 -= 1
            k2 -= 1
        end
        return k1, k2
    end



    # Define the form factor for Coulomb interaction in Landau level
    # This is the Fourier transform of the projected Coulomb interaction
    # V(q) = W₀ * 1/|ql| * tanh(|qD|) * exp(-0.5 * q²l²)
    # The exp(-0.5 * q²l²) factor comes from Landau level projection
    function VFF_monolayer(q1::Float64, q2::Float64)
        ql = sqrt(q1^2 + q2^2 - q1*q2) * Gl  # |q| in magnetic length units
        if ql == 0.0
            return W0 * D_l  # Regularization at q=0 (divergent part)
        end
        return W0 / ql * tanh(ql * D_l) * exp(-0.5 * ql^2)
    end




    # Define the form factor for Coulomb interaction in Landau level
    # This is the Fourier transform of the projected Coulomb interaction
    # V(q) = W₀ * 1/|ql| * tanh(|qD|) * exp(-0.5 * q²l²)
    # The exp(-0.5 * q²l²) factor comes from Landau level projection
    function VFF_bilayer(q1::Float64, q2::Float64; SameLayer::Bool)
        ql = sqrt(q1^2 + q2^2 - q1*q2) * Gl  # |q| in magnetic length units

        V = W0 * exp(-0.5 * ql^2)
        if ql == 0.0  # Regularization at q=0 (divergent part)
            if SameLayer
                V *= (D_l + d_l) * (D_l - d_l) / 2D_l
            else
                V *= (D_l - d_l)^2 / 2D_l
            end
        else
            expd = exp(-ql * d_l)
            expD = exp(-ql * D_l)
            if SameLayer
                V *= (inv(expd) - expD) * (expd - expD) / (1- expD^2) / ql
            else
                V *= (expd - expD)^2 / (1- expD^2) / ql / expd
            end
        end
        return V
    end



    # Two-body interaction matrix element
    # This implements the full Coulomb interaction with proper magnetic translation phases
    # The interaction is computed in momentum space with Landau level projection
    # Momentum inputs are Tuple(Float64, Float64) representing (k1, k2) in ratio of Gk
    function V_int_monolayer(kf1, kf2, ki2, ki1, cf1=1, cf2=1, ci2=1, ci1=1)::ComplexF64
        
        # Calculate momentum transfer (modulo reciprocal lattice)
        q = BZ(ki1 .- kf1)
        G_shift1 = round.(Int64, ki1 .- kf1 .- q, RoundNearest)
        G_shift2 = round.(Int64, kf2 .- ki2 .- q, RoundNearest)

        V_total = ComplexF64(0.0)
        # Sum over reciprocal lattice vectors for convergence
        # Nshell = 3 provides good convergence for this system
        Nshell = 3
        for g1 in -Nshell:Nshell, g2 in -Nshell:Nshell
            if abs(g1-g2) > Nshell
                continue
            end

            # Construct the full momentum transfer including reciprocal lattice
            qq1 = q[1] + g1
            qq2 = q[2] + g2

            # Calculate phase factors from magnetic translation algebra
            # These phases ensure proper commutation relations and gauge invariance
            phase_angle = ql_cross(ki1, kf1)
            phase_angle += ql_cross(ki1 .+ kf1, (qq1, qq2) )
            phase_angle += ql_cross(ki2, kf2)
            phase_angle += ql_cross(ki2 .+ kf2, (-qq1, -qq2) )

            phase = cispi(phase_angle)
            sign = ita(g1+G_shift1[1], g2+G_shift1[2]) * ita(g1+G_shift2[1], g2+G_shift2[2])

            V_total += sign * phase * VFF_monolayer(qq1, qq2)
        end

        return V_total
    end


    # Two-body interaction matrix element
    # This implements the full Coulomb interaction with proper magnetic translation phases
    # The interaction is computed in momentum space with Landau level projection
    # Momentum inputs are Tuple(Float64, Float64) representing (k1, k2) in ratio of Gk
    function V_int_bilayer(kf1, kf2, ki2, ki1, cf1, cf2, ci2, ci1)::ComplexF64
        
        # Layer conservation: interaction must conserve layer indices
        if ci1 != cf1 || ci2 != cf2
            return 0.0 + 0.0im
        end

        # Calculate momentum transfer (modulo reciprocal lattice)
        q = BZ(ki1 .- kf1)
        G_shift1 = round.(Int64, ki1 .- kf1 .- q, RoundNearest)
        G_shift2 = round.(Int64, kf2 .- ki2 .- q, RoundNearest)

        V_total = ComplexF64(0.0)
        # Sum over reciprocal lattice vectors for convergence
        # Nshell = 3 provides good convergence for this system
        Nshell = 3
        for g1 in -Nshell:Nshell, g2 in -Nshell:Nshell
            if abs(g1-g2) > Nshell
                continue
            end

            # Construct the full momentum transfer including reciprocal lattice
            qq1 = q[1] + g1
            qq2 = q[2] + g2

            # Calculate phase factors from magnetic translation algebra
            # These phases ensure proper commutation relations and gauge invariance
            phase_angle = ql_cross(ki1, kf1)
            phase_angle += ql_cross(ki1 .+ kf1, (qq1, qq2) )
            phase_angle += ql_cross(ki2, kf2)
            phase_angle += ql_cross(ki2 .+ kf2, (-qq1, -qq2) )

            phase = cispi(phase_angle)
            sign = ita(g1+G_shift1[1], g2+G_shift1[2]) * ita(g1+G_shift2[1], g2+G_shift2[2])

            V_total += sign * phase * VFF_bilayer(qq1, qq2; SameLayer = (ci1==ci2))
        end

        return V_total
    end



    # Define the Landau level infinitesimal form factor
    function Landau_ff_inf(k_f, k_i, c=1)
        dk = k_f .- k_i
        k = 0.5 .* (k_f .+ k_i)
        return -π * (k[1]*dk[2] - k[2]*dk[1])
    end

end
