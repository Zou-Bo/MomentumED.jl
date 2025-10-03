"Parameters and functions for Landau levels with triangular magnetic unit cell"
module LLT

    using ClassicalOrthogonalPolynomials: laguerrel
    using QuadGK

    W0 = 1.0                   # Interaction strength (energy unit)

    Gl = sqrt(2π/sqrt(0.75))   # Magnetic length scale from Brillouin zone area
    D_l = 10                   # Gate distance / magnetic length (D/l)
    d_l = 1                    # Interlayer distance / magnetic length (d/l)
    V_intra = [0.0; 1.0;]      # Intralayer Haldane pseudo-potential in unit of W0
    V_inter = [0.5; 0.0;]      # Interlayer Haldane pseudo-potential in unit of W0

    mix = 0.5                  # mix * Haldane + (1-mix) * Coulomb


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


    # Decompose Coulomb interaction into Haldane pseudopotentials
    # function can be V_Coulomb_monolayer, V_Coulomb_bilayer; keywords can be SameLayer = true/false
    function pseudo_potential_decomposition(m::Int64; kwds...)
        results = quadgk(0.0, Inf) do q2l2
            laguerrel(m, 0, q2l2) * V_Coulomb(sqrt(q2l2)/Gl, 0.0; form_factor = false, kwds...) * exp(-q2l2)
        end
        return results[1] / W0
    end






    # Define the Landau level projected Coulomb interaction with gate screening
    # This is the Fourier transform of the bilayer Coulomb interaction
    # when d_l = 0, it is reduced to monolayer interaction
    # V(q) = W₀ * 1/|ql| * (screening factors)
    # return V(q) * Form_Factor(q) * Form_Factor(-q)
    function V_Coulomb(q1::Float64, q2::Float64; 
        d_l::Float64 = d_l, same_layer::Bool=true, n_LL::Int64 = 0, form_factor::Bool = true
    )

        ql = sqrt(q1^2 + q2^2 - q1*q2) * Gl  # |q| in magnetic length units
        DD_l = 2D_l

        V = W0
        if ql == 0.0  # Regularization at q=0 (divergent part)
            if same_layer
                V *= (DD_l + d_l) * (DD_l - d_l) / 2DD_l
            else
                V *= (DD_l - d_l)^2 / 2DD_l
            end
        else
            expd = exp(-ql * d_l)
            expD = exp(-ql * DD_l)
            if same_layer
                V *= (inv(expd) - expD) * (expd - expD) / (1- expD^2) / ql
            else
                V *= (expd - expD)^2 / (1- expD^2) / ql / expd
            end
        end

        if form_factor
            V *= laguerrel(n_LL, 0.0, 0.5*ql^2)^2 * exp(-0.5 * ql^2)
        end

        return V 
    end




    # Define the Haldane pseudopotentials projected on n=0 Landau level
    # V(q) = W₀ * Σ_m V_m * L_m(q²l²)
    # return V(q) * Form_Factor(q) * Form_Factor(-q) 
    function V_Haldane(q1::Float64, q2::Float64;
        same_layer::Bool = true, n_LL::Int64 = 0, form_factor::Bool = true
    )

        ql_square = (q1^2 + q2^2 - q1*q2) * Gl^2  # (|q|l)^2

        if same_layer
            V_m = V_intra
        else
            V_m = V_inter
        end
        
        V = 0.0
        for i in eachindex(V_m)
            V += laguerrel(i-1, 0, ql_square) * V_m[i]
        end

        if form_factor
            V *= laguerrel(n_LL, 0.0, 0.5ql_square)^2 * exp(-0.5 * ql_square)
        end
        return W0 * V
    end





    # Callable structure to compute Two-body interaction matrix element
    # This implements the full Coulomb interaction with proper magnetic translation phases
    # The interaction is computed in momentum space with Landau level projection
    # Momentum inputs are Tuple(Float64, Float64) representing (k1, k2) in ratio of Gk
    struct LandauInteraction <: Function
        V_q::Function
        # layer_number::Int64

        function LandauInteraction(;
            interaction::Symbol, layer_number::Int64, level_index::Int64 = 0)

            @assert interaction ∈ (:Coulomb, :Haldane, :mix) """
                interaction can only be :Coulomb, :Haldane, or :mix.
            """

            if layer_number == 1
                if interaction == :Coulomb
                    return new( (q1, q2, same_layer) ->
                        V_Coulomb(q1, q2; d_l = 0.0, n_LL = level_index)
                    )
                elseif interaction == :Haldane
                    return new( (q1, q2, same_layer) ->
                        V_Haldane(q1, q2; same_layer = true, n_LL = level_index)
                    )
                elseif interaction == :mix
                    return new( (q1, q2, same_layer) ->
                        mix * V_Haldane(q1, q2; same_layer = true, n_LL = level_index) + 
                        (1-mix) * V_Coulomb(q1, q2; d_l = 0.0, n_LL = level_index)
                    )
                end
            elseif layer_number == 2
                if interaction == :Coulomb
                    return new( (q1, q2, same_layer) ->
                        V_Coulomb(q1, q2; same_layer = same_layer, n_LL = level_index)
                    )
                elseif interaction == :Haldane
                    return new( (q1, q2, same_layer) ->
                        V_Haldane(q1, q2; same_layer = same_layer, n_LL = level_index)
                    )
                elseif interaction == :mix
                    return new( (q1, q2, same_layer) ->
                        mix * V_Haldane(q1, q2; same_layer = same_layer, n_LL = level_index) + 
                        (1-mix) * V_Coulomb(q1, q2; same_layer = same_layer, n_LL = level_index)
                    )
                end
            else
                error("layer_number can only be 1 or 2.")
            end
        end
    end



    function (V_int::LandauInteraction)(kf1, kf2, ki2, ki1, cf1=1, cf2=1, ci2=1, ci1=1)::ComplexF64
        
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

            V_total += sign * phase * V_int.V_q(qq1, qq2, ci1==ci2)
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
