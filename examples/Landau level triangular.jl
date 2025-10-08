"Parameters and functions for Landau levels with triangular magnetic unit cell"
module LLT

    using ClassicalOrthogonalPolynomials: laguerrel
    using QuadGK

    export ComponentList, LandauInteraction

    const W0::Float64 = 1.0                     # Interaction strength (energy unit)
    const Gl::Float64 = sqrt(2π/sqrt(0.75))     # Magnetic length scale from Brillouin zone area

    # global variable: number of shells includes in the form factor computation
    Nshell::Int64 = 3           

    # a list of component (layer, level, Chern number)
    # layer can only be 1 or 2; Chern number can only be 1 or -1
    struct ComponentList
        
        layer::Vector{Int64}       # inter-/intra-layer interaction
        level::Vector{Int64}       # only level index is not conserved in the interaction
        Chern::Vector{Int64}       # affect sign of the form factor phase
        other::Vector{Int64}       # other indices (eg. spin, valley, sublattice, ...)

        function ComponentList(components::Tuple{Int64, Int64, Int64, Int64}...)
            layer = collect(getindex.(components, 1))
            level = collect(getindex.(components, 2))
            Chern = collect(getindex.(components, 3))
            other = collect(getindex.(components, 4))
            @assert all(in((1, 2)), layer) "layer number can only be 1 or 2"
            @assert all(0 .<= level .<= 10) "level index must >= 0 and <= 10"
            @assert all(in((1, -1)), Chern) "Chern number can only be 1 or -1"
            new(layer, level, Chern, other)
        end
    end


    # Cross product for 2D vectors (returns scalar z-component)
    # Used for computing geometric phases in the magnetic translation algebra
    function ql_cross(q1::Tuple{<: Real, <: Real}, q2::Tuple{<: Real, <: Real})::Real
        return q1[1] * q2[2] - q1[2] * q2[1]
    end



    # Sign function for reciprocal lattice vectors
    # This implements the phase structure of the magnetic translation group
    # The sign depends on the parity of the reciprocal lattice vector indices
    function ita(g1::Int64, g2::Int64)::Int64
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
    function pseudo_potential_decomposition(m::Int64; kwds...)::Float64
        results = quadgk(0.0, Inf) do q²l²
            laguerrel(m, 0, q²l²) * V_Coulomb(sqrt(q²l²); kwds...) * exp(-q²l²)
        end
        return results[1] / W0
    end




    # Landau level form factor
    function level_form_factor(n_f::Int64, n_i::Int64, ql::ComplexF64;
        Chern::Int64 = 1)::ComplexF64

        n_less, n_grater = minmax(n_f, n_i)
        ql_sqrt2 = abs(ql) / sqrt(2.0)
        ql_phase = angle(ql)

        F = exp(-ql_sqrt2^2/2) * laguerrel(n_less, n_grater - n_less, ql_sqrt2^2)
        for n in n_less+1 : n_grater
            F /= sqrt(n)
        end
        F *= (ql_sqrt2 * im)^(n_grater - n_less)
        F *= cis(- ql_phase * (n_f - n_i))

        if Chern == -1
            return F
        elseif Chern == +1
            return conj(F)
        else
            error("Chern number can only be +1 or -1")
        end

    end



    # Define the Fourier transform of the bilayer Coulomb interaction with gate screening
    # when d_l = 0, it is reduced to monolayer interaction
    # return V(q) = W₀ * 1/|ql| * (screening factors)
    function V_Coulomb(ql::Float64; same_layer::Bool=true,
        d_l::Float64, D_l::Float64)::Float64

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

        return V 
    end




    # Define the Haldane pseudopotentials
    # return V(q) = W₀ * Σ_m V_m * L_m(q²l²)
    function V_Haldane(ql::Float64; same_layer::Bool = true,
        V_intra::Vector{Float64}, V_inter::Vector{Float64})::Float64

        if same_layer
            V_m = V_intra
        else
            V_m = V_inter
        end
        
        V = 0.0
        for i in eachindex(V_m)
            V += laguerrel(i-1, 0, ql^2) * V_m[i]
        end

        return W0 * V
    end





    # Callable structure to compute Two-body interaction matrix element
    # This implements the full Coulomb interaction with proper magnetic translation phases
    # The interaction is computed in momentum space with Landau level projection
    # Momentum inputs are Tuple(Float64, Float64) representing (k1, k2) in ratio of Gk
    mutable struct LandauInteraction <: Function

        D_l::Float64                                # Gate distance / magnetic length (D/l)
        d_l::Float64                                # Interlayer distance / magnetic length (d/l)
        V_intra::Vector{Float64}                    # Intralayer Haldane pseudo-potential in unit of W0
        V_inter::Vector{Float64}                    # Interlayer Haldane pseudo-potential in unit of W0
        mix::Real                                   # mix * Haldane + (1-mix) * Coulomb

        const components::ComponentList

        function LandauInteraction( 
            components::Tuple{Int64, Int64, Int64, Int64}...
        )
            new(10.0, 1.0, [0.0; 0.7;], [1.5; 0.0;], 0.5, ComponentList(components...))
        end

    end
    LandauInteraction() = LandauInteraction((1,0,1,0));



    function (V_int::LandauInteraction)(
        kf1::Tuple{<: Real, <: Real}, kf2::Tuple{<: Real, <: Real}, 
        ki2::Tuple{<: Real, <: Real}, ki1::Tuple{<: Real, <: Real}, 
        cf1::Int64 = 1, cf2::Int64 = 1, ci2::Int64 = 1, ci1::Int64 = 1
    )::ComplexF64
        
        @inline read_component(v)  = v[cf1], v[cf2], v[ci2], v[ci1]
        lf1, lf2, li2, li1 = read_component(V_int.components.layer)
        nf1, nf2, ni2, ni1 = read_component(V_int.components.level)
        Cf1, Cf2, Ci2, Ci1 = read_component(V_int.components.Chern)
        sf1, sf2, si2, si1 = read_component(V_int.components.other) # a.k.a., pseudo-spin

        # Layer conservation: interaction must conserve layer indices
        if li1 != lf1 || li2 != lf2
            return 0.0 + 0.0im
        end

        # Chern number conservation: interaction must conserve Chern number
        if Ci1 != Cf1 || Ci2 != Cf2
            return 0.0 + 0.0im
        end

        # pseudo-spin index conservation: interaction must conserve pseudo-spin
        if si1 != sf1 || si2 != sf2
            return 0.0 + 0.0im
        end

        # Calculate momentum transfer (modulo reciprocal lattice)
        q = BZ(ki1 .- kf1)
        G_shift1 = round.(Int64, ki1 .- kf1 .- q, RoundNearest)
        G_shift2 = round.(Int64, kf2 .- ki2 .- q, RoundNearest)

        V_total = ComplexF64(0.0)
        # Sum over reciprocal lattice vectors for convergence
        for g1 in -Nshell:Nshell, g2 in -Nshell:Nshell
            if abs(g1-g2) > Nshell
                continue
            end

            # Construct the full momentum transfer including reciprocal lattice
            qq1 = q[1] + g1
            qq2 = q[2] + g2
            ql::ComplexF64 = Gl * (qq1 + qq2 * cispi(2//3))

            # mixed Coulomb and Haldane interaction
            V  = V_int.mix * V_Haldane(abs(ql); same_layer = (li1==li2), V_intra = V_int.V_intra, V_inter = V_int.V_inter)
            V += (1-V_int.mix) * V_Coulomb(abs(ql); same_layer = (li1==li2), d_l = V_int.d_l, D_l = V_int.D_l) 

            # Landau level form factor
            F1 = level_form_factor(nf1, ni1, -ql; Chern = Ci1)
            F2 = level_form_factor(nf2, ni2,  ql; Chern = Ci2)

            # Calculate phase factors from magnetic translation algebra
            # These phases ensure proper commutation relations and gauge invariance
            phase_angle  = -Ci1 * ql_cross(ki1, kf1)
            phase_angle += -Ci1 * ql_cross(ki1 .+ kf1, (qq1, qq2) )
            phase_angle += -Ci2 * ql_cross(ki2, kf2)
            phase_angle += -Ci2 * ql_cross(ki2 .+ kf2, (-qq1, -qq2) )
            phase = cispi(phase_angle)

            sign = ita(g1+G_shift1[1], g2+G_shift1[2]) * ita(g1+G_shift2[1], g2+G_shift2[2])

            V_total += sign * V * F1 * F2 * phase
        end

        return V_total
    end




    # Define the Landau level infinitesimal form factor
    function Landau_ff_inf(LI::LandauInteraction)::Function
        return (k_f::Tuple{<:Real, <:Real}, k_i::Tuple{<:Real, <:Real}, c::Int64 = 1) -> begin
            dk = k_f .- k_i
            k = 0.5 .* (k_f .+ k_i)
            return LI.components.Chern[c] * π * (k[1]*dk[2] - k[2]*dk[1])
        end
    end


end
