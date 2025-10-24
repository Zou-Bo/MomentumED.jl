"Parameters and functions for Landau levels with torus geometry and periodic boundary condition"
module LLT

    using ClassicalOrthogonalPolynomials: laguerrel
    using QuadGK

    export ReciprocalLattice, ComponentList, LandauInteraction
    export Landau_ff_inf, density_operator

    # Global variables, usually no need to change
    const W0::Float64 = 1.0                     # Interaction strength (energy unit)
    const l::Float64 = 1.0                      # Magnetic Length (length/momentum unit)
    shell_cutoff::Float64 = 20.0                 # number of shells (|Gl| <= shell_cutoff) included in the interaction 


    struct ReciprocalLattice
        G1::ComplexF64
        G2::ComplexF64
        exact_G2²_G1²::Real              # exact abs2(G2) / abs2(G1)
        exact_G1dotG2_G1²::Real          # exact G1 ⋅ G2 / abs2(G1)

        function ReciprocalLattice(G2_ratio_G1::Real, cos_angle::Real; G1phase_in_pi::Real = 0.0)
            sin_square = 1.0 - cos_angle^2
            @assert sin_square > 0.0 "abs(cosθ) should < 1.0"
            sin_angle = sqrt(sin_square)
            
            # G1 * G2 * sin(angle) = G1^2 * abs(ratio) * sin(angle) = 2π/l^2
            G1 = sqrt( 2π/l^2 / abs(G2_ratio_G1) / sin_angle ) * cispi(G1phase_in_pi)
            G2 = G1 * G2_ratio_G1 * (cos_angle + sin_angle*im)
            new(G1, G2, G2_ratio_G1^2, G2_ratio_G1*cos_angle)
        end
    end
    function ReciprocalLattice(s::Symbol)
        if s == :square
            return ReciprocalLattice(1, 0)
        elseif s == :triangular
            return ReciprocalLattice(1, -1//2)
        else
            throw(AssertionError("Symbol construction is for :square or :triangular, use ratio-angle construction instead."))
        end
    end


    # a list of component (layer, level, Chern number, other/pseudospin)
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

    

    # Shift a momentum to the Brillouin zone
    function BZ(k::Tuple{<: Real, <: Real}, lattice::ReciprocalLattice;
        # output = false
        )::Tuple{<: Real, <: Real}
        k1 = rem(k[1], 1, RoundDown)
        k2 = rem(k[2], 1, RoundDown)

        kdotG1 = k1 + k2 * lattice.exact_G1dotG2_G1²
        kdotG2 = k1 * lattice.exact_G1dotG2_G1² + k2 * lattice.exact_G2²_G1²
        distance00 = k1^2 + 2k1*k2 * lattice.exact_G1dotG2_G1² + k2^2 * lattice.exact_G2²_G1²
        distance01 = distance00 - 2kdotG2 + lattice.exact_G2²_G1²
        distance10 = distance00 - 2kdotG1 + 1
        distance11 = distance00 - 2kdotG1 - 2kdotG2 + 1 + 2lattice.exact_G1dotG2_G1² + lattice.exact_G2²_G1²

        # if output
        #     println((k1, k2), [distance11, distance10, distance01, distance00])
        # end

        if distance11 <= distance10 && distance11 <= distance01 && distance11 <= distance00
            return (k1-1, k2-1)
        elseif distance10 <= distance01 && distance10 <= distance00
            return (k1-1, k2)
        elseif distance01 <= distance00
            return (k1, k2-1)
        else
            return (k1, k2)
        end
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

        @assert abs(Chern) == 1 "Chern number can only be +1 or -1"

        n_less, n_grater = minmax(n_f, n_i)
        ql_sqrt2 = abs(ql) / sqrt(2.0)
        ql_phase = angle(ql)

        F = exp(-ql_sqrt2^2/2) * laguerrel(n_less, n_grater - n_less, ql_sqrt2^2)
        for n in n_less+1 : n_grater
            F /= sqrt(n)
        end
        F *= (ql_sqrt2 * im)^(n_grater - n_less)
        F *= cis(Chern * ql_phase * (n_f - n_i))

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
    # Momentum inputs are Tuple{Float64, Float64} representing (k1, k2) in ratio of Gk
    mutable struct LandauInteraction <: Function

        const lattice::ReciprocalLattice                  # {G1, G2}

        D_l::Float64                                # Gate distance / magnetic length (D/l)
        d_l::Float64                                # Interlayer distance / magnetic length (d/l)
        V_intra::Vector{Float64}                    # Intralayer Haldane pseudo-potential in unit of W0
        V_inter::Vector{Float64}                    # Interlayer Haldane pseudo-potential in unit of W0
        mix::Real                                   # mix * Haldane + (1-mix) * Coulomb

        const components::ComponentList

        function LandauInteraction(lattice::ReciprocalLattice,
            components::Tuple{Int64, Int64, Int64, Int64}...
        )
            new(lattice, 10.0, 0.0, [0.0; 0.7;], [1.5; 0.0;], 0.0, ComponentList(components...))
        end

    end



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
        q = BZ(ki1 .- kf1, V_int.lattice)
        G_shift1 = round.(Int64, ki1 .- kf1 .- q, RoundNearest)
        G_shift2 = round.(Int64, kf2 .- ki2 .- q, RoundNearest)

        V_total = ComplexF64(0.0)
        # Sum over reciprocal lattice vectors for convergence
        Nshell = round(Int64, shell_cutoff, RoundUp)
        # println("Nshell=$Nshell, q=$q, G_shift1=$G_shift1, G_shift2=$G_shift2")
        for g1 in -Nshell:Nshell, g2 in -Nshell:Nshell
            if abs2(g1 * V_int.lattice.G1 + g2 * V_int.lattice.G2) > shell_cutoff^2
                continue
            end

            # Construct the full momentum transfer including reciprocal lattice
            qq1 = q[1] + g1
            qq2 = q[2] + g2
            ql::ComplexF64 = qq1 * V_int.lattice.G1 + qq2 * V_int.lattice.G2
            
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

            # println("g1,g2=$((g1,g2)), ql=$(abs(ql)), VFF=$(V * F1 * F2), sign=$sign, phase=$phase.")
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



    using MomentumED

    function momentum_index_search(k1::Int64, k2::Int64; para::EDPara)::Int64
        Gk1, Gk2 = para.Gk
        for i in axes(para.k_list, 2)
            Gk1 == 0 && para.k_list[1,i] != k1 && continue
            Gk1 != 0 && mod(para.k_list[1,i], Gk1) != mod(k1, Gk1) && continue
            Gk2 == 0 && para.k_list[2,i] != k2 && continue
            Gk2 != 0 && mod(para.k_list[2,i], Gk2) != mod(k2, Gk2) && continue
            return i
        end
        return 0
    end

    # ρ_{cf, ci}(q) = Σ_{ki,kf} < kf| e^{iqr} | ki > * c†_{kf, cf} c_{ki, ci}
    function density_operator(q1::Int64, q2::Int64, cf::Int64=1, ci::Int64=1;
        para::EDPara, form_factor::Bool)::MBOperator
        cp = para.V_int.components
        Gk = para.Gk
        k_list = para.k_list

        @assert para.V_int isa LandauInteraction "the ED parameter is not generated by LandauInteraction in LLT module."
        if cp.Chern[cf] != cp.Chern[ci]
            error("currently cannot compute form factor between differnet Chern numbers.")
        end
        @assert Gk[1] > 0 && Gk[2] > 0

        q_coord = (q1 // Gk[1], q2 // Gk[2])
        ql = q_coord[1] * para.V_int.lattice.G1 + q_coord[2] * para.V_int.lattice.G2
        C = cp.Chern[ci]
        F = complex(1.0)
        if form_factor
            F = level_form_factor(cp.level[cf], cp.level[ci], ql; Chern = C)
        end

        scats = Scatter{1}[]
        for ki in axes(k_list, 2)
            kf = momentum_index_search(k_list[1, ki] + q1, k_list[2, ki] + q2; para = para)
            iszero(kf) && continue
            
            index_i = ki + para.Nk * (ci - 1)
            index_f = kf + para.Nk * (cf - 1)

            ki_coord = (k_list[1, ki] // Gk[1], k_list[2, ki] // Gk[2])
            kf_coord = (k_list[1, kf] // Gk[1], k_list[2, kf] // Gk[2])

            phase_angle  = C * ql_cross(kf_coord, ki_coord)
            phase_angle += C * ql_cross(kf_coord .+ ki_coord, q_coord )
            phase = cispi(phase_angle)

            G = round.(Int64, kf_coord .- ki_coord .- q_coord, RoundNearest)
            sign = ita(G[1], G[2])

            push!(scats, MomentumED.NormalScatter(sign*F*phase, index_f, index_i; upper_hermitian = false))
        end

        return MBOperator(scats; upper_hermitian = false)
    end




end
