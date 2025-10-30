"Parameters and functions for Landau levels with torus geometry and periodic boundary condition"
module LLD

    using ClassicalOrthogonalPolynomials: laguerrel
    using QuadGK
    using MomentumED
    using MomentumED.EDCore.Combinatorics

    export AngularComponentList, AngularLandauInteraction
    export Landau_ff_inf, density_operator

    # Global variables, usually no need to change
    const W0::Float64 = 1.0                     # Interaction strength (energy unit)
    const l::Float64 = 1.0                      # Magnetic Length (length/momentum unit)
    # Gl_cutoff::Float64 = 10.0                   # number of shells (|Gl| <= Gl_cutoff) included in the interaction 
    PRINT_INT_DETAIL::Bool = false              # print the details in calculating the interaction

    # a list of component (layer, level, Chern number, other/pseudospin)
    # layer can only be 1 or 2; Chern number can only be 1 or -1
    struct AngularComponentList
        
        layer::Vector{Int64}       # inter-/intra-layer interaction
        level::Vector{Int64}       # only level index is not conserved in the interaction
        Chern::Vector{Int64}       # affect sign of the form factor phase
        other::Vector{Int64}       # other indices (eg. spin, valley, sublattice, ...)

        function AngularComponentList(components::Tuple{Int64, Int64, Int64, Int64}...)
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


    


    # Decompose Coulomb interaction into Haldane pseudopotentials
    # function can be V_Coulomb_monolayer, V_Coulomb_bilayer; keywords can be SameLayer = true/false
    function pseudo_potential_decomposition(m::Int64; kwds...)::Float64
        results = quadgk(0.0, Inf) do q²l²
            laguerrel(m, 0, q²l²) * V_Coulomb(sqrt(q²l²); kwds...) * exp(-q²l²)
        end
        return results[1] / W0
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


    # Notice: prefactor 2^(-m1-m2) is left out
    function Clebsch_Gordan_2D(m1::Int64, m2::Int64, m_r_cutoff::Int64 = m1 + m2)
        M = m1 + m2
        V_m = zeros(ComplexF64, m_r_cutoff+1)
        for m_r in 0:m_r_cutoff
            M_com = M - m_r
            m_less, m_greater = minmax(m1, m2)
            M_less, M_greater = minmax(M_com, m_r)
            prefactor = sqrt(factorial(M_greater, m_less) / bimonial(m_greater, M_less))
            for k in 0:M
                V_m[1+m_r] += (-1)^(m_r-m1+k) * binomial(m1, k) * binomial(m2, M_com - k)
            end
            V_m[1+m_r] *= prefactor
        end
        return V_m
    end

    # there're should be mf1 + mf2 = mi1 + mi2, although no assertion
    function interaction_amplitude( V_m::Vector{Float64},
        mf1::Int64, mf2::Int64, mi2::Int64, mi1::Int64)

        m_r_cutoff = min(mi1 + mi2, length(V_m)-1)
        CGi = Clebsch_Gordan_2D(mi1, mi2, m_r_cutoff)
        CGf = Clebsch_Gordan_2D(mif, mif, m_r_cutoff)
        return sum(conj.(CGf) .* view(V_m, 1:m_r_cutoff+1) .* CGi)
    end

    # Callable structure to compute Two-body interaction matrix element
    # This implements the full Coulomb interaction with proper magnetic translation phases
    # The interaction is computed in momentum space with Landau level projection
    # Momentum inputs are Tuple{Float64, Float64} representing (k1, k2) in ratio of Gk
    mutable struct AngularLandauInteraction <: Function

        D_l::Float64                                # Gate distance / magnetic length (D/l)
        d_l::Float64                                # Interlayer distance / magnetic length (d/l)
        V_intra_Coulomb::Vector{Float64}            # Intralayer Haldane pseudo-potential in unit of W0
        V_inter_Coulomb::Vector{Float64}            # Intralayer Haldane pseudo-potential in unit of W0
        V_intra::Vector{Float64}                    # Intralayer Haldane pseudo-potential in unit of W0
        V_inter::Vector{Float64}                    # Interlayer Haldane pseudo-potential in unit of W0
        mix::Real                                   # mix * Haldane + (1-mix) * Coulomb

        const components::AngularComponentList

        function AngularLandauInteraction(
            components::Tuple{Int64, Int64, Int64, Int64}...
        )
            new(10.0, 0.0, Float64[], Float64[], [0.0; 0.7;], [1.5; 0.0;], 0.0, AngularComponentList(components...))
        end

    end

    # always call this to convert coulomb into V_m
    function generate_Coulomb_interaction(ALI::AngularLandauInteraction, m_cutoff::Int64)::Nothing
        ALI.V_intra_Coulomb = pseudo_potential_decomposition.(0:m_cutoff;
            samelayer = true, d_l = ALI.d_l, D_l = ALI.D_l
        )
        if iszero(ALI.d_l)
            ALI.V_inter_Coulomb = ALI.V_intra_Coulomb
        else
            ALI.V_inter_Coulomb = pseudo_potential_decomposition.(0:m_cutoff;
                samelayer = false, d_l = ALI.d_l, D_l = ALI.D_l
            )
        end
        return
    end

    function (V_int::AngularLandauInteraction)(
        kf1::Tuple{Int64,Int64}, kf2::Tuple{Int64,Int64}, ki2::Tuple{Int64,Int64}, ki1::Tuple{Int64,Int64}, 
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

        # Level conservation: this should not be required, but the current angular momentum conservation needs it.
        if ni1 != nf1 || ni2 != nf2
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

        mf1 = Int64(kf1[1])
        mf2 = Int64(kf2[1])
        mi2 = Int64(ki2[1])
        mi1 = Int64(k21[1])

        if li1 ==li2
            V_H = interaction_amplitude(V_intra, mf1, mf2, mi2, mi1)
            V_C = interaction_amplitude(V_intra_Coulomb, mf1, mf2, mi2, mi1)
        else
            V_H = interaction_amplitude(V_inter, mf1, mf2, mi2, mi1)
            V_C = interaction_amplitude(V_inter_Coulomb, mf1, mf2, mi2, mi1)
        end

        # mixed Coulomb and Haldane interaction
        return V_C + V_int.mix * (V_H - V_C)
    end




    # # Define the Landau level infinitesimal form factor
    # function Landau_ff_inf(LI::LandauInteraction)::Function
    #     return (k_f::Tuple{<:Real, <:Real}, k_i::Tuple{<:Real, <:Real}, c::Int64 = 1) -> begin
    #         dk = k_f .- k_i
    #         k = 0.5 .* (k_f .+ k_i)
    #         return LI.components.Chern[c] * π * (k[1]*dk[2] - k[2]*dk[1])
    #     end
    # end





    # function momentum_index_search(k1::Int64, k2::Int64; para::EDPara)::Int64
    #     Gk1, Gk2 = para.Gk
    #     for i in axes(para.k_list, 2)
    #         Gk1 == 0 && para.k_list[1,i] != k1 && continue
    #         Gk1 != 0 && mod(para.k_list[1,i], Gk1) != mod(k1, Gk1) && continue
    #         Gk2 == 0 && para.k_list[2,i] != k2 && continue
    #         Gk2 != 0 && mod(para.k_list[2,i], Gk2) != mod(k2, Gk2) && continue
    #         return i
    #     end
    #     return 0
    # end

    # # ρ_{cf, ci}(q) = Σ_{ki,kf} < kf| e^{iqr} | ki > * c†_{kf, cf} c_{ki, ci}
    # function density_operator(q1::Int64, q2::Int64, cf::Int64=1, ci::Int64=1;
    #     para::EDPara, form_factor::Bool)::MBOperator
    #     cp = para.V_int.components
    #     Gk = para.Gk
    #     k_list = para.k_list

    #     @assert para.V_int isa LandauInteraction "the ED parameter is not generated by LandauInteraction in LLT module."
    #     if cp.Chern[cf] != cp.Chern[ci]
    #         error("currently cannot compute form factor between differnet Chern numbers.")
    #     end
    #     @assert Gk[1] > 0 && Gk[2] > 0

    #     q_coord = (q1 // Gk[1], q2 // Gk[2])
    #     ql = (q_coord[1] * para.V_int.lattice.G1 + q_coord[2] * para.V_int.lattice.G2) * l
    #     C = cp.Chern[ci]
    #     F = complex(1.0)
    #     if form_factor
    #         F = level_form_factor(cp.level[cf], cp.level[ci], ql; Chern = C)
    #     end

    #     scats = Scatter{1}[]
    #     for ki in axes(k_list, 2)
    #         kf = momentum_index_search(k_list[1, ki] + q1, k_list[2, ki] + q2; para = para)
    #         iszero(kf) && continue
            
    #         index_i = ki + para.Nk * (ci - 1)
    #         index_f = kf + para.Nk * (cf - 1)

    #         ki_coord = (k_list[1, ki] // Gk[1], k_list[2, ki] // Gk[2])
    #         kf_coord = (k_list[1, kf] // Gk[1], k_list[2, kf] // Gk[2])

    #         phase_angle  = C * ql_cross(kf_coord, ki_coord)
    #         phase_angle += C * ql_cross(kf_coord .+ ki_coord, q_coord )
    #         phase = cispi(phase_angle)

    #         G = round.(Int64, kf_coord .- ki_coord .- q_coord, RoundNearest)
    #         sign = ita(G[1], G[2])

    #         push!(scats, MomentumED.NormalScatter(sign*F*phase, index_f, index_i; upper_hermitian = false))
    #     end

    #     return MBOperator(scats; upper_hermitian = false)
    # end




end
