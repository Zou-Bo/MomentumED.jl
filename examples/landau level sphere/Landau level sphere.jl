"Parameters and functions for Landau levels with sphere (Haldane) geometry — monopole at center."
module LLS

    using WignerSymbols: clebschgordan, wigner3j, wigner6j
    using FastGaussQuadrature: gausslegendre
    using MomentumED

    export AngularComponentList, SphereLandauInteraction
    export generate_Coulomb_interaction, pseudopotential_V_mrel, legendre_coeffs
    export density_operator, orbital_overlap


    # Global variables, usually no need to change
    const W0::Float64 = 1.0                     # Interaction strength (energy unit = e²/(ε ℓ_B))
    const l::Float64 = 1.0                      # Magnetic length (length unit)
    PRINT_INT_DETAIL::Bool = false              # print details in computing the interaction


    # ---------------------------------------------------------------------------
    # Component list — same structure as torus/disk version
    # ---------------------------------------------------------------------------
    struct AngularComponentList

        layer::Vector{Int64}       # inter-/intra-layer interaction
        level::Vector{Int64}       # Landau-level index
        Chern::Vector{Int64}       # Chern number
        other::Vector{Int64}       # other indices (spin, valley, sublattice, ...)

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


    # ---------------------------------------------------------------------------
    # Sphere Coulomb pseudopotentials
    # ---------------------------------------------------------------------------
    #
    # On a sphere of radius R = √S ℓ_B with monopole strength 2S, the Coulomb
    # potential between two particles at chord distance r is V(r) = e²/(ε r).
    # With r = 2R sin(θ/2), the standard Legendre expansion (Jackson §3, Edmonds
    # §4.6) is:
    #     V(θ) = e²/(ε R) Σ_{k=0}^{∞} P_k(cos θ),     v_k = e²/(ε R) ≡ W₀/R
    #
    # Bilayer: two concentric spheres of the same radius R separated radially by
    # distance d. The interlayer Coulomb potential at angular separation θ is
    #     V_inter(θ) = W₀ / √((2R sin(θ/2))² + d²)
    # which has no closed-form Legendre expansion; we compute v_k by
    # Gauss-Legendre quadrature.
    #
    # The Haldane pseudopotentials V_{m_rel} are eigenvalues of the two-body
    # interaction in the channel of relative angular momentum m_rel ∈ {0,...,2S}.
    # By rotational invariance the interaction is block-diagonal in total pair
    # angular momentum L and proportional to the identity within each block;
    # the eigenvalue in the L-block equals V_{m_rel} with m_rel = 2S - L.
    #
    # Using Wooten-Macek arXiv:1408.5379 Eq. (10) (originally Fano-Ortolani-Colombo
    # PRB 34, 2670 (1986) and Wu-Yang 1976/77):
    #
    #   V_L = (-1)^{2S+L} (2S+1)² Σ_{k=0}^{2S} v_k ·
    #         { L  S  S }      ( S  k  S )²
    #         { k  S  S }   ·  (-S  0  S )
    #
    # then V_{m_rel} = V_{L=2S-m_rel}. We store and expose everything in the
    # m_rel-indexed convention (Haldane's standard convention).
    #
    # For fermions only odd m_rel matter; the most repulsive Haldane channel is
    # V_{m_rel=1}, which stabilizes Laughlin 1/3 (e.g. V_1 with all others zero
    # is the exact parent Hamiltonian for the Laughlin state).
    #
    # Verified against direct Monte-Carlo integration of ⟨L,0|V|L,0⟩ for the
    # LLL pair state on the sphere; agreement to within MC noise (~1% at 3e5
    # samples) for 2S = 9.

    "Legendre coefficients {v_k}_{k=0}^{2S} for the sphere Coulomb potential.
     Convention: V(θ) = Σ_k v_k P_k(cos θ), with chord r = 2R sin(θ/2),
     interlayer separation d, total pair distance √(r²+d²), and R = √S ℓ_B.

     - `intralayer_formula = true` (default), `same_layer = true`:
       Closed form v_k = W₀/R for all k. Cheap and exact.
     - Otherwise (`intralayer_formula = false`, or `same_layer = false`):
       Numerical integration. Uses the substitution u = sin(θ/2) ∈ [0,1] which
       removes the θ → 0 Coulomb singularity (the u-Jacobian exactly cancels
       the 1/r factor). The integrand is smooth on [0,1] for any d ≥ 0; Gauss-
       Legendre quadrature with n ≈ 3(2S+1) nodes converges exponentially.

       Derivation:
         v_k = (2k+1)/2 · ∫₋₁¹ V(θ) P_k(cosθ) d(cosθ),  cosθ = 1 - 2u²
             = (2k+1) · ∫₀¹  [2u · V(u)] · P_k(1-2u²) du
             = (2k+1) · ∫₀¹  2u·W₀/√(4R²u² + d²) · P_k(1-2u²) du
       (For d=0: integrand = W₀/R · P_k(1-2u²), explicitly smooth.)"
    function legendre_coeffs(two_S::Int64; d_l::Float64 = 0.0,
            same_layer::Bool = true, intralayer_formula::Bool = true)::Vector{Float64}
        @assert two_S > 0
        R = sqrt(two_S / 2.0) * l

        if same_layer && intralayer_formula
            # Analytic: v_k = W₀/R for all k = 0, …, 2S
            return fill(W0 / R, two_S + 1)
        end

        # Numerical path (intralayer cross-check, or interlayer with d_l ≥ 0):
        # Substitute u = sin(θ/2), u ∈ [0,1]. Map Gauss-Legendre nodes from
        # [-1, 1] to [0, 1] via u = (t + 1)/2, with the corresponding 1/2 weight.
        d_eff = same_layer ? 0.0 : d_l
        n_nodes = max(3 * (two_S + 1), 32)
        ts, ws_raw = gausslegendre(n_nodes)
        us = @. (ts + 1) / 2                            # u nodes in [0, 1]
        ws = ws_raw ./ 2                                # weights for ∫₀¹ du
        # Build all P_k(x_j) at x_j = 1 - 2u_j² for k = 0, …, 2S via recurrence
        xs = @. 1 - 2 * us^2
        P = zeros(two_S + 1, n_nodes)
        P[1, :] .= 1.0                                  # P_0(x) = 1
        two_S >= 1 && (P[2, :] .= xs)                   # P_1(x) = x
        for k in 1:(two_S - 1)
            # (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}
            @views @. P[k + 2, :] = ((2k + 1) * xs * P[k + 1, :] - k * P[k, :]) / (k + 1)
        end
        # Integrand factor: 2u · V(u) where V(u) = W₀/√(4R²u² + d²)
        integrand_factor = @. 2 * us * W0 / sqrt(4 * R^2 * us^2 + d_eff^2)
        v = zeros(two_S + 1)
        for k in 0:two_S
            v[k + 1] = (2k + 1) * sum(ws .* integrand_factor .* @view P[k + 1, :])
        end
        return v
    end


    "Haldane pseudopotential V_{m_rel} at relative angular momentum m_rel ∈ {0,...,2S}
     on the sphere. Takes a precomputed vector v_k_vec of Legendre coefficients
     v_k for k = 0, …, 2S (see `legendre_coeffs`). Units: e²/(ε ℓ_B).

     Formula: V_{m_rel} = V_{L=2S-m_rel}, where
       V_L = (-1)^{2S+L} (2S+1)² Σ_k v_k · 6j{L S S; k S S} · 3j(S k S; -S 0 S)²
     (Wooten-Macek arXiv:1408.5379 Eq. (10).)"
    function pseudopotential_V_mrel(m_rel::Int64, two_S::Int64,
            v_k_vec::Vector{Float64})::Float64
        @assert 0 <= m_rel <= two_S
        @assert length(v_k_vec) == two_S + 1
        L = two_S - m_rel       # total pair angular momentum

        S_h = two_S // 2      # S = Q = l (LLL), as Rational
        V_total = 0.0

        for k in 0:two_S
            vk = v_k_vec[k + 1]
            iszero(vk) && continue

            # 3j ( S  k  S ; -S  0  S )
            tj = Float64(wigner3j(Float64, S_h, k, S_h, -S_h, 0, S_h))
            iszero(tj) && continue
            # 6j { L  S  S ; k  S  S }
            sj = Float64(wigner6j(Float64, L, S_h, S_h, k, S_h, S_h))
            iszero(sj) && continue

            V_total += vk * sj * tj^2
        end

        sign = isodd(two_S + L) ? -1.0 : 1.0      # (-1)^(2S + L)
        V_total *= sign * (two_S + 1)^2

        return V_total
    end


    # ---------------------------------------------------------------------------
    # Interaction struct and callable
    # ---------------------------------------------------------------------------

    "Two-body LLL interaction on the Haldane sphere.

     Momenta: single-component tuples (m̃,) with m̃ ∈ {0, 1, ..., 2S}.
     Physical L_z = m̃ - S.

     Pseudopotential arrays V_intra_Coulomb, V_inter_Coulomb, V_intra, V_inter
     are all indexed by relative angular momentum: V_intra[m_rel + 1] gives the
     Haldane pseudopotential at relative angular momentum m_rel ∈ {0, 1, ..., 2S}.
     Arrays shorter than 2S+1 are treated as having trailing zeros (useful for
     sparse Haldane PPs, e.g. V_intra = [0.0, 1.0] for Laughlin 1/3 parent).

     The callable returns N_phi · ⟨m̃f1 m̃f2 | V | m̃i1 m̃i2⟩ so after MomentumED
     divides by N_k = N_phi, one recovers ⟨...|V|...⟩."
    mutable struct SphereLandauInteraction <: Function

        N_phi::Int64                                # 2S + 1 (number of orbitals)
        d_l::Float64                                # interlayer distance / ℓ_B (future)
        V_intra_Coulomb::Vector{Float64}            # V_{m_rel}, m_rel=0..2S, intralayer
        V_inter_Coulomb::Vector{Float64}            # V_{m_rel}, m_rel=0..2S, interlayer
        V_intra::Vector{Float64}                    # model Haldane pseudopotentials (intralayer)
        V_inter::Vector{Float64}                    # model Haldane pseudopotentials (interlayer)
        mix::Real                                   # mix*Haldane + (1-mix)*Coulomb

        const components::AngularComponentList

        function SphereLandauInteraction(N_phi::Int64,
                components::Tuple{Int64, Int64, Int64, Int64}...)
            @assert N_phi >= 2 "need N_phi ≥ 2 (2S ≥ 1)"
            new(N_phi, 0.0, Float64[], Float64[], Float64[], Float64[], 0.0,
                AngularComponentList(components...))
        end
    end

    "2S (monopole strength) from the interaction struct."
    two_S(SLI::SphereLandauInteraction)::Int64 = SLI.N_phi - 1


    "Fill SLI.V_intra_Coulomb (and V_inter_Coulomb if d_l ≠ 0) with sphere
     Haldane pseudopotentials V_{m_rel} for m_rel = 0, 1, …, 2S.

     Keyword `intralayer_formula = true` (default) uses the analytic v_k = W₀/R.
     Setting to `false` would use numerical quadrature for cross-checking the
     formula path (not yet implemented).

     For interlayer Coulomb with d_l ≠ 0, v_k is always computed by Gauss-Legendre
     quadrature. When d_l == 0, V_inter_Coulomb is just a copy of V_intra_Coulomb."
    function generate_Coulomb_interaction(SLI::SphereLandauInteraction;
            intralayer_formula::Bool = true)::Nothing
        tS = two_S(SLI)

        # Intralayer
        v_k_intra = legendre_coeffs(tS; d_l = 0.0, same_layer = true,
                                    intralayer_formula = intralayer_formula)
        SLI.V_intra_Coulomb = [pseudopotential_V_mrel(m, tS, v_k_intra) for m in 0:tS]

        # Interlayer
        if iszero(SLI.d_l)
            SLI.V_inter_Coulomb = copy(SLI.V_intra_Coulomb)
        else
            v_k_inter = legendre_coeffs(tS; d_l = SLI.d_l, same_layer = false)
            SLI.V_inter_Coulomb = [pseudopotential_V_mrel(m, tS, v_k_inter) for m in 0:tS]
        end
        return
    end


    function (V_int::SphereLandauInteraction)(
            kf1::Tuple{<:Real}, kf2::Tuple{<:Real},
            ki2::Tuple{<:Real}, ki1::Tuple{<:Real},
            cf1::Int64 = 1, cf2::Int64 = 1, ci2::Int64 = 1, ci1::Int64 = 1
        )::ComplexF64

        @inline read_component(v) = v[cf1], v[cf2], v[ci2], v[ci1]
        lf1, lf2, li2, li1 = read_component(V_int.components.layer)
        nf1, nf2, ni2, ni1 = read_component(V_int.components.level)
        Cf1, Cf2, Ci2, Ci1 = read_component(V_int.components.Chern)
        sf1, sf2, si2, si1 = read_component(V_int.components.other)

        # Conservation: layer, Chern, pseudo-spin, Landau level (LLL-only)
        (li1 != lf1 || li2 != lf2) && return 0.0 + 0.0im
        (Ci1 != Cf1 || Ci2 != Cf2) && return 0.0 + 0.0im
        (si1 != sf1 || si2 != sf2) && return 0.0 + 0.0im
        (ni1 != nf1 || ni2 != nf2) && return 0.0 + 0.0im

        m̃f1 = Int64(kf1[1])
        m̃f2 = Int64(kf2[1])
        m̃i1 = Int64(ki1[1])
        m̃i2 = Int64(ki2[1])

        N_phi = V_int.N_phi
        tS = N_phi - 1

        # L_z conservation
        (m̃f1 + m̃f2 != m̃i1 + m̃i2) && return 0.0 + 0.0im

        # Physical L_z: m = m̃ - S (Rational to handle half-integer S)
        S_h = tS // 2
        mi1 = m̃i1 - S_h
        mi2 = m̃i2 - S_h
        mf1 = m̃f1 - S_h
        mf2 = m̃f2 - S_h
        M = mi1 + mi2

        same_layer = (li1 == li2)
        V_C = same_layer ? V_int.V_intra_Coulomb : V_int.V_inter_Coulomb
        V_H = same_layer ? V_int.V_intra          : V_int.V_inter

        # ⟨mf1 mf2 | V | mi1 mi2⟩ = Σ_L V_{m_rel=2S-L} ·
        #     CG(S mi1; S mi2|L M) · CG(S mf1; S mf2|L M)
        # Arrays are indexed by m_rel; short arrays treated as zero-padded.
        @inline function lookup(V_arr::Vector{Float64}, m_rel::Int64)
            (m_rel + 1) <= length(V_arr) ? V_arr[m_rel + 1] : 0.0
        end

        V_sum::ComplexF64 = 0.0
        for L in 0:tS
            abs(M) > L && continue
            cgi = clebschgordan(S_h, mi1, S_h, mi2, L, M)
            iszero(cgi) && continue
            cgf = clebschgordan(S_h, mf1, S_h, mf2, L, M)
            iszero(cgf) && continue

            m_rel = tS - L
            V_L_eff = (1 - V_int.mix) * lookup(V_C, m_rel) + V_int.mix * lookup(V_H, m_rel)
            V_sum += V_L_eff * cgi * cgf
        end

        PRINT_INT_DETAIL && println("($m̃f1,$m̃f2|$m̃i1,$m̃i2)  V=$V_sum")

        return V_sum
    end


    # ---------------------------------------------------------------------------
    # TODO: density operator
    # ---------------------------------------------------------------------------
    """
        density_operator(L::Int64, M::Int64, cf::Int64 = 1, ci::Int64 = 1;
                         para::EDPara) -> MBOperator

    Projected density operator ρ_{L,M} in the LLL on the sphere:
        ρ_{L,M} = Σ_{m,m'} ⟨S,m'| Y_{L,M} |S,m⟩  c†_{m'} c_m
    via Wigner-Eckart (3j symbols).

    TODO: implement.
    """
    function density_operator(L::Int64, M::Int64, cf::Int64 = 1, ci::Int64 = 1;
            para::EDPara)
        error("density_operator not yet implemented for sphere")
    end


    # ---------------------------------------------------------------------------
    # TODO: orbital overlap for Berry curvature / QGT
    # ---------------------------------------------------------------------------
    """
        orbital_overlap(para::EDPara, k_f, k_i) -> Vector{ComplexF64}

    Single-particle form factor for geometric-response calculations
    (Wigner d-matrix rotation matrix elements).

    TODO: implement.
    """
    function orbital_overlap(para::EDPara, k_f::Tuple{<:Real}, k_i::Tuple{<:Real})
        error("orbital_overlap not yet implemented for sphere")
    end

end # module LLS