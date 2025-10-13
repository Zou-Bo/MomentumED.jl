# These packages are not included in the MomentumED package. 
# Use the following line to add them:
# using Pkg; Pkg.add("CairoMakie"); Pkg.add("QuadGK"); Pkg.add("ClassicalOrthogonalPolynomials")

# Import the momentum-conserved exact diagonalization package
using MomentumED
include("../examples/Landau level triangular.jl")
using .LLT

using CairoMakie # for plotting
CairoMakie.activate!()

# Plot the energy spectrum
function plot_ed_spectrum(subtitle=nothing)
    fig = Figure();
    ax = Axis(fig[1, 1];
        xlabel = "$(Gk[2])k1+k2",
        ylabel = "Energy per unit cell (W₀ = e²/ϵl)"
    )
    ax_top = Axis(fig[1, 1];
        xlabel = "momentum block number",
        xaxisposition = :top
    )
    top_ticks = ([], [])
    hidespines!(ax_top)
    hidexdecorations!(ax_top; label = false, ticklabels = false)
    hideydecorations!(ax_top)
    linkxaxes!(ax, ax_top)

    # Plot energy levels for each momentum block
    for i in 1:length(subspaces)
        x = Gk[2] * ss_k1[i] + ss_k2[i]
        push!(top_ticks[1], x)
        push!(top_ticks[2], string(i))
        for e in energies[i]
            scatter!(ax, x, e/Nk/LLT.W0, color = :blue, marker=:hline)
        end
    end
    ax_top.xticks = top_ticks
    if subtitle isa String
        ax_top.subtitle = subtitle
    end
    fig
end

# Define k-mesh for bilayer system (4×3 mesh, Nk=12) in triangular lattice
k_list = [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3;
          0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3]
Nk = 16
Gk = (4, 4)  # Grid dimensions

# number of electrons in each layer
Ne1 = 4
Ne2 = 4

# Number of layers (components) for bilayer system
Nc_conserve = 2


# Set up one-body Hamiltonian matrix for inter-layer tunneling
ΔE = 0.0 * LLT.W0                # Energy difference between the two layers

# H_onebody[c1, c2, cc, k] : no hopping from component c2 to c1
# For bilayer system: Nc_hopping=1, Nc_conserve=2, No tunneling
H_onebody_bilayer_no_tunneling = zeros(ComplexF64, 1, 1, Nc_conserve, Nk);
for k_idx in 1:Nk
    H_onebody_bilayer_no_tunneling[1, 1, 1, k_idx] = 0.5ΔE
    H_onebody_bilayer_no_tunneling[1, 1, 2, k_idx] = -0.5ΔE
end

# Set up component parameters: (layer, level, Chern number, pseudospin)
sys_int = LandauInteraction(
    (1, 0, 1, 0),
    (2, 0, 1, 0)
);

# Interaction parameters

# Coulomb
sys_int.D_l = 10.0                  # Screening length D/l
sys_int.d_l = 0.1                  # Inter-layer distance d/l
# compute the pseudo-potential components
intra_PP = LLT.pseudo_potential_decomposition.(0:5; same_layer = true,  D_l = sys_int.D_l, d_l = sys_int.d_l);
inter_PP = LLT.pseudo_potential_decomposition.(0:5; same_layer = false, D_l = sys_int.D_l, d_l = sys_int.d_l);
display(intra_PP);
display(inter_PP);

# Haldane pseudo-potential
sys_int.V_intra = [0.0; 0.8; 0.0; 0.0]          # Intralayer Haldane pseudo-potential in unit of W0
sys_int.V_inter = [1.5; 0.0; 0.0; 0.0]          # Interlayer Haldane pseudo-potential in unit of W0

# or use Coulomb interaction with a cutoff in m
sys_int.V_intra = copy(intra_PP);
sys_int.V_inter = copy(inter_PP);

# choose a linear mixing between Haldane and Coulomb interaction
sys_int.mix = 1                  # mix * Haldane + (1-mix) * Coulomb

# Create parameter structure for bilayer system
para_bilayer = EDPara(
    k_list = k_list, 
    Gk = Gk, 
    Nc_hopping = 1,
    Nc_conserve = Nc_conserve,
    H_onebody = H_onebody_bilayer_no_tunneling,
    V_int = sys_int,
    FF_inf_angle = Landau_ff_inf(sys_int),
);

NG = 2
index_shift = NG .* Gk .+ 1
densities = MBOperator[density_operator(q1, q2, lf, li; 
        para = para_bilayer, form_factor = true)
    for q1 in -NG*Gk[1]:NG*Gk[1], q2 in -NG*Gk[2]:NG*Gk[2], lf = 1:2, li=1:2
];
function structure_factor_expectation(myvec)
    structure_factor = similar(densities, ComplexF64)
    for q1 in -NG*Gk[1]:NG*Gk[1], q2 in -NG*Gk[2]:NG*Gk[2]
        for lf = 1:2, li = 1:2
            structure_factor[index_shift[1]+q1, index_shift[2]+q2, lf, li] = 
                ED_bracket_threaded(myvec, 
                    densities[index_shift[1]-q1, index_shift[2]-q2, li, lf], 
                    densities[index_shift[1]+q1, index_shift[2]+q2, lf, li], myvec
                )
            if mod(q1, Gk[1]) == 0 && mod(q2, Gk[2])== 0 && lf==li
                structure_factor[index_shift[1]+q1, index_shift[2]+q2, lf, li] -= 
                    ED_bracket(myvec, densities[index_shift[1]-q1, index_shift[2]-q2, li, lf], myvec) * 
                    ED_bracket(myvec, densities[index_shift[1]+q1, index_shift[2]+q2, lf, li], myvec)
            end
        end
    end
    structure_factor ./= Nk
end


# Create momentum blocks for bilayer system
subspaces, ss_k1, ss_k2 = 
    ED_momentum_subspaces(para_bilayer, (Ne1, Ne2));
display(length.(subspaces))
subspaces[1]
# Generate Scatter lists for efficient Hamiltonian construction
scat_list1_conserve = ED_sortedScatterList_onebody(para_bilayer);
scat_list2_conserve = ED_sortedScatterList_twobody(para_bilayer);


Neigen = 10  # Number of eigenvalues to compute per block
energies = Vector{Vector{Float64}}(undef, length(subspaces));
vectors = Vector{Vector{<:MBS64Vector}}(undef, length(subspaces));
for i in eachindex(subspaces)[1:1]
    println("Processing block #$i with size $(length(subspaces[i])), momentum $(ss_k1[i]), $(ss_k2[i])")
    energies[i], vectors[i] = EDsolve(subspaces[i], scat_list2_conserve, scat_list1_conserve;
        N = Neigen, showtime=true, krylovdim = 25, maxiter = 150, verbosity = 2
    )
end

hmlt = MBOperator(scat_list1_conserve, scat_list2_conserve; upper_hermitian = true)

for i in eachindex(subspaces)[1:1]
    println("Processing block #$i with size $(length(subspaces[i])), momentum $(ss_k1[i]), $(ss_k2[i])")
    energies[i], vectors[i] = EDsolve(subspaces[i], hmlt;
        N = Neigen, showtime=true, krylovdim = 25, maxiter = 150, verbosity = 2
    )
end

using MomentumED: LinearMap
@time hmlt_lm = MomentumED.LinearMap(hmlt, subspaces[1], Float64);

x = rand(ComplexF64, length(subspaces[1]));
y = similar(x);
@time hmlt_lm(y, x)


plot_ed_spectrum("Ne1=$Ne1   Ne2=$Ne2")

bn = 1;
energies[bn]./Nk

using MomentumED: ED_HamiltonianMatrix_threaded
@code_warntype ED_HamiltonianMatrix_threaded(subspaces[1], scat_list2_conserve, scat_list1_conserve);
@time H = ED_HamiltonianMatrix_threaded(subspaces[1], scat_list2_conserve, scat_list1_conserve);

@time z = H * x;
maximum(abs.(z .- y))

Base.summarysize(subspaces)
Base.summarysize(subspaces)
Base.summarysize(subspaces)
Base.summarysize(H)

using Combinatorics
binomial(16, 2)^2 * 16


vec331_1 = MBS64Vector(vectors[bn][1], subspaces[bn]);

@time str_fac331_1 = structure_factor_expectation(vec331_1);
maximum(abs.(imag.(str_fac331_1)))
extrema(real.(str_fac331_1))
str_fac331_1[index_shift..., 1,2].re
str_fac331_1[index_shift..., 2,1].re





vec331_2 = MBS64Vector(vectors[bn][7], subspaces[bn]);

@time str_fac331_2 = structure_factor_expectation(vec331_2);
maximum(abs.(imag.(str_fac331_2)))
extrema(real.(str_fac331_2))
str_fac331_2[index_shift..., 1,2].re
str_fac331_2[index_shift..., 2,1].re



mapping = MomentumED.create_state_mapping(blocks[bn]);
vec_mystery = MBS64Vector(vectors[bn][1], mapping);

@time str_fac_mystery = structure_factor_expectation(vec_mystery);
maximum(abs.(imag.(str_fac_mystery)))
extrema(real.(str_fac_mystery))
str_fac_mystery[index_shift..., 1,2]
str_fac_mystery[index_shift..., 2,1]




let structure_factor = str_fac331_1
    layer = (1,2)
    fig = Figure();
    ax = Axis(fig[1,1])
    hm = heatmap!(ax, (-2Gk[1]:2Gk[1])./Gk[1], (-2Gk[2]:2Gk[2])./Gk[2], 
        real.(structure_factor[:,:,layer...]);
        colorrange = (0.0, maximum(real.(structure_factor[:,:,layer...]))),
        colormap = range(Makie.Colors.colorant"white", stop=Makie.Colors.colorant"#ec2f41", length=15)
    )
    Colorbar(fig[1, 2], hm)
    fig
end

# many-body Chern number
begin
    bn =  1                    # block number
    nstates = 1                # number of degenerating states
    
    # twist angle path for the Wilson loop integral
    N_shift = 10  # number of shifts along each edge
    path = Tuple{Float64, Float64}[(0.0, 0.0)]
    push!(path, (1/N_shift, 0.0))
    push!(path, (1/N_shift, 1/N_shift))
    push!(path, (0.0, 1/N_shift))
    push!(path, (0.0, 0.0))

    psi_before = reduce(hcat, vectors[bn][1:nstates])
    ED_connection_gaugefixing!(psi_before)  # fix global phase
    psi_after = similar(psi_before)

    WilsonLoopIntegral= Vector{Float64}(undef, 4)
    for i in eachindex(WilsonLoopIntegral)

        println("path point #$i \t $(path[i+1])")

        scat_list = ED_sortedScatterList_twobody(para_bilayer; kshift = path[i+1]);
        vecs = EDsolve(blocks[bn], scat_list; N = 6,
            showtime = false,
        )[2][1:nstates]
        psi_after .= reduce(hcat, vecs)
        ED_connection_gaugefixing!(psi_after)  # fix global phase

        WilsonLoopIntegral[i] = ED_connection_step(blocks[bn], 
            psi_after, psi_before, path[i+1], path[i], para_bilayer;
            wavefunction_tol = 1e-8, print_amp = true,
            amp_warn_tol = 0.7, amp_warn = true
        )

        psi_before .= psi_after
    end
    ManyBodyChernNumber = sum(WilsonLoopIntegral) / (2π) * N_shift^2
end

