using MomentumED
include("../examples/Landau level triangular.jl")
using .LLT

using CairoMakie # for plotting
CairoMakie.activate!()

# Plot the energy spectrum
function plot_ed_spectrum()
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
    for i in 1:length(blocks)
        x = Gk[2] * block_k1[i] + block_k2[i]
        push!(top_ticks[1], x)
        push!(top_ticks[2], string(i))
        for e in energies[i]
            scatter!(ax, x, e/Nk/LLT.W0, color = :blue, marker=:hline)
        end
    end
    ax_top.xticks = top_ticks
    fig
end

# Define 18 k-mesh for 1/3 filling Laughlin state calculation
k_list = [0 3 2 5 1 4 0 3 2 5 1 4;
          0 0 1 1 2 2 3 3 4 4 5 5];

# System parameters
Nk = 12;         # Total number of k-points
Gk = (6, 6);     # Grid dimensions (G1_direction, G2_direction)
Ne = 4;          # Ne electrons for this system

# Set up component parameters: (layer, level, Chern number, pseudospin)
sys_int = LandauInteraction((1, 0, 1, 0));

# Coulomb
sys_int.D_l = 10.0;                  # Screening length D/l

# choose a linear mixing between Haldane and Coulomb interaction
sys_int.mix = 0;                 # mix * Haldane + (1-mix) * Coulomb

# Create parameter structure for bilayer system
para = EDPara(k_list = k_list, 
    Gk = Gk, 
    V_int = sys_int,
    FF_inf_angle = LLT.Landau_ff_inf(sys_int)
);


blocks, block_k1, block_k2, k0number = 
    ED_momentum_block_division(para, ED_mbslist(para, (Ne,)));
@show length.(blocks)
# one-body terms are all zero in flat Landau level
scat = ED_sortedScatteringList_twobody(para);

Neigen = 10;  # Number of eigenvalues to compute per block
energies = Vector{Vector{Float64}}(undef, length(blocks));
vectors = Vector{Vector{Vector{ComplexF64}}}(undef, length(blocks));
for i in eachindex(blocks)
    println("Processing block #$i with size $(length(blocks[i])), momentum $(block_k1[i]), $(block_k2[i])")
    energies[i], vectors[i] = EDsolve(blocks[i], scat; 
        N = Neigen, showtime=true
    )
end

plot_ed_spectrum()


bn = 1;
energies[bn]./Nk
mapping = MomentumED.create_state_mapping(blocks[bn]);
myvec = MBS64Vector(vectors[bn][1], mapping);

densities = MBSOperator[density_operator(q1, q2, 1, 1; para=para)
    for q1 in -2Gk[1]:2Gk[1], q2 in -2Gk[2]:2Gk[2]
];
begin
    structure_factor = similar(densities, ComplexF64)
    index_shift = 2 .* Gk .+ 1
    for q1 in -2Gk[1]:2Gk[1], q2 in -2Gk[2]:2Gk[2]
        structure_factor[index_shift[1]+q1, index_shift[2]+q2] = 
            ED_bracket(myvec, 
                densities[index_shift[1]-q1, index_shift[2]-q2], 
                densities[index_shift[1]+q1, index_shift[2]+q2], myvec
            )
        if mod(q1, Gk[1]) == 0 && mod(q2, Gk[2])== 0
            structure_factor[index_shift[1]+q1, index_shift[2]+q2] -= 
                ED_bracket(myvec, densities[index_shift[1]-q1, index_shift[2]-q2], myvec) * 
                ED_bracket(myvec, densities[index_shift[1]+q1, index_shift[2]+q2], myvec)
        end
    end
    structure_factor ./= Nk
    nothing
end

maximum(abs.(imag.(structure_factor)))
extrema(real.(structure_factor))

begin
    fig = Figure();
    ax = Axis(fig[1,1])
    hm = heatmap!(ax, (-2Gk[1]:2Gk[1])./Gk[1], (-2Gk[2]:2Gk[2])./Gk[2], 
        real.(structure_factor);
        colorrange = (0.0, maximum(real.(structure_factor))),
        colormap = range(Makie.Colors.colorant"white", stop=Makie.Colors.colorant"#ec2f41", length=15)
    )
    Colorbar(fig[1, 2], hm)
    fig
end


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

        scat_list = ED_sortedScatteringList_twobody(para; kshift = path[i+1]);
        vecs = EDsolve(blocks[bn], scat_list; N = 6,
            showtime = false,
        )[2][1:nstates]
        psi_after .= reduce(hcat, vecs)
        ED_connection_gaugefixing!(psi_after)  # fix global phase

        WilsonLoopIntegral[i] = ED_connection_step(blocks[bn], 
            psi_after, psi_before, path[i+1], path[i], para;
            wavefunction_tol = 1e-8, print_amp = true,
            amp_warn_tol = 0.7, amp_warn = true
        )

        psi_before .= psi_after
    end
    ManyBodyChernNumber = sum(WilsonLoopIntegral) / (2π) * N_shift^2
end
