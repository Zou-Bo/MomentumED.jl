# These packages are not included in the MomentumED package. 
# Use the following line to add them:
# using Pkg; Pkg.add("CairoMakie"); Pkg.add("QuadGK"); Pkg.add("ClassicalOrthogonalPolynomials")

# Import the momentum-conserved exact diagonalization package
using MomentumED, LinearAlgebra
include("Landau level torus.jl")
using .LLT

using CairoMakie # for plotting
CairoMakie.activate!()

# Plot the energy spectrum
function plot_ed_spectrum(energies, ss_k, Gk; 
    title = nothing, ylims = (nothing, nothing),
    ylabel = "Energy per unit cell (W₀ = e²/ϵl)")

    fig = Figure();
    ax = Axis(fig[1, 1];
        xlabel = "k1+$(Gk[1])k2",
        ylabel = ylabel
    )
    ax_top = Axis(fig[1, 1];
        xaxisposition = :top
    )
    top_ticks = ([], [])
    hidespines!(ax_top)
    hidexdecorations!(ax_top; label = false, ticklabels = false)
    hideydecorations!(ax_top)
    linkxaxes!(ax, ax_top)

    # Plot energy levels for each momentum block
    for i in eachindex(ss_k)
        x = ss_k[i][1] + Gk[1] * ss_k[i][2]
        push!(top_ticks[1], x)
        push!(top_ticks[2], string(i))
        if isassigned(energies,i)
            for e in energies[i]
                scatter!(ax, x, e, color = :blue, marker=:hline)
            end
        end
    end
    ylims!(ax, ylims...)
    ax_top.xticks = top_ticks
    if title isa String
        ax_top.subtitle = title
    end
    display(fig)
    fig
end

# System parameters
k_list = [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3;
          0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5]
Nk = 24         # Total number of k-points
Gk = (4, 6)     # Grid dimensions (G1_direction, G2_direction)
Ne = 8          # Ne electrons for this system
# Set up component parameters: (layer, level, Chern number, pseudospin)
sys_int = LandauInteraction(ReciprocalLattice(:triangular), (1, 0, 1, 0));
sys_int.D_l = 10.0                  # Screening length D/l

# Create parameter structure for bilayer system
para = EDPara(; k_list, Gk, H_two = sys_int);

# Create momentum blocks (Hilbert subspace)
subspaces, ss_k = ED_momentum_subspaces(para, (Ne, ));
display(length.(subspaces))

# one-body terms are all zero in flat Landau level
scat = ED_scatterlist_twobody(para);

hmlt = MBOperator(scat, upper_hermitian = true)



using CUDA
MomentumED.CUDA_MEMORY_MONITOR[] # default value true
MomentumED.CUDA_MEMORY_MONITOR[] = false


MomentumED.release_cuda()


e_sparse, _ = EDsolve(subspaces[1], hmlt; N=4, method=:sparse);
e_gpu, _    = EDsolve(subspaces[1], hmlt; N=4, method=:cuda_map);
@show maximum(abs.(e_sparse .- e_gpu))



Neigen = 4  # Number of eigenvalues to compute per subspace
energies = Vector{Vector{Float64}}(undef, length(subspaces));
vectors = Vector{Vector{<:MBS64Vector}}(undef, length(subspaces));
range = 1:8
@time for i in eachindex(subspaces)[range]
    println("Processing subspace #$i with size $(length(subspaces[i])), momentum $(ss_k[i])")
    energies[i], vectors[i] = EDsolve(subspaces[i], hmlt;
        N = Neigen, showtime = true, ishermitian = true, method_info = false,
        # method = :sparse,
        # method = :map,
        method = :gpu_map, 
        verbosity = 2
    )
end

plot_ed_spectrum(energies[range]/Nk/LLT.W0, ss_k, Gk;
    title = "Nk = $Nk, Ne = $Ne",
);


