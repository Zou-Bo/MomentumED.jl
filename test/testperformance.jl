# These packages are not included in the MomentumED package. 
# Use the following line to add them:
# using Pkg; Pkg.add("CairoMakie"); Pkg.add("QuadGK"); Pkg.add("ClassicalOrthogonalPolynomials")


# Import the momentum-conserved exact diagonalization package
using MomentumED
using MomentumED: ED_bracket_threaded, multiplication_threaded
include("../examples/Landau level triangular.jl")
using .LLT


k_list = [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3;
          0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]

# System parameters
Nk = 20         # Total number of k-points
Gk = (4, 5)     # Grid dimensions (G1_direction, G2_direction)
Ne = 4          # Ne electrons for this system

using LinearAlgebra
function pseudopotential(m::Int64)
    local sys_int = LandauInteraction()
    sys_int.V_intra = [fill(0.0, m); 1.0]
    sys_int.mix = 1.0

    local para = EDPara(k_list = k_list, Gk = Gk, V_int = sys_int)
    local scat = ED_sortedScatteringList_twobody(para);
    return MBSOperator{Float64}(scat; upper_triangular = true)
end

m_list = 0:20
ops = Vector{MBSOperator}(undef, length(m_list));
for i in eachindex(m_list)
    ops[i] = pseudopotential(m_list[i])
end

# Set up component parameters: (layer, level, Chern number, pseudospin)
sys_int = LandauInteraction((1, 0, 1, 0));

# use Haldane pseudopotential
sys_int.V_intra = [0.0; 1.0; 0.0; 1.0]

# choose a linear mixing between Haldane and Coulomb interaction
sys_int.mix = 1                  # mix * Haldane + (1-mix) * Coulomb

# Create parameter structure for bilayer system
para = EDPara(k_list = k_list, Gk = Gk, 
    V_int = sys_int, FF_inf_angle = LLT.Landau_ff_inf(sys_int));
blocks, block_k1, block_k2, k0number = 
    ED_momentum_block_division(para, ED_mbslist(para, (Ne,)));
# one-body terms are all zero in flat Landau level
scat = ED_sortedScatteringList_twobody(para);

Neigen = 10  # Number of eigenvalues to compute per block
energies = Vector{Vector{Float64}}(undef, length(blocks));
vectors = Vector{Vector{Vector{ComplexF64}}}(undef, length(blocks));
for i in eachindex(blocks)
    println("Processing block #$i with size $(length(blocks[i])), momentum $(block_k1[i]), $(block_k2[i])")
    energies[i], vectors[i] = EDsolve(blocks[i], scat; N = Neigen,
        showtime=true
    )
end

bn = 11
using MomentumED: create_state_mapping, ED_apply
mapping = MomentumED.create_state_mapping(blocks[bn])
myvec = MBS64Vector(vectors[bn][1], mapping);
opop = MBSOperator{Float64}(ops[1].scats[15:15]; upper_triangular = true);
opop.scats



using BenchmarkTools


@benchmark opop * myvec
@benchmark multiplication_threaded(opop, myvec, multi_thread=false)

@code_warntype opop * myvec
@code_warntype multiplication_threaded(opop, myvec, multi_thread=false)

@code_llvm opop * myvec
@code_llvm multiplication_threaded(opop, myvec, multi_thread=false)



E_m1 = zeros(Float64, length(m_list));
@time for i in eachindex(m_list)
    print(i, ' ')
    E_m1[i] += ED_bracket(myvec, ops[i], myvec) |> real
end

E_m2 = zeros(Float64, length(m_list));
@time for i in eachindex(m_list)
    print(i, ' ')
    E_m2[i] += myvec ⋅ (ops[i] * myvec) |> real
end

E_m3 = zeros(Float64, length(m_list));
@time for i in eachindex(m_list)
    print(i, ' ')
    for scat in ops[i].scats
        E_m3[i] += vectors[bn][1] ⋅ ED_apply(scat, vectors[bn][1], mapping) |> real
    end
end

E_m4 = zeros(Float64, length(m_list));
@time for i in eachindex(m_list)
    print(i, ' ')
    E_m4[i] += ED_bracket_threaded(myvec, ops[i], myvec) |> real
end

E_m5 = zeros(Float64, length(m_list));
@time for i in eachindex(m_list)
    print(i, ' ')
    E_m5[i] += myvec ⋅ multiplication_threaded(ops[i], myvec) |> real
end

E_m6 = zeros(Float64, length(m_list));
@time for i in eachindex(m_list)
    print(i, ' ')
    E_m6[i] += myvec ⋅ multiplication_threaded(ops[i], myvec; multi_thread = false) |> real
end


using CairoMakie
begin
    fig = Figure();
    ax = Axis(fig[1,1])
    scatterlines!(ax, m_list, E_m1, label = "1")
    scatterlines!(ax, m_list, E_m2, label = "2")
    # scatterlines!(ax, m_list, E_m3, label = "3")
    scatterlines!(ax, m_list, E_m4, label = "4")
    scatterlines!(ax, m_list, E_m5, label = "5")
    scatterlines!(ax, m_list, E_m6, label = "6")
    axislegend(ax)
    fig
end



