using Test

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using EDCore

@testset "EDCore" begin
    @testset "manybodystate_basis.jl" begin
        include("test_manybodystate_basis.jl")
    end
    @testset "hilbert_subspace.jl" begin
        include("test_hilbert_subspace.jl")
    end
    @testset "manybodystate_vector.jl" begin
        include("test_manybodystate_vector.jl")
    end
    @testset "colex_mbslist.jl" begin
        include("test_colex_mbslist.jl")
    end
    @testset "scattering.jl" begin
        include("test_scattering.jl")
    end
    @testset "operator.jl" begin
        include("test_operator.jl")
    end
    @testset "multiplication.jl" begin
        include("test_multiplication.jl")
    end
end
