using LinearAlgebra

@testset "MBS64Vector" begin
    sorted_list = [MBS64{8}(UInt(i)) for i in 0:4]
    space = HilbertSubspace(sorted_list, dict=true)
    
    vec_data = rand(ComplexF64, 5)
    mbs_vec = MBS64Vector(vec_data, space)

    @test length(mbs_vec) == 5
    @test size(mbs_vec) == (5,)

    mbs_vec2 = similar(mbs_vec)
    @test mbs_vec2.space === space
    @test length(mbs_vec2.vec) == 5

    @test dot(mbs_vec, mbs_vec) ≈ dot(vec_data, vec_data)
end
