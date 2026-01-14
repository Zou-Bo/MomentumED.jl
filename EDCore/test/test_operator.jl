@testset "MBOperator" begin
    s1 = Scatter(1.0+0.0im, 1, 2, bits=10)
    s2 = Scatter(0.5+0.0im, 1, 2, 4, 3, bits=10)
    op = MBOperator([s1], [s2]; upper_hermitian=false)

    @test !isupper(op)
    @test length(op.scats) == 2
    @test get_body(op.scats[1]) == (1, 1)
    @test get_body(op.scats[2]) == (2, 2)

    op_adj = adjoint(op)
    @test op_adj.scats[1] == adjoint(s1)
    @test op_adj.scats[2] == adjoint(s2)
end