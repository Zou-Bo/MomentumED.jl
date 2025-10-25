
@testset "MBOperator" begin
    s1 = NormalScatter(1.0+0.0im, 1, 2)
    s2 = NormalScatter(0.5+0.0im, 1, 2, 4, 3)
    op = MBOperator([s1], [s2]; upper_hermitian=false)

    @test !isupper(op)
    @test length(op.scats) == 2
    @test eltype(op.scats[1]) == Scatter{1}
    @test eltype(op.scats[2]) == Scatter{2}

    op_adj = adjoint(op)
    @test op_adj.scats[1][1] == adjoint(s1)
    @test op_adj.scats[2][1] == adjoint(s2)
end
