
@testset "Scatter" begin
    s1 = Scatter(1.0, 1, 2)
    @test s1.Amp == 1.0
    @test s1.out == (1,)
    @test s1.in == (2,)
    @test get_body(s1) == 1

    s2 = Scatter(0.5, 1, 2, 4, 3)
    @test s2.Amp == 0.5
    @test s2.out == (1, 2)
    @test s2.in == (3, 4)
    @test get_body(s2) == 2

    @test adjoint(s1) == Scatter{1}(1.0, (2,), (1,))
    @test isdiagonal(Scatter(1.0, 1, 1))
    @test !isdiagonal(s1)
end

@testset "NormalScatter" begin
    # N=1
    s = NormalScatter(1.0 + 1.0im, 2, 1; upper_hermitian=true)
    @test s.in == (2,)
    @test s.out == (1,)
    @test s.Amp == 1.0 - 1.0im

    # N=2
    s = NormalScatter(1.0 + 0.0im, 2, 1, 3, 4; upper_hermitian=true)
    @test s.in == (4,3)
    @test s.out == (2,1)
    @test s.Amp == 1.0

    s = NormalScatter(1.0 + 0.0im, 1, 2, 3, 4; upper_hermitian=true)
    @test s.in == (4,3)
    @test s.out == (2,1)
    @test s.Amp == -1.0
end

@testset "Scatter Operations" begin
    s1 = Scatter(1.0, 1, 2)
    s2 = Scatter(2.0, 1, 2)
    s3 = Scatter(1.0, 2, 1)

    @test s1 == s2
    @test s1 != s3
    @test isless(s3, s1)

    @test (s1 + s2).Amp == 3.0
    @test (2 * s1).Amp == 2.0
end
