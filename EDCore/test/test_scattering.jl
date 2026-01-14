@testset "Scatter" begin
    s1 = Scatter(1.0, 1, 2, bits=10)
    @test s1.Amp == 1.0
    @test s1.out == MBS64(10, (1,))
    @test s1.in == MBS64(10, (2,))
    @test get_body(s1) == (1, 1)

    s2 = Scatter(0.5, 1, 2, 4, 3, bits=10)
    @test s2.Amp == 0.5
    @test s2.out == MBS64(10, (1, 2))
    @test s2.in == MBS64(10, (3, 4))
    @test get_body(s2) == (2, 2)

    @test adjoint(s1) == Scatter(1.0, 2, 1, bits=10)
    @test isdiagonal(Scatter(1.0, 1, 1, bits=10))
    @test !isdiagonal(s1)
end

@testset "Scatter with normal ordering" begin
    # N=1
    s = Scatter(1.0 + 1.0im, 2, 1; bits=10, upper_hermitian=true)
    @test s.in == MBS64(10, (2,))
    @test s.out == MBS64(10, (1,))
    @test s.Amp == 1.0 - 1.0im

    # N=2
    s = Scatter(1.0 + 0.0im, 2, 1, 3, 4; bits=10, upper_hermitian=true)
    @test s.in == MBS64(10, (4,3))
    @test s.out == MBS64(10, (2,1))
    @test s.Amp == 1.0

    s = Scatter(1.0 + 0.0im, 1, 2, 3, 4; bits=10, upper_hermitian=true)
    @test s.in == MBS64(10, (4,3))
    @test s.out == MBS64(10, (2,1))
    @test s.Amp == -1.0
end

@testset "Scatter Operations" begin
    s1 = Scatter(1.0, 1, 2, bits=10)
    s2 = Scatter(2.0, 1, 2, bits=10)
    s3 = Scatter(1.0, 2, 1, bits=10)

    @test s1 == s2
    @test s1 != s3
    @test isless(s3, s1)

    @test (s1 + s2).Amp == 3.0
    @test (2 * s1).Amp == 2.0
end

@testset "sort_merge_scatlist" begin
    s1 = Scatter(1.0, 1, 2, bits=10)
    s2 = Scatter(2.0, 1, 2, bits=10)
    s3 = Scatter(1.0, 2, 1, bits=10)
    s4 = Scatter(1.0, 1, 1, bits=10, upper_hermitian=true)

    # Test merging
    merged_list = sort_merge_scatlist([s1, s2]; check_upper=false)
    @test length(merged_list) == 1
    @test merged_list[1].Amp == 3.0

    # Test sorting
    merged_list = sort_merge_scatlist([s1, s3]; check_upper=false)
    @test length(merged_list) == 2
    @test isless(merged_list[1], merged_list[2])

    # Test check_upper
    @test_throws AssertionError sort_merge_scatlist([s3]; check_upper=true)
    @test sort_merge_scatlist([s4]; check_upper=true) == [s4]
end