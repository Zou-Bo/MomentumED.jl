
@testset "HilbertSubspace" begin
    sorted_list = [MBS64{8}(UInt(i)) for i in 0:4]
    space = HilbertSubspace(sorted_list)

    @test length(space) == 5
    @test get_bits(space) == 8

    @test get(space, MBS64{8}(UInt(2))) == 3
    @test !index_fit(get(space, MBS64{8}(UInt(5))), space, MBS64{8}(UInt(5)))

end
