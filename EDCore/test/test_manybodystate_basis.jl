
@testset "MBS64" begin
    @test MBS64{8}(UInt(10)).n == 10
    @test get_bits(MBS64{8}(UInt(10))) == 8
    @test isphysical(MBS64{8}(UInt(10)))
    @test !isphysical(reinterpret(MBS64{8}, UInt(256)))
end

@testset "MBS64 Operations" begin
    mbs1 = MBS64{4}(UInt(5)) # 0101
    mbs2 = MBS64{4}(UInt(10)) # 1010
    @test (mbs1 + mbs2) == MBS64{4}(UInt(15)) # 1111

    mbs3 = MBS64{2}(UInt(1)) # 01
    mbs4 = MBS64{3}(UInt(2)) # 010
    @test (mbs3 * mbs4) == MBS64{5}(UInt(10)) #01010

    @test MBS64{8}(UInt(10)) == MBS64{8}(UInt(10))
    @test MBS64{8}(UInt(10)) != MBS64{8}(UInt(11))
    @test isless(MBS64{8}(UInt(10)), MBS64{8}(UInt(11)))
end

@testset "MBS64 Manipulation" begin
    mbs = MBS64{8}(UInt(10)) # 00001010
    @test occ_list(mbs) == [2, 4]
    @test make_mask64([2, 4]) == UInt(10)
    @test MBS64(8, [2, 4]) == mbs

    @test isoccupied(mbs, [2, 4])
    @test !isoccupied(mbs, [1, 2])
    @test isempty(mbs, [1, 3])
    @test !isempty(mbs, [2, 3])

    @test occupy!(MBS64{8}(UInt(0)), [2, 4]) == mbs
    @test empty!(mbs, [2, 4]) == MBS64{8}(UInt(0))

    @test scat_occ_number(mbs, (2, 4)) == 0
    @test scat_occ_number(mbs, (1, 5)) == 2
end

@testset "MBS64 Flip" begin
    mbs = MBS64{8}(UInt(10)) # 00001010
    @test flip!(mbs, [2, 4]) == MBS64{8}(UInt(0))
    @test flip!(mbs, [1, 5]) == MBS64{8}(UInt(27)) # 00011011
    @test flip!(MBS64{8}(UInt(0)), [2, 4]) == mbs
end
