
@testset "HilbertSubspace" begin
    sorted_list = [MBS64{8}(UInt(i)) for i in 0:4]
    space_no_dict = HilbertSubspace(sorted_list, dict=false)
    space_with_dict = HilbertSubspace(sorted_list, dict=true)

    @test length(space_no_dict) == 5
    @test get_bits(space_no_dict) == 8
    @test idtype(space_with_dict) == Int

    @test get(space_no_dict, MBS64{8}(UInt(2))) == 3
    @test get(space_with_dict, MBS64{8}(UInt(2))) == 3
    @test get(space_no_dict, MBS64{8}(UInt(5))) == 0

    make_dict!(space_no_dict)
    @test !isempty(space_no_dict.dict)
    @test get(space_no_dict, MBS64{8}(UInt(2))) == 3

    delete_dict!(space_no_dict)
    @test isempty(space_no_dict.dict)
end
