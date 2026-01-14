using Test
using EDCore

@testset "ColexMBS64" begin
    # Test basic functionality for n=4, t=2
    c = ColexMBS64(4, 2)
    mbs_list = collect(c)
    
    # Expected combinations in colex order: (1,2), (1,3), (2,3), (1,4), (2,4), (3,4)
    expected_mbs = [
        MBS64(4, (1, 2)),
        MBS64(4, (1, 3)),
        MBS64(4, (2, 3)),
        MBS64(4, (1, 4)),
        MBS64(4, (2, 4)),
        MBS64(4, (3, 4)),
    ]
    @test length(mbs_list) == binomial(4, 2)
    @test mbs_list == expected_mbs

    # Test edge case t=0
    c_t0 = ColexMBS64(3, 0)
    @test collect(c_t0) == [MBS64(3, Int64[])] # Empty set combination
    @test length(collect(c_t0)) == binomial(3, 0)

    # Test edge case t=n
    c_tn = ColexMBS64(3, 3)
    @test collect(c_tn) == [MBS64(3, (1, 2, 3))]
    @test length(collect(c_tn)) == binomial(3, 3)

    # Test edge case n=0, t=0
    c_n0t0 = ColexMBS64(0, 0)
    @test collect(c_n0t0) == [MBS64(0, Int64[])]
    @test length(collect(c_n0t0)) == binomial(0, 0)

    # Test larger case
    c_large = ColexMBS64(5, 3)
    mbs_large_list = collect(c_large)
    @test length(mbs_large_list) == binomial(5, 3)
    @test issorted(mbs_large_list) # Check colexicographical order
end

@testset "ColexMBS64Mask" begin
    # Test basic functionality with mask
    mask = [1, 3, 5] # Corresponds to actual orbitals 1, 3, 5
    c_mask = ColexMBS64Mask(5, 2, mask) # n=5 (total bits), t=2 (occupied), mask=[1,3,5]
    mbs_list_mask = collect(c_mask)

    # Expected combinations (indices within mask): (1,2) -> (1,3), (1,3) -> (1,5), (2,3) -> (3,5)
    expected_mbs_mask = [
        MBS64(5, (1, 3)),
        MBS64(5, (1, 5)),
        MBS64(5, (3, 5)),
    ]
    @test length(mbs_list_mask) == binomial(length(mask), 2)
    @test mbs_list_mask == expected_mbs_mask

    # Test edge case t=0 with mask
    c_mask_t0 = ColexMBS64Mask(3, 0, mask)
    @test collect(c_mask_t0) == [MBS64(3, Int64[])]
    @test length(collect(c_mask_t0)) == binomial(length(mask), 0)

    # Test edge case t=length(mask)
    mask2 = [2, 4]
    c_mask_tn = ColexMBS64Mask(5, 2, mask2)
    @test collect(c_mask_tn) == [MBS64(5, (2, 4))]
    @test length(collect(c_mask_tn)) == binomial(length(mask2), 2)

    # Test larger mask and combinations
    mask_large = [1, 2, 3, 4, 5, 6]
    c_mask_large = ColexMBS64Mask(10, 3, mask_large)
    mbs_large_list_mask = collect(c_mask_large)
    @test length(mbs_large_list_mask) == binomial(length(mask_large), 3)
    @test issorted(mbs_large_list_mask) # Check colexicographical order
end