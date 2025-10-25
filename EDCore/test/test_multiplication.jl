
@testset "Scatter * MBS64" begin
    s = Scatter(2.0, 1, 2) # 2.0 * c†_1 c_2
    mbs_in = MBS64{4}(UInt(2)) # |0010>
    amp, mbs_out = s * mbs_in
    @test amp == 2.0
    @test mbs_out == MBS64{4}(UInt(1)) # |0001>

    # Test sign change
    s_sign = Scatter(1.0, 2, 1) # c†_2 c_1
    mbs_in_sign = MBS64{4}(UInt(5)) # |0101>
    amp_sign, mbs_out_sign = s_sign * mbs_in_sign
    @test amp_sign == 1.0
    @test mbs_out_sign == MBS64{4}(UInt(6)) # |0110>
end

@testset "MBOperator * MBS64Vector" begin
    s1 = Scatter(1.0, 1, 2) # c†_1 c_2
    s2 = Scatter(1.0, 2, 1) # c†_2 c_1
    op = MBOperator([s1, s2]; upper_hermitian=false)

    list = [MBS64{2}(UInt(1)), MBS64{2}(UInt(2))] # |01>, |10>
    space = HilbertSubspace(list, dict=true)
    
    # Input |01>
    vec_in = MBS64Vector(ComplexF64[1.0, 0.0], space)
    vec_out = op * vec_in
    @test vec_out.vec ≈ [0.0, 1.0]

    # Input |10>
    vec_in = MBS64Vector(ComplexF64[0.0, 1.0], space)
    vec_out = op * vec_in
    @test vec_out.vec ≈ [1.0, 0.0]
end

@testset "ED_bracket" begin
    s1 = Scatter(1.0, 2, 1) # c†_2 c_1
    op = MBOperator([s1]; upper_hermitian=false)

    list = [MBS64{2}(UInt(1)), MBS64{2}(UInt(2))] # |01>, |10>
    space = HilbertSubspace(list, dict=true)

    bra = MBS64Vector(ComplexF64[0.0, 1.0], space) # <10|
    ket = MBS64Vector(ComplexF64[1.0, 0.0], space) # |01>

    @test ED_bracket(bra, op, ket) ≈ 1.0
end
