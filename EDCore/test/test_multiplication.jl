@testset "Scatter * MBS64" begin
    s = Scatter(2.0, 1, 2, bits=4) # 2.0 * c†_1 c_2
    mbs_in = MBS64(4, (2,)) # |0010>
    amp, mbs_out = s * mbs_in
    @test amp == 2.0
    @test mbs_out == MBS64(4, (1,)) # |0001>

    # Test sign change
    s_sign = Scatter(1.0, 2, 1, bits=4) # c†_2 c_1
    mbs_in_sign = MBS64(4, (1, 3)) # |0101>
    amp_sign, mbs_out_sign = s_sign * mbs_in_sign
    @test amp_sign == 1.0
    @test mbs_out_sign == MBS64(4, (2, 3)) # |0110>
end

@testset "MBOperator * MBS64Vector" begin
    s1 = Scatter(1.0, 1, 2, bits=2) # c†_1 c_2
    s2 = Scatter(1.0, 2, 1, bits=2) # c†_2 c_1
    op = MBOperator([s1], [s2]; upper_hermitian=false)

    list = [MBS64(2, (1,)), MBS64(2, (2,))] # |01>, |10>
    space = HilbertSubspace(list; dict=true)
    
    # Input |01>
    vec_in_val = zeros(ComplexF64, length(space))
    vec_in_val[space.dict[MBS64(2, (1,))]] = 1.0
    vec_in = MBS64Vector(vec_in_val, space)
    vec_out = op * vec_in
    @test vec_out.vec[space.dict[MBS64(2, (2,))]] ≈ 1.0

    # Input |10>
    vec_in_val = zeros(ComplexF64, length(space))
    vec_in_val[space.dict[MBS64(2, (2,))]] = 1.0
    vec_in = MBS64Vector(vec_in_val, space)
    vec_out = op * vec_in
    @test vec_out.vec[space.dict[MBS64(2, (1,))]] ≈ 1.0
end

@testset "MBOperator * MBS64Vector (upper_hermitian)" begin
    s1 = Scatter(1.0 + 1.0im, 1, 2, bits=2, upper_hermitian=true) # c†_1 c_2
    op = MBOperator([s1]; upper_hermitian=true)

    list = [MBS64(2, (1,)), MBS64(2, (2,))] # |01>, |10>
    space = HilbertSubspace(list; dict=true)
    
    # Input |01>
    vec_in_val = zeros(ComplexF64, length(space))
    vec_in_val[space.dict[MBS64(2, (1,))]] = 1.0
    vec_in = MBS64Vector(vec_in_val, space)
    vec_out = op * vec_in
    @test vec_out.vec[space.dict[MBS64(2, (1,))]] ≈ 0.0
    @test vec_out.vec[space.dict[MBS64(2, (2,))]] ≈ 1.0 - 1.0im

    # Input |10>
    vec_in_val = zeros(ComplexF64, length(space))
    vec_in_val[space.dict[MBS64(2, (2,))]] = 1.0
    vec_in = MBS64Vector(vec_in_val, space)
    vec_out = op * vec_in
    @test vec_out.vec[space.dict[MBS64(2, (1,))]] ≈ 1.0 + 1.0im
    @test vec_out.vec[space.dict[MBS64(2, (2,))]] ≈ 0.0
end

@testset "ED_bracket" begin
    s1 = Scatter(1.0, 2, 1, bits=2) # c†_2 c_1
    op = MBOperator([s1]; upper_hermitian=false)

    list = [MBS64(2, (1,)), MBS64(2, (2,))] # |01>, |10>
    space = HilbertSubspace(list; dict=true)

    bra_val = zeros(ComplexF64, length(space))
    bra_val[space.dict[MBS64(2, (2,))]] = 1.0
    bra = MBS64Vector(bra_val, space) # <10|

    ket_val = zeros(ComplexF64, length(space))
    ket_val[space.dict[MBS64(2, (1,))]] = 1.0
    ket = MBS64Vector(ket_val, space) # |01>

    @test ED_bracket(bra, op, ket) ≈ 1.0
end