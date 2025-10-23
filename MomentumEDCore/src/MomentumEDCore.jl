module MomentumEDCore

# Include utilities
include("types/manybodystate_basis.jl")
include("types/hilbert_subspace.jl")
include("types/manybodystate_vector.jl")

export MBS64, HilbertSubspace, MBS64Vector
export get_bits, isphysical, occ_list, make_mask64, MBS64_complete
export isoccupied, isempty, occupy!, empty, scat_occ_number
export idtype, make_dict!, delete_dict!, get_from_list, get_from_dict

# Include utilities
include("types/scattering.jl")
include("types/operator.jl")
include("types/multiplication.jl")

export Scatter, MBOperator
export NormalScatter, get_body, isnormal, isnormalupper, isdiagonal, sort_merge_scatlist
export isupper, ED_bracket, ED_bracket_threaded # , multiplication_threaded

# Using these to force precompile multiplications
Scatter{1}(0.1, (1,), (2,)) * reinterpret(MBS64{3}, UInt64(2))
Scatter{2}(0.1, (1,2), (2,3)) * reinterpret(MBS64{3}, UInt64(6))
Scatter{3}(0.1, (1,2,3), (1,2,3)) * reinterpret(MBS64{3}, UInt64(7))
reinterpret(MBS64{3}, UInt64(1)) * Scatter{1}(0.1, (1,), (2,))
reinterpret(MBS64{3}, UInt64(3)) * Scatter{2}(0.1, (1,2), (2,3))
reinterpret(MBS64{3}, UInt64(7)) * Scatter{3}(0.1, (1,2,3), (1,2,3))



end