module EDCore

export MBS64, HilbertSubspace, MBS64Vector
export get_bits, isphysical, occ_list, make_mask64, MBS64_complete
export isoccupied, isempty, occupy!, empty!, flip!, scat_occ_number
export idtype, make_dict!, delete_dict!, get_from_list, get_from_dict

export Scatter, MBOperator
export NormalScatter, get_body, isnormal, isnormalupper, isdiagonal
export sort_merge_scatlist, isupper

export ED_bracket, ED_bracket_threaded
export ColexMBS64, ColexMBS64Mask

# Include utilities
include("types/manybodystate_basis.jl")
include("types/hilbert_subspace.jl")
include("types/manybodystate_vector.jl")

# Include utilities
include("types/scattering.jl")
include("types/operator.jl")

# Include utilities
include("multiplication.jl")
include("colex_mbslist.jl")

if ccall(:jl_generating_output, Cint, ()) == 1
    include("precompile.jl")
end

end