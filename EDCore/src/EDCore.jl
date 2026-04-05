module EDCore

export MBS64, HilbertSubspace, MBS64Vector
export get_bits, isphysical, occ_list, for_each_occ, make_mask64, MBS64_complete
export isoccupied, isempty, occupy!, empty!, flip!, scat_occ_number, index_fit

export Scatter, MBOperator
export get_body, isupper, isdiagonal, sort_merge_scatlist

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