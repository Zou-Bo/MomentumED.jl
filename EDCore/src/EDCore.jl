module EDCore

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
export NormalScatter, get_body, isnormal, isnormalupper, isdiagonal
export sort_merge_scatlist, isupper, ED_bracket, ED_bracket_threaded

if ccall(:jl_generating_output, Cint, ()) == 1
    include("precompile.jl")
end

end