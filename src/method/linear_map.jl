
include("search.jl")


function mul_add!(vec_out::Vector{Complex{F}}, 
    scat::Scattering, vec_in::Vector{Complex{F}},
    sorted_mbs_block_list::Vector{MBS64{bits}}) where {bits, F <: Real}

    for (j, mbs_in) in enumerate(sorted_mbs_block_list)
        amp, mbs_out = scat * mbs_in
        i = my_searchsortedfirst(sorted_mbs_block_list, mbs_out)
        vec_out[i] += amp * vec_in[j]
    end

end

function Hmlt_map(sorted_mbs_block_list::Vector{<: MBS64}, 
    sorted_scat_lists::Vector{<: Scattering}...;
    element_type::Type = Float64,
)::Function

    @assert element_type ∈ (Float64, Float32, Float16) "Use element_type Float64, Float32, or Float16."
    
    function H_threaded(vec_in::Vector{Complex{element_type}})::Vector{Complex{element_type}}
        n_states = length(sorted_mbs_block_list)
        
        
        # # Thread-local storage for COO format
        # n_threads = Threads.nthreads()
        # thread_I = [Vector{index_type}() for _ in 1:n_threads]
        # thread_J = [Vector{index_type}() for _ in 1:n_threads]
        # thread_V = [Vector{Complex{element_type}}() for _ in 1:n_threads]
        
        # # Parallel construction over columns
        # Threads.@threads for j in 1:n_states
        #     tid = Threads.threadid()
        #     mbs_in = sorted_mbs_block_list[j]
        # end
    end

    return H_threaded
end
