# directly find the k-th state in colex(n, t) without iteration
function unrank_colex(n::Int64, t::Int64, k::Int64)::Vector{Int64}
    t == 0 && return Int64[]
    k <= 0 && return Int64[]
    k > binomial(n, t) && return Int64[]
    
    result = Int64[]
    x = k - 1
    
    for i in t:-1:1
        c = i - 1
        while c + 1 < n && binomial(c + 1, i) <= x
            c += 1
        end
        
        push!(result, c+1)
        x -= binomial(c, i)
    end
    
    return sort!(result)
end

"""
The Combinations iterator in colex order (meaning sorted MBS64 list)

```julia
for mbs in ColexMBS64(7, 3)
    println(mbs)
end
```
"""
struct ColexMBS64
    n::Int64
    t::Int64

    start_number::Int64
    end_number::Int64

    function ColexMBS64(n::Int64, t::Int64)
        @assert 0 <= n <= 64 

        # when n >= 0 , smallest binomial(n, t) is 0 
        # if binomial(n, t)==0, return empty iterator
        end_number = binomial(n, t)
        new(n, t, 1, end_number)
    end

    function ColexMBS64(n::Int64, t::Int64, start_number::Int64, end_number::Int64)
        @assert 0 <= n <= 64
        @assert start_number > 0

        full_length = binomial(n, t)
        if start_number > full_length || end_number < start_number
            # return empty iterator
            return new(n, t, start_number, start_number - 1)
        end
        if end_number > full_length
            end_number = full_length
        end
        new(n, t, start_number, end_number)
    end
end

# Iteration
@inline function Base.iterate(c::ColexMBS64) # starting point
    if c.start_number > c.end_number
        # @info "iteration $c is empty."
        return
    end
    if c.start_number == 1
        return (MBS64(c.n, 1:c.t), [collect(1:c.t); c.n+1; c.start_number + 1])
    end
    occlist = unrank_colex(c.n, c.t, c.start_number)
    return (MBS64(c.n, occlist), [occlist; c.n+1; c.start_number + 1])
end
@inline function Base.iterate(c::ColexMBS64, s)
    if s[end] > c.end_number
        return
    end
    s[end] += 1
    for i in 1:c.t
        if s[i] < s[i+1] -1
            s[i] += 1
            for j in 1:i-1
                s[j] = j
            end
            return (MBS64(c.n, view(s, 1:c.t)), s)
        end
    end
    return
end

"""
The Combinations iterator in colex order where occupied orbitals are in the mask.
Returning list is sorted when the mask is sorted.

```julia
for mbs in ColexMBS64Mask(7, 3, [1; 2; 5; 6; 7])
    println(mbs)
end
```
"""
struct ColexMBS64Mask
    n::Int
    t::Int
    mask::Vector{Int64}  # should be sorted

    start_number::Int64
    end_number::Int64

    function ColexMBS64Mask(n::Int64, t::Int64, mask::Vector{Int64})
        @assert 0 <= n <= 64
        @assert issorted(mask) && allunique(mask)
        len = length(mask)
        if len > 0
            @assert 0 <= mask[begin] && mask[end] <= n
        end

        # when len >= 0 , smallest binomial(len, t) is 0 
        # if binomial(len, t)==0, return empty iterator
        end_number = binomial(len, t)
        new(n, t, mask, 1, end_number)
    end

    function ColexMBS64Mask(n::Int64, t::Int64, mask::Vector{Int64}, start_number::Int64, end_number::Int64)
        @assert 0 <= n <= 64
        @assert issorted(mask) && allunique(mask)
        len = length(mask)
        if len > 0
            @assert 0 <= mask[begin] && mask[end] <= n
        end
        @assert start_number > 0

        full_length = binomial(len, t)
        if start_number > full_length || end_number < start_number
            # return empty iterator
            return new(n, t, mask, start_number, start_number - 1)
        end
        if end_number > full_length
            end_number = full_length
        end
        new(n, t, mask, start_number, end_number)
    end
end

# Iteration
@inline function Base.iterate(c::ColexMBS64Mask) # starting point
    if c.start_number > c.end_number
        # @info "iteration $c is empty."
        return
    end
    if c.start_number == 1
        return (MBS64(c.n, 1:c.t, c.mask), [collect(1:c.t); length(c.mask)+1; c.start_number + 1])
    end
    occlist = unrank_colex(length(c.mask), c.t, c.start_number)
    return (MBS64(c.n, occlist, c.mask), [occlist; length(c.mask)+1; c.start_number + 1])
end
@inline function Base.iterate(c::ColexMBS64Mask, s)
    if s[end] > c.end_number
        return
    end
    s[end] += 1
    for i in 1:c.t
        if s[i] < s[i+1] -1
            s[i] += 1
            for j in 1:i-1
                s[j] = j
            end
            return (MBS64(c.n, view(s, 1:c.t), c.mask), s)
        end
    end
    return
end

Base.length(c::ColexMBS64) = c.end_number - c.start_number + 1
Base.length(c::ColexMBS64Mask) = c.end_number - c.start_number + 1

Base.eltype(c::ColexMBS64) = MBS64{c.n}
Base.eltype(c::ColexMBS64Mask) = MBS64{c.n}