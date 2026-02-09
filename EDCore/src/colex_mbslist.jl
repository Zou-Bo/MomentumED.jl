# to-do: allow starting and ending positions for ColexMBS64Mask

# directly find the k-th state in colex(n, t) without iteration
function unrank_colex(n::Int64, t::Int64, k::Int64)::UInt64
    t == 0 && return UInt64(0)
    k <= 0 && return UInt64(0)
    k > binomial(n, t) && return UInt64(0)
    
    result = UInt64(0)
    x = k - 1
    
    for i in t:-1:1
        c = i - 1
        while c + 1 < n && binomial(c + 1, i) <= x
            c += 1
        end
        
        result |= (UInt64(1) << c)
        x -= binomial(c, i)
    end
    
    return result
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
    mbs = MBS64{c.n}(unrank_colex(c.n, c.t, c.start_number))
    return (mbs, [occ_list(mbs); c.n+1; c.start_number + 1])
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
    mask::Vector{Int64}
end

# Iteration
@inline function Base.iterate(c::ColexMBS64Mask) # starting point
    try
        @assert c.t >= 0
        MBS64(c.n, 1:c.t, c.mask)
    catch
        # @info "iteration $c is empty."
        return
    end
    return (MBS64(c.n, 1:c.t, c.mask), [collect(1:c.t); length(c.mask)+1])
end
@inline function Base.iterate(c::ColexMBS64Mask, s)
    if c.t == 0
        return
    end
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
Base.length(c::ColexMBS64Mask) = binomial(length(c.mask), c.t)

Base.eltype(c::ColexMBS64) = MBS64{c.n}
Base.eltype(c::ColexMBS64Mask) = MBS64{c.n}