
"""
The Combinations iterator in colex order (meaning sorted MBS64 list)

```julia
for mbs in ColexMBS64(7, 3)
    println(mbs)
end
```
"""
struct ColexMBS64
    n::Int
    t::Int
end

# Iteration
@inline function Base.iterate(c::ColexMBS64) # starting point
    try
        @assert c.t >= 0
        MBS64(c.n, 1:c.t)
    catch
        # @info "iteration $c is empty."
        return
    end
    return (MBS64(c.n, 1:c.t), [collect(1:c.t); c.n+1])
end
@inline function Base.iterate(c::ColexMBS64, s)
    if c.t == 0
        return
    end
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

Base.length(c::ColexMBS64) = binomial(c.n, c.t)
Base.length(c::ColexMBS64Mask) = binomial(length(c.mask), c.t)

Base.eltype(c::ColexMBS64) = MBS64{c.n}
Base.eltype(c::ColexMBS64Mask) = MBS64{c.n}