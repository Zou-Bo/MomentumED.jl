"""
Many-body Berry connection calculation functions for momentum-conserved exact diagonalization.

This file implements the calculation of Berry connections and geometric phases between
different kshift points in momentum space, which is essential for computing topological
invariants like Chern numbers.
"""


using LinearAlgebra

"""
    validate_berry_connection_inputs(kshift1, kshift2, ψ1, ψ2)

Validate inputs for berry_connection function.
Throws ArgumentError for invalid inputs.
"""
function validate_berry_connection_inputs(kshift1, kshift2, ψ1, ψ2)
    # Check vector dimensions match
    length(ψ1) == length(ψ2) || 
        throw(ArgumentError("Ground state vectors must have same dimension"))
    
    # Check kshift dimensions
    length(kshift1) == 2 && length(kshift2) == 2 ||
        throw(ArgumentError("kshift must be 2D tuples"))
    
    # Check for zero vectors
    norm(ψ1) > 1e-12 && norm(ψ2) > 1e-12 ||
        throw(ArgumentError("Ground state vectors cannot be zero"))
    
    return true
end

"""
    ED_connection_integral(kshift1, kshift2, ψ1, ψ2, momentum_axis_angle; average_connection=false)

Compute the many-body Berry connection integral between two kshift points in momentum space.

# Arguments
- `kshift1::Tuple{Float64, Float64}`: First momentum shift point (kx1, ky1)
- `kshift2::Tuple{Float64, Float64}`: Second momentum shift point (kx2, ky2)  
- `ψ1::AbstractVector{<:Complex}`: Ground state eigenvector at kshift1
- `ψ2::AbstractVector{<:Complex}`: Ground state eigenvector at kshift2
- `momentum_axis_angle::Float64`: Angle for momentum space geometry correction
- `average_connection::Bool`: Whether to divide by momentum space distance (default: false)

# Returns
- `Float64`: Geometric phase φ = arg(⟨ψ2|ψ1⟩) if average_connection=false
              Berry connection A = φ/||δk|| if average_connection=true

# Mathematical Framework
Berry connection: A = ⟨ψ|i∂_k|ψ⟩ ≈ arg(⟨ψ(k+δk)|ψ(k)⟩) / ||δk||
For small δk = kshift2 - kshift1, inner product ⟨ψ2|ψ1⟩ ≈ exp(i * A * ||δk||)
Geometric phase: φ = arg(⟨ψ2|ψ1⟩)

When average_connection=true, returns the Berry connection (phase divided by distance).
When average_connection=false, returns just the geometric phase.

# Examples
```julia
# Calculate geometric phase between two close kshift points
kshift1 = (0.0, 0.0)
kshift2 = (0.01, 0.0)
ψ1 = ground_state_at_kshift1  # Complex vector
ψ2 = ground_state_at_kshift2  # Complex vector
angle = π/2  # Orthogonal momentum axes

phase = ED_connection_integral(kshift1, kshift2, ψ1, ψ2, angle)
println("Geometric phase: ", phase)

# Calculate Berry connection (average connection)
berry_conn = ED_connection_integral(kshift1, kshift2, ψ1, ψ2, angle; average_connection=true)
println("Berry connection: ", berry_conn)
```
"""
function ED_connection_integral(kshift1, kshift2, ψ1, ψ2, momentum_axis_angle; average_connection=false)
    # Validate inputs
    validate_berry_connection_inputs(kshift1, kshift2, ψ1, ψ2)
    
    # Compute complex inner product ⟨ψ2|ψ1⟩ using Julia's built-in dot
    inner_prod = dot(ψ2, ψ1)
    
    # Extract geometric phase using angle() function (equivalent to arg)
    geometric_phase = angle(inner_prod)
    
    if average_connection
        # Calculate momentum space distance accounting for non-orthogonal coordinates
        δk = kshift2 .- kshift1
        # For non-orthogonal momentum space with angle θ between axes:
        # d² = δk₁² + δk₂² + 2*δk₁*δk₂*cos(θ) 
        # This is the proper metric tensor approach
        distance = sqrt(δk[1]^2 + δk[2]^2 + 2*δk[1]*δk[2]*cos(momentum_axis_angle))
        
        # Handle numerical stability for very close points
        if distance < 1e-12
            @warn "kshift points very close, numerical instability possible"
            distance = 1e-12
        end
        
        # Berry connection A = φ / ||δk||
        berry_conn = geometric_phase / distance
        return berry_conn
    else
        # Return just the geometric phase
        return geometric_phase
    end
end