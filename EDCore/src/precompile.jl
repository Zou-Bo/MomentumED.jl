# Precompilation script for EDCore

# The calls here are used to trigger precompilation for common method signatures.
# This helps reduce the time-to-first-plot (TTFP) latency.

# normal scatter
Scatter(ComplexF64(1.0), 3, 1; bits = 10)
Scatter(ComplexF64(1.0), 1, 2, 3, 4; bits = 10)
Scatter(ComplexF64(1.0), 3, 1, 2, 4, 5, 6; bits = 10)

# multiplication
Scatter(0.1, 1, 2; bits = 10) * MBS64{10}(UInt64(2))
Scatter(0.1, 1, 2, 2, 3; bits = 10) * MBS64{10}(UInt64(6))
Scatter(0.1, 1, 2, 3, 1, 2, 3; bits = 10) * MBS64{10}(UInt64(7))
MBS64{10}(UInt64(1)) * Scatter(0.1, 1, 2; bits = 10)
MBS64{10}(UInt64(3)) * Scatter(0.1, 1, 2, 2, 3; bits = 10)
MBS64{10}(UInt64(7)) * Scatter(0.1, 1, 2, 3, 1, 2, 3; bits = 10)
