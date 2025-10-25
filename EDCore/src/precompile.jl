# Precompilation script for EDCore

# The calls here are used to trigger precompilation for common method signatures.
# This helps reduce the time-to-first-plot (TTFP) latency.

# multiplication
Scatter{1}(0.1, (1,), (2,)) * MBS64{3}(UInt64(2))
Scatter{2}(0.1, (1,2), (2,3)) * MBS64{3}(UInt64(6))
Scatter{3}(0.1, (1,2,3), (1,2,3)) * MBS64{3}(UInt64(7))
MBS64{3}(UInt64(1)) * Scatter{1}(0.1, (1,), (2,))
MBS64{3}(UInt64(3)) * Scatter{2}(0.1, (1,2), (2,3))
MBS64{3}(UInt64(7)) * Scatter{3}(0.1, (1,2,3), (1,2,3))
