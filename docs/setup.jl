# Setup script for documentation build
# This handles the local package dependency for unregistered packages

using Pkg

# Activate the docs project environment first
Pkg.activate(@__DIR__)

# Add the parent directory as a local package dependency
# This allows Documenter to access the MomentumED module
Pkg.develop(path=joinpath(@__DIR__, ".."))

# Instantiate the docs project dependencies
Pkg.instantiate()