# Setup script for documentation build
# This handles the local package dependency for unregistered packages

using Pkg

# Activate the docs project environment first
Pkg.activate(@__DIR__)

# Define the paths to the local packages
pathToEDCore = joinpath(@__DIR__, "..", "EDCore")
pathToMomentumED = joinpath(@__DIR__, "..", "MomentumED")

# Add the local packages using Pkg.develop
# This allows Documenter to find the modules and their docstrings
Pkg.develop([
    Pkg.PackageSpec(path=pathToEDCore),
    Pkg.PackageSpec(path=pathToMomentumED)
])
# Instantiate the docs project dependencies
Pkg.instantiate()