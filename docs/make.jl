# Setup local package dependency first
include("setup.jl")

using Documenter
using EDCore
using MomentumED
using LinearAlgebra

makedocs(;
    modules=[EDCore, MomentumED],
    authors="Zou Bo <zou.bo.phys@gmail.com>",
    repo="https://github.com/Zou-Bo/MomentumED.jl/blob/{commit}{path}#{line}",
    sitename="MomentumED.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Zou-Bo.github.io/MomentumED.jl",
        assets=String[],
    ),
    doctest = true,
    pages=[
        "Home" => "index.md",
        "EDCore" => [
            # "Tutorial" => "EDCore/tutorial.md",
            # "Manual" => "EDCore/manual.md",
            "API Reference" => "EDCore/api.md",
        ],
        "MomentumED" => [
            # "Tutorial" => "MomentumED/tutorial.md",
            # "Manual" => "MomentumED/manual.md",
            "API Reference" => "MomentumED/api.md",
        ],
    ],
    warnonly=true,  # Allow build to continue despite warnings
)

deploydocs(;
    repo="github.com/Zou-Bo/MomentumED.jl",
    devbranch="main",
    versions = ["stable" => "v^", "v#.#"],
)