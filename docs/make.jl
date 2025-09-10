# Setup local package dependency first
include("setup.jl")

using Documenter
using MomentumED

makedocs(;
    modules=[MomentumED],
    authors="Zou Bo <zoubo.physics@gmail.com>",
    repo="https://github.com/Zou-Bo/MomentumED.jl/blob/{commit}{path}#{line}",
    sitename="MomentumED.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Zou-Bo.github.io/MomentumED.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Physics Background" => "physics.md",
        "Examples" => "examples.md",
        "Performance Guide" => "performance.md",
    ],
    warnonly=true,  # Allow build to continue despite warnings
)

deploydocs(;
    repo="github.com/Zou-Bo/MomentumED.jl",
    devbranch=nothing,
    versions = ["stable" => "v^", "v#.#"],
)