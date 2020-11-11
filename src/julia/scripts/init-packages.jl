## Script to generate Manifest.jl and Project.jl.
# Run from project root.

using Pkg
Pkg.activate(".");

# Add packages.
packages = ["Pkg",
            "ArgParse",
            "DataFrames",
            "DataStructures",
            "Random",
            "LabelledArrays",
            "LinearAlgebra",
            "Distributions",
            "StaticArrays",
            "Dates",
            "JLD2",
            "ArgParse",
            "Statistics",
            "RCall",
            "StatsBase",
            "CSV",
            "InfoZIP"];
map(x -> Pkg.add(x), packages);
Pkg.add(PackageSpec(url="https://github.com/mvihola/AdaptiveParticleMCMC.jl.git"));

# Pin packages.
stdlibs = ["Pkg", "Random", "LinearAlgebra", "Dates", "Statistics"];
Pkg.pin(setdiff(packages, stdlibs));
Pkg.pin("AdaptiveParticleMCMC");
