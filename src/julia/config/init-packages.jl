# Script to install all relevant packages.
import Pkg;
Pkg.add("")
Pkg.add("ArgParse");
Pkg.add("DataFrames");
Pkg.add("DataStructures");
Pkg.add("Random");
Pkg.add("LabelledArrays");
Pkg.add("LinearAlgebra");
Pkg.add("Distributions");
Pkg.add("StaticArrays");
Pkg.add("Dates");
Pkg.add("JLD2");
Pkg.add("ArgParse");
Pkg.add("Statistics");
Pkg.add("Optim");
Pkg.add("RCall");
Pkg.add("StatsBase");
Pkg.add("CSV");
Pkg.add("InfoZIP");
Pkg.add(PackageSpec(url="https://github.com/mvihola/AdaptiveParticleMCMC.jl.git"))
