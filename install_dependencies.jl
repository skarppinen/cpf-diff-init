# Script to load all package dependencies specified by Manifest.jl.
import Pkg
Pkg.activate(".");
Pkg.instantiate();
