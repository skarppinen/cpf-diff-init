## Definition of the SV model.
include("../../lib/pfilter.jl");
using Random
using Distributions

## Particle type and required functions for model.
mutable struct SVParticle <: Particle
    x::Float64
end
function SVParticle()
    SVParticle(NaN);
end
import Base.copy!
function copy!(dest::SVParticle, src::SVParticle)
    dest.x = src.x;
    dest;
end
function SVector{1, Float64}(p::SVParticle)
    SVector{1, Float64}(p.x);
end

## Model definition.
SV = let
    function Mi!(p::SVParticle, data, θ)
        p.x = rand(Normal(θ[:x1_mean], exp(θ[:log_sigma_x1])));
        nothing
    end
    function lMi(p::SVParticle, data, θ)
        logpdf(Normal(θ[:x1_mean], exp(θ[:log_sigma_x1])), p.x);
    end
    function M!(pnext::SVParticle, pcur::SVParticle, t::Int, data, θ)
        noise = rand(Normal(0.0, exp(θ[:log_sigma_x])));
        pnext.x = exp(θ[:log_rho]) * pcur.x + noise;
        nothing
    end
    function lM(pnext::SVParticle, pcur::SVParticle, t::Int, data, θ)
      rho = exp(θ[:log_rho]);
      sigma = exp(θ[:log_sigma_x]);
      logpdf(Normal(rho * pcur.x, sigma), pnext.x);
    end
    function lGi(p::SVParticle, data, θ)
        #println("lGi: exp(p.x) is $(exp(p.x))");
        #println("lGi: sigma_y is $(exp(θ[:log_sigma_y]))");
        logpdf(Normal(0.0, exp(θ[:log_sigma_y]) * exp(p.x)), data.y[1]);
    end
    function lG(pprev::SVParticle, pcur::SVParticle, t::Int, data, θ)
        #println("lG: exp(pcur.x) is $(exp(pcur.x))");
        #println("lG: sigma_y is $(exp(θ[:log_sigma_y]))");
        logpdf(Normal(0.0, exp(θ[:log_sigma_y]) * exp(pcur.x)), data.y[t]);
    end
    GenericSSM(SVParticle, Mi!, lMi, M!, lM, lGi, lG);
end
println("Model named SV loaded.");

## Required by some of the diffuse initialisation functions.
"""
Get the value of the particle from an SVector or MVector.
"""
function copy!(dest::SVParticle, src::StaticArray{Tuple{1}, Float64, 1})
    dest.x = src[1];
    dest;
end

function SVector(p::SVParticle)
    SVector{1, Float64}(p.x);
end

## Additional functions.
function SV_simobs(y::AVec{<: AFloat},
                   pnext::SVParticle,
                   t::Integer, data, θ)
    y[1] = rand(Normal(0.0, exp(θ[:log_sigma_y]) * exp(pnext.x)));
    nothing;
end
