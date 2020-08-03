# The definition of the noisy AR(1) model (and noisy RW model).

include("../../lib/pfilter.jl");
using Random
using Distributions
using StaticArrays

## Particle type and required functions for model.
mutable struct NoisyarParticle <: Particle
    x::Float64
end
function NoisyarParticle()
    NoisyarParticle(NaN);
end
import Base.copy!
function copy!(dest::NoisyarParticle, src::NoisyarParticle)
    dest.x = src.x;
    dest;
end
function SVector{1, Float64}(p::NoisyarParticle)
    SVector{1, Float64}(p.x);
end

## Model definition.
NOISYAR = let
    # Initial distribution.
    function Mi!(p::NoisyarParticle, data, θ)
        p.x = rand(Normal(θ[:x1_mean], exp(θ[:log_sigma_x1])));
        nothing
    end
    function lMi(p::NoisyarParticle, data, θ)
        logpdf(Normal(θ[:x1_mean], exp(θ[:log_sigma_x1])), p.x);
    end

    # Forward model.
    function M!(pnext::NoisyarParticle,
                pcur::NoisyarParticle, t::Int, data, θ)
        noise = rand(Normal(0.0, exp(θ[:log_sigma_x])));
        rho = exp(θ[:log_rho]);
        pnext.x = rho * pcur.x + noise;
        nothing;
    end
    function lM(pnext::NoisyarParticle,
                pcur::NoisyarParticle, t::Int, data, θ)
        rho = exp(θ[:log_rho]);
        sigma = exp(θ[:log_sigma_x]);
        logpdf(Normal(rho * pcur.x, sigma), pnext.x);
    end

    # Weights.
    function lGi(p::NoisyarParticle, data, θ)
        logpdf(Normal(data.y[1], exp(θ[:log_sigma_y])), p.x);
    end
    function lG(pprev::NoisyarParticle, pcur::NoisyarParticle, t::Int, data, θ)
        logpdf(Normal(data.y[t], exp(θ[:log_sigma_y])), pcur.x);
    end
    GenericSSM(NoisyarParticle, Mi!, lMi, M!, lM, lGi, lG);
end
println("Model named NOISYAR loaded.");

## Required by some of the diffuse initialisation functions.
function copy!(dest::NoisyarParticle, src::StaticArray{Tuple{1}, Float64, 1})
    dest.x = src[1];
    dest;
end

function SVector(p::NoisyarParticle)
    SVector{1, Float64}(p.x);
end

## Additional functions.
"""
Function to simulate observations given the latent state from the model.
"""
function NOISYAR_simobs(y::AVec{<: AFloat}, p::NoisyarParticle,
                        t::Integer, data, θ)
    y[1] = rand(Normal(p.x, exp(θ[:log_sigma_y])));
    nothing;
end

using RCall
function KFAS_noisyar_diff_mean_var(y, p, diffuse::Bool = true)
        x = R"""
        library(KFAS)

        # Parameters.
        par <- c(log_sigma_x1 = $(p[:log_sigma_x1]),
                 log_sigma_x = $(p[:log_sigma_x]),
                 log_sigma_y = $(p[:log_sigma_y]),
                 log_rho = $(p[:log_rho]))

        y <- $y

        # Define model matrices.
        Zt <- matrix(1)
        Ht <- matrix(exp(2.0 * par["log_sigma_y"]))
        Tt <- matrix(exp(par["log_rho"]))
        Rt <- matrix(1)
        Qt <- matrix(exp(2.0 * par["log_sigma_x"]))
        a1 <- matrix(0)

        if ($diffuse) {
            P1 <- matrix(0)
            P1inf <- diag(1) # Mark diffuse initialisation elements.
        } else {
            P1 <- matrix(exp(2.0 * par["log_sigma_x1"]))
            P1inf <- matrix(0, 1, 1)
        }

        # Define model object.
        model_ar1_diff <- SSModel(y ~ -1 + SSMcustom(Z = Zt, T = Tt, R = Rt,
                                                     Q = Qt, a1 = a1, P1 = P1,
                                                     P1inf = P1inf), H = Ht)

        # Run Kalman filter and smoother, get first smoothed mean and var.
        out <- KFS(model_ar1_diff)
        mean1 <- as.numeric(out[["alphahat"]])[1]
        var1 <- as.numeric(out[["V"]])[1]
        c(mean1, var1)
        """;
        res = rcopy(x);
        (mean = res[1], var = res[2]);
end
