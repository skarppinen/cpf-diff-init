# Definition of the SEIR model.
include(joinpath(LIB_PATH, "Gaussians.jl"));
include(joinpath(LIB_PATH, "math-functions.jl"));

## Parameters (estimated and fixed) of the model are:
# a: 1/a is the "mean incubation period", affecting flow from E -> I
# \gamma: 1/\gamma is the "mean recovery time", affecting flow from I -> R
# eff: Test effort, on average how many % of infected are observed. (15%?)
# \sigma: Variability of R0 wrt. time. (estimated)
# r0_max: Maximum r0 (4-5?)

## Fields required by this model in the argument `data`:
# y: Vector of the number of cases.
# dt: Vector of delta values.
# N: A constant giving the total population.

# Run with debug mode (very slow)?
const DEBUG_SEIR = false;
## Particle type.
mutable struct SEIRParticle <: Particle
    s::Int # Susceptible
    e::Int # Exposed
    i::Int # Infected
    r::Int # Recovered
    ρ::Float64 # Transformed basic reproduction rate, r0 = r0_max * invlogit(\rho).
end

# Initialise a "null particle".
SEIRParticle() = begin
    i = 0;
    SEIRParticle(i, i, i, i, NaN);
end

import Base.copy!
function copy!(dest::SEIRParticle, src::SEIRParticle)
    dest.s = src.s;
    dest.e = src.e;
    dest.i = src.i;
    dest.r = src.r;
    dest.ρ = src.ρ;
    dest;
end

# Static array from particle.
function SVector{5, Float64}(x::SEIRParticle)
    SVector{5, Float64}(x.s, x.e, x.i, x.r, x.ρ);
end

# Static array from particle (fully diffuse parts)
function SVector{3, Float64}(x::SEIRParticle)
    SVector{3, Float64}(x.e, x.i, x.ρ);
end

# Particle to MVector array.
function copy!(dest::MVector{5, Float64}, src::SEIRParticle)
    dest[1] = src.s;
    dest[2] = src.e;
    dest[3] = src.i;
    dest[4] = src.r;
    dest[5] = src.ρ;
    dest;
end

# For transfering fully diffuse parts.
function copy!(dest::SEIRParticle, src::VecND{3, Float64})
    dest.e = ceil(Int, src[1]);
    dest.i = ceil(Int, src[2]);
    dest.ρ = src[3];
    dest;
end


## Model definition.
function build_SEIR(; variant::Symbol = :betabinomial)
    @assert variant in (:betabinomial, :negativebinomial) string("`variant` ",
    "must be `:betabinomial` or `:negativebinomial`.");

    let variant = variant
        # Sample initial particles. Ignored with diffuse initialisation.
        function Mi!(p::SEIRParticle, data, θ)
            N = data[:N];
            p.e = θ[:e_init];
            p.i = θ[:i_init];
            p.ρ = θ[:ρ_init];
            p.r = 0;
            p.s = N - p.r - p.i - p.e;
            nothing;
        end
        # Compute logpdf of initial particles. Ignored with diffuse initialisation.
        function lMi(p::SEIRParticle, data, θ)
            0.0;
        end
        # Compute initial weights.
        function lGi(pcur::SEIRParticle, data, θ)
            if variant == :betabinomial
                pcur.i < data.y[1] && (return -Inf);
                p = θ[:eff] * (1.0 - exp(-θ[:γ]));
                α = p * (1.0 / θ[:alpha]); αc = 1.0;
                β = (1.0 / p - 1.0) * α; βc = (1.0 / p - 1.0) * αc;
                dist = BetaBinomial(pcur.i, αc + α * pcur.i,
                                            βc + β * pcur.i);
                return logpdf(dist, data.y[1]);
            else
                pcur.i == 0 && (return -Inf);
                p = invlogit(θ[:logit_p]);
                r = θ[:eff] * (1.0 - exp(-θ[:γ])) * p / (1.0 - p);
                dist = NegativeBinomial(pcur.i * r, p);
                return logpdf(dist, data.y[1]);
            end
        end
        # Sample next particles given previous.
        function M!(pnext::SEIRParticle, pcur::SEIRParticle, t, data, θ)
            Δ = data[:dt][t]; sΔ = sqrt(Δ);
            N = data[:N]; r0 = get_r0(pcur.ρ, θ);
            pa = 1.0 - exp(-Δ * θ[:a]);
            pγ = 1.0 - exp(-Δ * θ[:γ]);
            β = r0 * pγ;
            pβ = 1.0 - exp(-Δ * β * pcur.i / N);

            # Sample differences.
            de = rand(Binomial(pcur.s, pβ));
            di = rand(Binomial(pcur.e, pa));
            dr = rand(Binomial(pcur.i, pγ));

            # Advance state.
            pnext.s = pcur.s - de;
            pnext.e = pcur.e + de - di;
            pnext.i = pcur.i + di - dr;
            pnext.r = pcur.r + dr;
            pnext.ρ = rand(Normal(pcur.ρ, sΔ * exp(θ[:log_σ])));
            nothing;
        end
        # Compute logpdf of next particles given previous.
        function lM(pnext::SEIRParticle, pcur::SEIRParticle, t, data, θ)
            Δ = data[:dt][t]; sΔ = sqrt(Δ);
            N = data[:N]; r0 = get_r0(pcur.ρ, θ);
            pa = 1.0 - exp(-Δ * θ[:a]);
            pγ = 1.0 - exp(-Δ * θ[:γ]);
            β = r0 * pγ;
            pβ = 1.0 - exp(-Δ * β * pcur.i / N);

            # Compute how much the states changed.
            de = pcur.s - pnext.s;
            di = pcur.e + de - pnext.e;
            dr = pcur.i + di - pnext.i;

            # Compute logpdf of the changes.
            lp = 0.0;
            lp += logpdf(Normal(pcur.ρ, sΔ * exp(θ[:log_σ])), pnext.ρ);
            lp += logpdf(Binomial(pcur.s, pβ), de);
            lp += logpdf(Binomial(pcur.e, pa), di);
            lp += logpdf(Binomial(pcur.i, pγ), dr);
            lp;
        end
        # Compute weights of successive particles.
        function lG(pprev::SEIRParticle, pcur::SEIRParticle, t, data, θ)
            if variant == :betabinomial
                pcur.i < data.y[t] && (return -Inf);
                p = θ[:eff] * (1.0 - exp(-θ[:γ]));
                α = p * (1.0 / θ[:alpha]); αc = 1.0;
                β = (1.0 / p - 1.0) * α;
                βc = (1.0 / p - 1.0) * αc;
                dist = BetaBinomial(pcur.i, αc + α * pcur.i,
                                            βc + β * pcur.i);
                return logpdf(dist, data.y[t]);
            else
                pcur.i == 0 && (return -Inf);
                p = invlogit(θ[:logit_p]);
                r = θ[:eff] * (1.0 - exp(-θ[:γ])) * p / (1.0 - p);
                dist = NegativeBinomial(pcur.i * r, p);
                return logpdf(dist, data.y[t]);
            end
        end
        GenericSSM(SEIRParticle, Mi!, lMi, M!, lM, lGi, lG);
    end
end

## Some required functions.
"""
Simulate an observation from the SEIR model given values for latent states.
This is the negative binomial version.
"""
function SEIR_simobs!(y::AVec{<: Integer}, pcur::SEIRParticle,
                      t::Integer, data, θ)
    p = invlogit(θ[:logit_p]);
    r = θ[:eff] * (1.0 - exp(-θ[:γ])) * p / (1.0 - p);
    dist = NegativeBinomial(pcur.i * r, p);
    y[1] = rand(dist);
    y;
end

# Functions for transforming \rho -> r0.
get_r0(ρ::AFloat, θ) = θ[:r0_max] * invlogit(ρ);
