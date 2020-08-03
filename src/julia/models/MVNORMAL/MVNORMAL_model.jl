# The definition of the multivariate normal model.

include("../../lib/pfilter.jl");
using Random
using Distributions
using StaticArrays

struct FloatParticle{N} <: Particle
    x::MVector{N, Float64}
end
function FloatParticle{N}() where N
    FloatParticle(zero(MVector{N, Float64}));
end
function copy!(dest::FloatParticle{N}, src::FloatParticle{N}) where N
    for i in eachindex(dest.x)
        @inbounds dest.x[i] = src.x[i];
    end
    dest;
end
function copy!(dest::FloatParticle{N}, src::StaticArray{Tuple{N}, Float64, 1}) where N
    for i in eachindex(dest.x)
        @inbounds dest.x[i] = src[i];
    end
    dest;
end
function SVector{N, Float64}(p::FloatParticle{N}) where N
    SVector{N, Float64}(p.x);
end

function statenames(::Type{FloatParticle{N}}) where N
    Symbol.("x" .* string.(collect(1:N)));
end

function build_MVNORMAL(STATEDIM::Int)
    out = let STATEDIM = STATEDIM
        function lGi(p, data, θ)
            s = 0.0;
            sigma = exp(θ[:log_sigma]);
            dist = Normal(0.0, sigma);
            for i in 1:STATEDIM
                s += logpdf(dist, p.x[i]);
            end
            s;
        end
        # Since T = 1, and M1 is fully diffuse, these are just placeholders.
        Mi!(x, data, p) = nothing;
        lMi(x, data, p) = nothing;
        M!(x_next, x_cur, t::Int, data, p) = nothing;
        lM(x_next, x_cur, t::Int, data, p) = nothing;
        lG(x_prev, x_cur, t::Int, data, p) = nothing;
        GenericSSM(FloatParticle{STATEDIM}, Mi!, lMi, M!, lM, lGi, lG);
    end
    out;
end
