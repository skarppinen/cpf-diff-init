## Adaptive Metropolis algorithms for adapting auxiliary distributions
#  used in diffuse initialisation of CPFs.
# See examples of using these in the src/julia/scripts folder.

abstract type MetropolisState end
dimension(ms::MetropolisState)::Int = typeof(ms).parameters[2];

using LinearAlgebra
using StaticArrays
include("../../../config.jl");
include(joinpath(LIB_PATH, "Gaussians.jl"));
include(joinpath(LIB_PATH, "math-functions.jl"));

"""
Adaptation constants common to AM, ASWAM and AdaptiveCN.
"""
struct AdaptConstants{T <: AFloat}
    n_adapt::Int
    adapt_offset::Int
    gamma::T
    max_eta::T
    function AdaptConstants(n_adapt::Integer, adapt_offset::Integer,
                            gamma::AFloat, max_eta::AFloat)
        @assert (gamma > 0.5 && gamma < 1.0) "'gamma' must belong to the interval (0.5, 1.0).";
        @assert n_adapt >= 0 "'n_adapt' must be non-negative.";
        @assert adapt_offset >= 0 "'adapt_offset' must be non-negative.";
        @assert 1.0 > max_eta > 0.0 "`max_eta` must be in the interval (0, 1)";
        new{typeof(gamma)}(n_adapt, adapt_offset, gamma, max_eta);
    end
end

"""
Default adapt constants.
"""
function AdaptConstants(;n_adapt::Int = typemax(Int), adapt_offset::Int = 0,
                         gamma::Float64 = 0.66, max_eta::Float64 = 0.5)
    AdaptConstants(n_adapt, adapt_offset, gamma, max_eta);
end

"""
Get next value for sequence \$\\eta\$ decaying to zero.
Arguments:
`n`: Iteration number.
`ac`: Adaptation constants.
"""
function get_eta(n::Integer, ac::AdaptConstants)
    min(ac.max_eta, (ac.adapt_offset + n) ^ (-ac.gamma));
end

## RWMKernel.
struct RWMKernel{T <: AFloat, N, MS <: MetropolisState, QFun <: Function}
    Q::QFun # Function that returns the next proposal. Currently assumed to have
            # signature Q(cur, o) where o is the MetropolisState.
            # Q should return an SVector.
    x0::MVector{N, T} # Preallocated vector for latest proposal.
    n::Base.RefValue{Int} # Number of proposals generated.
    o::MS # A metropolis state object. ASWAM, AM, AdaptiveCN...
    function RWMKernel(Q::Function, x0::AVec{<: AFloat}, o::MetropolisState)
        x0 = MVector{dimension(o), eltype(x0)}(x0); n = Ref(0);
        new{eltype(x0), dimension(o), typeof(o), typeof(Q)}(Q, x0, n, o);
    end
end

function propose!(K::RWMKernel, cur::SVector{N, <: AFloat}) where N
    K.x0 .= K.Q(cur, K.o);
    K.n[] += 1;
    nothing;
end

function reset!(K::RWMKernel)
    K.n[] = 0;
    reset!(K.o);
end

## ASWAM
struct ASWAM{T <: AFloat, N, L} <: MetropolisState
    mu::MVector{N, T} # Latest mean of Markov chain.
    cov::MMatrix{N, N, T, L} # Latest proposal covariance of Markov chain.
    sigma::MMatrix{N, N, T, L} # Helper matrix to compute proposal covariance.
    ac::AdaptConstants{T} # Constants of adaptation.
    log_delta::Base.RefValue{T} # Log of current adaptive scaling parameter.
    target::Base.RefValue{T} # Target value for adaptation.

    # Internals / temporaries.
    _mu_init::SVector{N, T} # Initial value for mu.
    _cov_init::SMatrix{N, N, T, L} # Initial value for cov.
    _log_delta_init::T # Initial value for delta. Used by 'reset!'.
    _tmp::Vector{T} # Preallocated vector used in adaptation.
    _w::Matrix{T} # Preallocated matrix used in adaptation.
    function ASWAM(mu::MVector{N, T}, cov::MMatrix{N, N, T, L},
                          ac::AdaptConstants{T}, log_delta::AFloat,
                          target::AFloat) where {T <: AFloat, N, L}
        @assert exp(log_delta) > 0.0 "'exp(log_delta)' must be strictly positive.";
        @assert 1.0 >= target > 0.0 "`target` must be in (0, 1]";

        sigma = copy(cov);
        log_delta = Ref(log_delta);
        _log_delta_init = log_delta[];
        _tmp = zeros(T, N);
        _w = zeros(T, N, N);
        _mu_init = SVector(mu);
        _cov_init = SMatrix(cov);
        new{T, N, L}(mu, cov, sigma, ac, log_delta, Ref(target),
                     _mu_init, _cov_init, _log_delta_init, _tmp, _w);
    end
end

function ASWAM(mu::MVector{N, T} = zero(MVector{N, T}),
               cov::MMatrix{N, N, T, L} = one(MMatrix{N, N, T});
               ac::AdaptConstants{T} = AdaptConstants(),
               log_delta::AFloat = 2.0 * log(2.38) - log(N),
               target::AFloat = 0.8) where {N, T <: AFloat, L}
    ASWAM(mu, cov, ac, log_delta, target);
end

function dimension(aswam::ASWAM{T, N, L}) where {T, N, L}
    N;
end

function reset!(aswam::ASWAM)
    aswam.log_delta[] = aswam._log_delta_init;
    aswam.mu .= aswam._mu_init;
    aswam.sigma .= aswam._cov_init;
    aswam.cov .= aswam._cov_init;
    nothing;
end

function adapt!(n::Integer, aswam::ASWAM{T, N},
                ps::ParticleStorage, refi::Integer) where {T, N}
    # No adaptation after 'n_adapt' iterations.
    aswam.ac.n_adapt < n && (return nothing;)

    # Get pointers.
    X = ps.X; weights = ps.wnorm;
    mu = aswam.mu; sigma = aswam.sigma; cov = aswam.cov;
    log_delta = aswam.log_delta;
    _w = aswam._w; _tmp = aswam._tmp;

    ## Adapt.
    eta = get_eta(n, aswam.ac);

    target = 1.0 - weights[refi];
    log_delta[] = log_delta[] + eta * (target - aswam.target[]);

    # Covariance update.
    # sigma = (1.0 - eta) * sigma + eta * [sum w[i] * (x[i] - mu)*(x[i] - mu)^T]
    # cov = exp(log_delta) * sigma
    _w .= zero(eltype(_w));
    for i in 1:length(weights)
        # Note: `cov` is used as a temp here.
        @inbounds _tmp .= SVector{N, T}(X[i, 1]) .- mu;
        mul!(cov, _tmp, transpose(_tmp));
        @inbounds _w .= _w .+ weights[i] .* cov;
    end
    sigma .= (1.0 - eta) .* sigma .+ eta .* _w;
    cov .= exp(log_delta[]) .* sigma;
    symmetrise!(cov); # Smooth out small numerical errors.

    # Mean update.
    # mu = (1.0 - eta) * mu + eta * [sum w[i] * (x[i] - mu)]
    _tmp .= zero(eltype(_tmp));
    for i in 1:length(weights)
        @inbounds _tmp .= _tmp .+ weights[i] .* SVector{N, T}(X[i, 1]);
    end
    mu .= (1.0 - eta) .* mu .+ eta .* _tmp;
    nothing;
end

## AM
struct AM{T <: AFloat, N, L} <: MetropolisState
    mu::MVector{N, T} # Latest mean of Markov chain.
    sigma::MMatrix{N, N, T, L} # Unscaled covariance
    cov::MMatrix{N, N, T, L} # c * sigma: Latest proposal covariance of Markov chain.
    c::T # Covariance scaling.
    ac::AdaptConstants{T} # Constants of adaptation.

    # Internals / temporaries.
    _mu_init::SVector{N, T} # Initial value for mu.
    _sigma_init::SMatrix{N, N, T, L} # Initial value for cov.
    function AM(mu::MVector{N, T}, sigma::MMatrix{N, N, T, L},
                ac::AdaptConstants{T} = AdaptConstants(),
                c::T = T(2.38 ^ 2 / N)) where {T <: AFloat, N, L}
        cov = c * sigma;
        _mu_init = SVector(mu);
        _sigma_init = SMatrix(cov);
        new{T, N, L}(mu, sigma, cov, c, ac, _mu_init, _sigma_init);
    end
end

function adapt!(n::Integer, am::AM, ps::ParticleStorage, refi::Integer)
    # No adaptation after 'n_adapt' iterations.
    am.ac.n_adapt < n && (return nothing;)
    eta = get_eta(n, am.ac);
    mu = am.mu; sigma = am.sigma; cov = am.cov;
    xref = SVector(ps.X[ps.ref[1], 1]);
    xmudiff = xref - mu;

    # Update sigma, mu and cov.
    sigma .= (1.0 - eta) .* sigma .+ eta .* xmudiff * transpose(xmudiff);
    mu .= (1.0 - eta) .* SVector(mu) .+ eta .* xref;
    cov .= am.c .* sigma;
    nothing;
end

function reset!(am::AM)
    am.mu .= am._mu_init;
    am.sigma .= am._sigma_init;
    am.cov .= am.c .* am._sigma_init;
    nothing;
end


## AdaptiveCN.
struct AdaptiveCN{T <: AFloat, N, L} <: MetropolisState
    mu::MVector{N, T}
    cov::MMatrix{N, N, T, L}
    logit_beta::Base.RefValue{T}
    target::Base.RefValue{T} # Target acceptance.
    ac::AdaptConstants{T} # Constants of adaptation.

    # Internals / temporaries.
    _logit_beta_init::T
    function AdaptiveCN(mu::MVector{N, <: AFloat}, cov::MMatrix{N, N, <: AFloat, L},
                        beta::AFloat, target::AFloat, ac::AdaptConstants) where {N, L}
        @assert 0.0 <= beta <= 1.0 "`beta` must be in the interval [0, 1]";
        @assert 0.0 <= target <= 1.0 "`target` must be in the interval [0, 1]";
        logit_beta = Ref(logit(beta));
        target = Ref(target);
        _logit_beta_init = logit_beta[];
        new{typeof(beta), N, L}(mu, cov, logit_beta, target, ac,
                                _logit_beta_init);
    end
end

function AdaptiveCN(mu, cov; beta::AFloat, target::AFloat, ac = AdaptConstants())
    AdaptiveCN(mu, cov, beta, target, ac);
end

function reset!(acn::AdaptiveCN)
    nothing;
end

function adapt!(n::Integer, acn::AdaptiveCN, ps::ParticleStorage, refi::Integer)
    # No adaptation after 'n_adapt' iterations.
    acn.ac.n_adapt < n && (return nothing;)
    logit_beta = acn.logit_beta; target = acn.target[];

    eta = get_eta(n, acn.ac);
    accrate = 1.0 - ps.wnorm[refi];
    logit_beta[] = logit_beta[] + eta * (accrate - target);
    nothing;
end
