# Simplistic implementation of the RAM algorithm by Vihola (2012).

include("../../../config.jl");
using LinearAlgebra
using StaticArrays

"""
An abstract type labeling MCMCAlgorithms.
"""
abstract type MCMCAlgorithm end

"""
A type wrapping an unnormalised logposterior density function and an MCMC algorithm.

Fields:
`lp`: The unnormalised logposterior density function.
`alg`: The MCMC algorithm. Must be a subtype of MCMCAlgorithm.
"""
struct SimpleMCMCProblem{MCMCAlg <: MCMCAlgorithm, Logpost <: Function}
    lp::Logpost
    alg::MCMCAlg
end

"""
A type for storing adaptation settings for the RAM algorithm.
"""
struct RAMOptions{T <: AFloat, N}
    alpha_star::T # Target acceptance.
    gamma::T # Parameter for sequence decaying to zero.
    n_adapt::Int # How many iterations to adapt.
    max_eta::T

    _u::MVector{N, T} # Standard normal random vector used in adaptation.
    _v::Vector{T} # Helper vector used in adaptation.
    function RAMOptions{T, N}(alpha_star::T, gamma::T,
                              n_adapt::Int, max_eta::T) where {T <: AFloat, N}
        @assert alpha_star > 0.0 && alpha_star < 1.0 "'alpha_star' must belong to the interval (0.0, 1.0).";
        @assert gamma > 0.5 && gamma < 1.0 "'gamma' must belong to the interval (0.5, 1.0).";
        @assert 1.0 > max_eta > 0.0 "'max_eta' must belong to the interval (0.0, 1.0)"
        @assert n_adapt >= 0 "'n_adapt' must be non-negative.";
        _u = zero(MVector{N, T});
        _v = zeros(T, N);
        new{T, N}(alpha_star, gamma, n_adapt, max_eta, _u, _v);
    end
end

function RAMOptions{T, N}(;alpha_star::T = default_alpha_star(N, T),
                    gamma::T = T(2.0 / 3.0),
                    n_adapt::Int = typemax(Int),
                    max_eta::T = T(0.5)) where {N, T}
    RAMOptions{T, N}(alpha_star, gamma, n_adapt, max_eta);
end

# Target acceptance is set according to the dimension of the vector sampled.
# The 'optimal values' are according to the paper "Efficient Metropolis jumping rules" (1996) by
# Gelman, Roberts & Gilks. The value of 0.234 is optimal in the limit as 'dim' -> ∞.
function default_alpha_star(dim::Integer, T::DataType = Float64)
    t = T.([0.441, 0.352, 0.316, 0.279, 0.275, 0.266, 0.261, 0.255, 0.261, 0.267]);
    dim > 10 ? T(0.234) : t[dim];
end

import Base.eltype
function eltype(ro::RAMOptions{T, N}) where {T, N}
    T;
end

struct RAM{T <: AFloat, N, L} <: MCMCAlgorithm
    S::Cholesky{T, MMatrix{N, N, T, L}} # Cholesky factor of the covariance matrix
                                # of the proposal distribution.
    cur::MVector{N, T} # Current sample.
    proposal::MVector{N, T} # Latest proposal.
    lp_current::Base.RefValue{T} # Latest logposterior value.
    n::Base.RefValue{Int} # Number of proposals generated.
    acc_rate::Base.RefValue{T} # Latest acceptance rate.
    recalculate::Bool # A boolean value determining if the denominator of the
                      # acceptance ratio should be recalculated during sampling.
                      # Some targets require this.
    o::RAMOptions{T, N} # Adaptation options.

    function RAM(S::Cholesky{T, MMatrix{N, N, T, L}},
                 o::RAMOptions{T, N},
                 recalculate::Bool = false) where {T <: AFloat, N, L}
        cur = zero(MVector{N, T});
        proposal = zero(MVector{N, T});
        lp_current = Ref(-Inf);
        n = Ref(0);
        acc_rate = Ref(-Inf);
        new{T, N, L}(S, cur, proposal, lp_current, n, acc_rate, recalculate, o);
    end
end

function RAM{T, N, L}(S::Cholesky{T, MMatrix{N, N, T, L}} =
                      cholesky(one(MMatrix{N, N, T, L}));
                      o::RAMOptions{T, N} = RAMOptions{T, N}(),
                      recalculate::Bool = true) where {T <: AFloat, N, L}
    RAM(S, o, recalculate);
end

function dimension(r::RAM{T, N}) where {T, N}
    N;
end

function reset!(r::RAM)
    r.n[] = 0;
    r.lp_current[] = -Inf;
    r.acc_rate[] = 0.0;
    nothing;
end

"""
Reset the RAM object and set initial value and initial lp value.
"""
function init!(r::RAM{T, N}, init::StaticArray{Tuple{N}, T, 1}, lp::Function) where {N, T}
    reset!(r);
    r.cur .= init;
    r.lp_current[] = lp(r.cur);
end

"""
Function adapts the covariance matrix of the proposal distribution.
"""
function adapt!(r::RAM{T, N, L}) where {N, T, L}
    opt = r.o;

    # No adaptation after 'n_adapt' iterations.
    opt.n_adapt < r.n[] && (return nothing;)

    # Adapt.
    eta = min(opt.max_eta, N * r.n[] ^ (-opt.gamma));
    delta = r.acc_rate[] - opt.alpha_star;

    #v = S.L * u / norm(u) * sqrt(eta * abs(delta));
    normalize!(opt._u);
    opt._u .= sqrt(eta * abs(delta)) .* opt._u;
    mul!(opt._v, r.S.L, opt._u); # This allocates..
    if delta >= 0.0
        lowrankupdate!(r.S, opt._v);
    else
        lowrankdowndate!(r.S, opt._v);
    end
    nothing;
end

"""
Function generates a RAM proposal.
The result is calculated to a preallocated vector 'proposal'.
"""
function propose!(r::RAM)
    opt = r.o;
    # Sample standard normal random vector to r.u,
    # which is stored for adaptation.
    for i in 1:length(opt._u)
        opt._u[i] = randn();
    end
    # Store transformed normal vector to proposal.
    r.proposal .= r.S.L * SVector(opt._u);
    r.proposal .= r.proposal .+ r.cur;

    # Increase number of proposals generated.
    r.n[] += 1;
    nothing;
end

"""
Sample next sample using the RAM algorithm.
The proposal distribution is automatically adapted after proposing next value.
This function does not output anything, the next sample is computed to
`r.cur`.
"""
function sample!(r::RAM, lp::Function)
    # Propose and compute lp for proposal.
    propose!(r); # r.proposal gets computed.
    lp_proposal = lp(r.proposal);

    # Calculate acceptance ratio.
    if r.recalculate
        r.acc_rate[] = min(1.0, exp(lp_proposal - lp(r.cur)));
    else
        r.acc_rate[] = min(1.0, exp(lp_proposal - r.lp_current[]));
    end
    # Handle rare case that acc_rate might become NaN,
    # when acc_rate = min(1.0, exp(-Inf - -Inf))
    # Let the chain move in this case.
    isnan(r.acc_rate[]) && (r.acc_rate[] = 1.0);

    # Adapt.
    adapt!(r);

    # Accept reject step.
    if rand() <= r.acc_rate[]
        r.cur .= r.proposal;
        r.lp_current[] = lp_proposal;
    end
    nothing
end
