using AdaptiveParticleMCMC, Distributions

# Define the particle type for the model (here, latent is univariate random walk)
mutable struct SEIRParticle
    s::Int # Susceptible
    e::Int # Exposed
    i::Int # Infected
    r::Int # Recovered
    trans_R0::Float64 # Transformed basic reproduction rate
    SEIRParticle() = new(0,0,0,0,3.0) # Void constructor required!
end

# Model parameters:
mutable struct SEIRParam
    trans_R0_init::Float64 # First value for R0
    σ::Float64      # AR(1) deviation of log R0
    p::Float64      # Observation model NegBin(ir, p)
    r::Float64
    incubation_rate::Float64 # Incubation rate
    recovery_rate::Float64 # Recovery rate
    effort::Float64 # Sampling effort
    i1::Int         # Number of infected at the beginning
    e1::Int         # Number of exposed at the beginning
    SEIRParam() = new(0.0, 1.0, 1.0, 1.0,
    1/seir_const.incubation_time, 1/seir_const.recovery_time,
    seir_const.sampling_effort, 0, 0)
end

# Inverse logit transform:
inv_logit(x) = 1.0/(1.0+exp(-x)) # (-∞,∞) → (0,1)
# Get R0 from transformed R0
R0_transform(x) = seir_const.R0_max*inv_logit(x)

# This will be the SequentialMonteCarlo "particle scratch", which
# will contain both model data & parameters, and which will be
# the 'scratch' argument of M_ar1!, lM_ar1, lG_sv
struct SEIRScratch
    par::SEIRParam      # Parameters
    n::Vector{Int64}    # Observations
    dt::Vector{Float64} # Time differences
    n_tot::Int          # Total population
end
SEIRScratch() = SEIRScratch(SEIRParam(), data[:,seir_const.cases_name],
vcat(1,Dates.days.(diff(data[:,seir_const.date_name]))), seir_const.population)


# Transition *simulator*
function M_SEIR!(x, rng, k, x_prev, scratch)
    if k == 1
        x.e = scratch.par.e1; x.i = scratch.par.i1
        x.r = 0; x.s = max(1, scratch.n_tot - x.e - x.i - x.r);
        x.trans_R0 = scratch.par.trans_R0_init
    else
        dt = scratch.dt[k]; sdt = sqrt(dt)
        R0 = R0_transform(x_prev.trans_R0)

        μ = R0*(1.0-exp(-scratch.par.recovery_rate*dt))*(x_prev.i/scratch.n_tot)
        de = rand(rng, Binomial(x_prev.s, 1.0-exp(-μ*dt)))
        di = rand(rng, Binomial(x_prev.e, 1.0-exp(-scratch.par.incubation_rate*dt)))
        dr = rand(rng, Binomial(x_prev.i, 1.0-exp(-scratch.par.recovery_rate*dt)))
        x.s = x_prev.s-de
        x.e = x_prev.e+de-di
        x.i = x_prev.i+di-dr; x.r = x_prev.r + dr
        x.trans_R0 = rand(rng, Normal(x_prev.trans_R0, sdt*scratch.par.σ))
    end
    nothing
end

# Transition *density/probability mass*
function lM_SEIR(k, x_prev, x, scratch)
    p = 0.0
    if k > 1
        dt = scratch.dt[k]; sdt = sqrt(dt)
        if k == 2
            trans_R0_ = scratch.par.trans_R0_init
            e_ = scratch.par.e1; i_ = scratch.par.i1
            r_ = 0; s_ = scratch.n_tot - e_ - i_ - r_
        else
            trans_R0_ = x_prev.trans_R0
            s_, e_, i_, r_ = x_prev.s, x_prev.e, x_prev.i, x_prev.r
        end
        p = logpdf(Normal(trans_R0_, sdt*scratch.par.σ), x.trans_R0)
        R0 = R0_transform(trans_R0_)
        de = s_ - x.s
        di = e_+de-x.e
        dr = i_+di-x.i
        μ = R0*(1.0-exp(-scratch.par.recovery_rate*dt))*(i_/scratch.n_tot)
        p += logpdf(Binomial(s_, 1.0-exp(-μ*dt)), de)
        p += logpdf(Binomial(e_, 1.0-exp(-scratch.par.incubation_rate*dt)), di)
        p += logpdf(Binomial(i_, 1.0-exp(-scratch.par.recovery_rate*dt)), dr)
    end
    p
end

# Observation model
function lG_SEIR(k, x, scratch)
    n = scratch.n[k]; dt = scratch.dt[k]
    i = (k == 1) ? scratch.par.i1 : x.i
    i == 0 ? -Inf : logpdf(NegativeBinomial(i*scratch.par.r, scratch.par.p), n)
end

# Observation model simulator (used for posterior predictive check only)
function G_SEIR(k, x, scratch)
    n = scratch.n[k]; dt = scratch.dt[k]
    i = (k == 1) ? scratch.par.i1 : x.i
    rand(NegativeBinomial(i*scratch.par.r, scratch.par.p))
end

# We are sampling transformed parameters
# This function sets the model parameters based on sampled (transformed)
function set_param!(scratch, θ)
    scratch.par.σ = exp(θ.log_sigma)
    scratch.par.e1 = round(Int, θ.e); #1 + ceil(θ.e)
    scratch.par.i1 = round(Int, θ.i); #1 + ceil(θ.i)
    scratch.par.trans_R0_init = θ.trans_R0_init
    scratch.par.p = inv_logit(θ.logit_p)
    scratch.par.r = scratch.par.effort * (1.0-exp(-scratch.par.recovery_rate)) * scratch.par.p/(1-scratch.par.p)
end

function prior(θ)
    # Uniform prior on e and i, but there are constraints
    i = round(Int, θ.i);
    e = round(Int, θ.e);
    if i < 0 || e < 0 || i + e > seir_const.population
        #!(0 <= θ.i && 0 <= θ.e && θ.i+θ.e+2 <= seir_const.population)
        return -Inf;
    else
        p = 0.0
        p += logpdf(Normal(-2.0, 0.3), θ.log_sigma)
        p += logpdf(Normal(0, 10.0), θ.logit_p)
        return p
    end
end
