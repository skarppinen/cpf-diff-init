## Algorithms and helper functions for diffuse initialisation of particle filters.
include(joinpath(LIB_PATH, "adaptive-aux.jl"));
include(joinpath(LIB_PATH, "RAM.jl"));

"""
An output object for diffuse CPF algorithms storing, for a time series of length
`T` and number of iterations `M`:

* `x`: A matrix of particles with dimension (`T` * `M`). Each column
of this matrix is reserved for one trajectory of the latent state.
* `x0`: A vector of particles with length `M`. Each particle of this vector
is reserved for the x0 value of that iteration.
"""
struct DiffuseCPFOutput{P <: Particle}
    x::Matrix{P}
    x0::Vector{P}
    function DiffuseCPFOutput(x::Matrix{P}, x0::Vector{P}) where {P <: Particle}
        @assert !isempty(x) "`x` must not be degenerate";
        @assert !isempty(x0) "`x0` must not be degenerate";
        new{P}(x, x0);
    end
end

function DiffuseCPFOutput(::Type{P}; ts_length::Integer, nsim::Integer) where {P <: Particle}
    @assert ts_length > 0 "`ts_length` should be > 0";
    @assert nsim > 0 "`nsim` should be > 0";
    x = [P() for i in 1:ts_length, j in 1:nsim];
    x0 = [P() for i in 1:nsim];
    DiffuseCPFOutput(x, x0);
end

"""
Symmetrise a matrix in place. Useful for dealing with small numerical
inconsistencies that make the Cholesky decomposition fail, for instance.
"""
@inline function symmetrise!(X::AMat{<: Real})
    for j in 2:size(X)[2]
        for i in 1:(j-1)
            @inbounds m = (X[i, j] + X[j, i]) / 2.0;
            @inbounds X[i, j] = X[j, i] = m;
        end
    end
    nothing;
end

function check_thinning(n::Integer, n_thin::Integer)
    @assert n % n_thin == 0 string("invalid thinning, `n_thin` = ", n_thin,
                                    " does not divide `n` = ", n, " evenly.");
end

"""
Run multiple iterations of the adaptive auxiliary initialisation CPF (AAI-CPF).
This function can also be run without adaptation by configuring the adaptation
settings of the RWMKernel object.
It is assumed that `ssm.model.Mi!` e.g M1 of the Feynman-Kac model has been set
to sample from the same auxiliary distribution Q, which the RWMKernel `rk` contains.
This is because Mi! needs to be able to know the latest value for the
auxiliary variable x0.
In practise, it is easiest to achieve this with a "let block function"
such that Mi! is referencing `rk`. See the script files for examples of doin
this.

Arguments:
* cpfout: A DiffuseCPFOutput object, see separate documentation for the object.
The dimensions of this object determine the number of iterations. The simulation
results will be computed in place to this object.
* ssm: The state space model object with the auxiliary distribution Q! as Mi!.
* rk: An RWMKernel responsible for updates to the auxiliary variable x0.
* θ: Parameters of the model.

Optional:
* thin: Thinning. Thin = 10 means every tenth iteration after burnin is saved.
* burnin: Amount of burnin iterations before saving iterations.
* traceback: A type <: Traceback determining the traceback algorithm to use,
  currently `BackwardSampling` or `AncestorTracing` are supported.
* resampling: An object of type <: Resampling determining which resampling to
  use. `MultinomialResampling()` is the only option.
"""
function aai_cpf!(cpfout::DiffuseCPFOutput,
                 ssm::SSMInstance{<: GenericSSM},
                 rk::RWMKernel{T, DiffDim}, θ;
                 thin::Int = 1,
                 burnin::Int = 200,
                 traceback::Type{<: Traceback} = BackwardSampling,
                 resampling::Resampling = MultinomialResampling()) where {T, DiffDim, P <: Particle}
    @assert burnin >= 0 "`burnin` must be >= 0.";
    @assert thin >= 1 "`thin` must be >= 1";
    reset!(rk);
    ts_length = length(ssm);
    iter = size(cpfout.x, 2) * thin + burnin;

    # Initialise by running the standard particle filter
    # once and setting reference.
    ref = ssm.storage.ref; X = ssm.storage.X;
    pf!(ssm, θ; resampling = resampling);
    set_reference!(ssm.storage);
    refi_old = refi_new = ref[1];

    # Run simulations.
    for i in 1:iter
        # 1. x0 update: update rk.x0
        propose!(rk, SVector{DiffDim, T}(X[refi_new, 1]));

        # 2. Update latent trajectory. ssm should reference a pointer to rk.
        pf!(ssm, θ; resampling = resampling, conditional = true);
        traceback!(ssm, θ, traceback);
        refi_new = ref[1];

        # Adapt RWM algorithm.
        adapt!(i, rk.o, ssm.storage, refi_old);
        refi_old = refi_new;

        # Save results.
        if i > burnin
            j = i - burnin;
            if j % thin == 0
                j = Int(j / thin);
                @inbounds trace_reference!(cpfout.x, j, ssm.storage);
                @inbounds copy!(cpfout.x0[j], rk.x0);
            end
        end
    end
    nothing;
end

"""
Adaptive FDI-CPF with RAM updates for model parameters.
This algorithm uses two different adaptive Metropolis schemes to:
1. Tune the auxiliary distribution.
2. Tune the proposal distribution for the parameters.
See the script files for examples of running this function.

Arguments:
* `out`: Currently, this should be a Tuple{DiffuseCPFOutput, Vector{MVector}},
where sampling results are computed to. The number of iterations is determined
based on appropriate dimensions of this object.
The second field in the tuple is a container for the sampled parameter values,
the dimension of the MVectors should match the parameter dimension.
* `ssm`: An SSMInstance containing a GenericSSM, which has the auxiliary sampler Q
set as the field `Mi!` in `ssm.model`.
* rk: An RWMKernel responsible for updates to the auxiliary variable x0.
* `θ_mcmc`: A `SimpleMCMCProblem` object, that contains the logposterior pdf
and the RAM algorithm for updating the model parameters. See file `RAM.jl` for
details.
* `θ`: Initial values for the parameters of the model.

Optional:
* traceback: A type <: Traceback determining the traceback algorithm to use,
  currently `BackwardSampling` or `AncestorTracing` are supported.
* resampling: An object of type <: Resampling determining which resampling to
  use. `MultinomialResampling()` is the only option.
* burnin: Amount of burnin iterations before saving iterations.
* thin: Thinning. Thin = 10 means every tenth iteration after burnin is saved.
"""
function fdi_pg!(out, ssm::SSMInstance{<: GenericSSM},
                 rk::RWMKernel{T, DiffDim},
                 θ_mcmc::SimpleMCMCProblem{<: RAM{T, ThetaDim}}, θ;
                 traceback::Type{<: Traceback} = BackwardSampling,
                 resampling::Resampling = MultinomialResampling(),
                 burnin::Int = 100, thin::Int = 1) where {T, DiffDim, ThetaDim}
    @assert burnin >= 0 "`burnin` must be >= 0.";
    θ = deepcopy(θ);
    ts_length = length(ssm);
    cpfout = out[1]; θ_out = out[2];
    iter = size(cpfout.x, 2) * thin + burnin;
    θ_lp = θ_mcmc.lp; θ_alg = θ_mcmc.alg;

    # Reset fully diffuse adaptation object to inits at first construction.
    reset!(rk);

    # Initialise reference trajectory.
    ref = ssm.storage.ref; X = ssm.storage.X;
    pf!(ssm, θ; resampling = resampling);
    set_reference!(ssm.storage);
    refi_old = refi_new = ref[1];

    # Initialise MCMC for θ.
    init!(θ_alg, SVector{ThetaDim, T}(θ), θ_lp);

    for i in 1:iter
        ## Sample x[0] (to rk.x0).
        propose!(rk, SVector{DiffDim, T}(X[refi_old, 1]));

        ## Sample x[1:K] | x[0].
        pf!(ssm, θ; resampling = resampling, conditional = true);
        traceback!(ssm, θ, traceback);
        refi_new = ref[1];

        ## Adapt Q.
        adapt!(i, rk.o, ssm.storage, refi_old);
        refi_old = refi_new;

        ## Sample parameters (and adapt).
        sample!(θ_alg, θ_lp);
        copy!(θ, θ_alg.cur);

        ## Save sampled values after burnin.
        if i > burnin
            j = i - burnin;
            if j % thin == 0
                j = Int(j / thin);
                @inbounds trace_reference!(cpfout.x, j, ssm.storage);
                @inbounds copy!(cpfout.x0[j], rk.x0);
                @inbounds copy!(θ_out[j], θ);
            end
        end
    end
    nothing;
end

"""
The DPG-BS e.g diffuse initialisation by treating the first state
as a parameter, and the remaining states as the latent state process,
e.g a particle Gibbs algorithm with parameters x[1] as a parameter and x[2:T]
as the latend process.
The RAM algorithm is used to propose x[1], and CPF-BS to propose x[2:T].
See the script file `run-dpg-cpf-noisyar-sv.jl` for an example of running
this function.

Arguments:
* `out`: A matrix of particles with dimensions T times M where T is
         the length of the time series and M is the number of simulations.
         The simulation results will be computed in place here.
* `ssm`: A `GenericSSM` object that contains the model
       \$ \\prod_{t=2}^{T}M_t(x_t | x_{t-1})G_t(x_t)\$. `Mi!` should depend on
       `data` such that the sampling is done conditional on `ssm.data[:x1]`.
       Hence, the data passed to the model function must contain the field `x1`
       of the appropriate size.
       See function `shift` to construct an appropriate model object conveniently.
* `θ_mcmc`: A `SimpleMCMCProblem` wrapping a `RAM` object initialised with
    `recalculate = true`, and a function of x returning the log of
    \$M_1(x)G_1(x)M_2(y | x)G_2(x, y)\$ where y is the current value of \$x_2\$.
    The function is the target for updating the initial state.
* `θ`: (Fixed) parameters of the model.

Optional:
* resampling: An object of type <: Resampling determining which resampling to
  use. `MultinomialResampling()` is the only option.
* burnin: Amount of burnin iterations before saving iterations.
* thin: Thinning. Thin = 10 means every tenth iteration after burnin is saved.
"""
function dpg_bs!(cpfout::Matrix{P},
                  ssm::SSMInstance{<: GenericSSM{P}},
                  θ_mcmc::SimpleMCMCProblem, θ;
                  resampling::Resampling = MultinomialResampling(),
                  burnin::Int = 100,
                  thin::Int = 1) where {P <: Particle}

    @assert !isempty(cpfout) "`cpfout` must not be degenerate";
    X = ssm.storage.X; ref = ssm.storage.ref;
    ts_length = size(cpfout, 1);
    iter = size(cpfout, 2) * thin + burnin;
    θ_lp = θ_mcmc.lp; θ_alg = θ_mcmc.alg;

    # Initialise reference for x[2:T].
    pf!(ssm, θ; resampling = resampling);
    set_reference!(ssm.storage);

    # Initialise RAM.
    init!(θ_alg, SVector(ssm.data[:x1]), θ_lp);

    # Run simulations and save results to `cpfout`.
    # Note that the particle storage object has ts_length - 1 time points,
    # (from indices 2:T), hence the t - 1 indexing when saving to `cpfout`.
    for i in 1:iter
        # Sample latent state.
        pf!(ssm, θ; resampling = resampling, conditional = true);
        traceback!(ssm, θ, BackwardSampling);

        # Sample parameters and adapt.
        sample!(θ_alg, θ_lp); copy!(ssm.data[:x1], θ_alg.cur);

        # Save iteration.
        if i > burnin
            j = i - burnin;
            if j % thin == 0
                j = Int(j / thin);
                for t in 2:ts_length
                    @inbounds copy!(cpfout[t, j], X[ref[t - 1], t - 1]);
                end
                @inbounds copy!(cpfout[1, j], ssm.data[:x1]);
            end
        end
    end
    nothing;
end

"""
'Shift' a `GenericSSM` model
\$M_1(x_1)G_1(x_1)\\prod_{t=2}^{T}M_t(x_t | x_{t-1})
G_t(x_{t-1}, x_t)\$ such that the model output from this function is
\$ \\prod_{t=2}^{T}M_t(x_t | x_{t-1})G_t(x_t)\$. It is assumed that
\$x_1\$ is passed in the `data` parameter of the model building functions.
Currently, this function only works for models with univariate state.
"""
function shift(m::GenericSSM{P}) where P <: Particle
    let lMi = m.lMi, M! = m.M!, lM = m.lM,
        lGi = m.lGi, lG = m.lG

        Mi!_new(p::P, data, θ) = M!(p, data[:x1], 2, data, θ);
        lMi_new(p::P, data, θ) = lM(p, data[:x1], 2, data, θ);
        M!_new(pnext::P, pcur::P, t::Int, data, θ) = M!(pnext, pcur, t + 1, data, θ);
        lM_new(pnext::P, pcur::P, t::Int, data, θ) = lM(pnext, pcur, t + 1, data, θ);
        lGi_new(p::P, data, θ) = lG(data[:x1], p, 2, data, θ);
        lG_new(pprev::P, pcur::P, t::Int, data, θ) = lG(pprev, pcur, t + 1, data, θ);

        GenericSSM(P, Mi!_new, lMi_new,
                   M!_new, lM_new,
                   lGi_new, lG_new);
    end;
end
