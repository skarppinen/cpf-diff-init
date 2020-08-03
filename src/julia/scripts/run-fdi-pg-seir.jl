## Script to run the SEIR simulations with the FDI-PG.
# Run `julia run-fdi-pg-seir.jl --help" for help.

## Parse arguments.
include("../../../config.jl");
include(joinpath(CONFIG_PATH, "parse-settings.jl"));
args = parse_fdi_pg_seir_args(ARGS);

## Load code.
include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "diffuse-initialisation.jl"));
include(joinpath(LIB_PATH, "data-functions.jl"));
include(joinpath(LIB_PATH, "Parameter.jl"));
include(joinpath(MODELS_PATH, "SEIR", "SEIR_model.jl"));
using StaticArrays
using JLD2
using Dates

## Get arguments.
SEIR_VARIANT = Symbol(args["variant"]);
iterations = args["iterations"];
thin = args["thin"];
burnin = args["burnin"];
npar = args["npar"];
ram_gamma = args["ram_gamma"];
ram_max_eta = args["ram_max_eta"];
aux_target = args["aux_target"];
aux_gamma = args["aux_gamma"];
aux_max_eta = args["aux_max_eta"];
uusimaa_only = args["uusimaa"];
data_max_day = args["data_max_day"];
data_max_month = args["data_max_month"];
download_data = args["download"];
verbose = args["verbose"];
outfolder = args["outfolder"];

## Derive some values.
check_thinning(iterations, thin);
nsim = Int(iterations / thin);
max_date = Date(string("2020-", data_max_month, "-", data_max_day));
verbose && println(string("Maximum date used is ", max_date, "."));

# Population size.
Npop = uusimaa_only ? 1_638_469 : 5_513_000;

# Define output location.
model_id = SEIR_VARIANT == :betabinomial ? "bb" : "nb";
prefix = string("fdi-pg-seir");
if outfolder == ""
    # If outfolder is empty, use this default.
    outfolder = joinpath(RESULTS_PATH, prefix);
end;
outfile = string(prefix, "-", model_id, "-", string(today()), ".jld2"); # Name of output file.
outpath = joinpath(outfolder, outfile);

## Define all model parameters (inits for estimated, and values for fixed).
if SEIR_VARIANT == :betabinomial
    EST_PARAMS = [:log_σ];
    θ = Parameter((r0_max = 10.0,
                   eff = 0.15,
                   a = 1.0/3.0,
                   γ = 1.0/7.0,
                   alpha = 1000.0,
                   log_σ = log(1.0)),
                   estimated = EST_PARAMS);
    θ_dim = length(estimated(θ));
else
    EST_PARAMS = [:log_σ, :logit_p];
    θ = Parameter((r0_max = 10.0,
                   eff = 0.15,
                   a = 1.0/3.0,
                   γ = 1.0/7.0,
                   log_σ = log(1.0),
                   logit_p = 0.0),
                   estimated = EST_PARAMS);
    θ_dim = length(estimated(θ));
end;

rwmkernel = let max_eta = aux_max_eta, gamma = aux_gamma,
                target = aux_target, N = Npop
    function Qkernel(cur::SVector{3, Float64}, aobj)
        # Note: it is assumed cur is floating point but at integer points!
        v = rand(Gaussian(cur, aobj.cov)); # Sample from N(pcur, latest_cov..)

        # Check that sample is within required support.
        @inbounds _e = round(Int, v[1]);
        @inbounds _i = round(Int, v[2]);
        @inbounds _ρ = v[3];
        if _e < 0 || _i < 0 || _i + _e > N
            # Sampled value is not valid, next is cur.
            return cur;
        end
        # Sample is valid, next gets sampled values.
        return SVector{3, Float64}(_e, _i, _ρ);
    end
    ac = AdaptConstants(max_eta = max_eta, gamma = gamma);
    init_Q_e = 50.0; init_Q_i = 50.0; log_init_Q_ρ = -2.0;
    mu = MVector{3, Float64}(init_Q_e, init_Q_i, log_init_Q_ρ);
    adapt_obj = ASWAM(mu, 0.1 * one(MMatrix{3, 3, Float64, 9}),
                      ac = ac, target = target);
    RWMKernel(Qkernel, copy(mu), adapt_obj);
end

"""
The Q for FDI-CPF for the SEIR model.
"""
SEIR_Q! = let rwmkernel = rwmkernel, N = Npop
    function(p::SEIRParticle, data, θ)
        v = rwmkernel.Q(SVector(rwmkernel.x0), rwmkernel.o);

        # Check that sample is inbounds.
        @inbounds _e = convert(Int, v[1]);
        @inbounds _i = convert(Int, v[2]);
        @inbounds _ρ = v[3];
        if _e < 0 || _i < 0 || _i + _e > N
            # Sampled value is not valid, value does not change.
            @inbounds p.e = convert(Int, rwmkernel.x0[1]);
            @inbounds p.i = convert(Int, rwmkernel.x0[2]);
            @inbounds p.ρ = rwmkernel.x0[3];
            p.r = 0;
            p.s = N - p.r - p.i - p.e;
            return nothing;
        end
        # Sampled value is valid.
        p.e = _e;
        p.i = _i;
        p.ρ = _ρ;
        p.r = 0;
        p.s = N - p.r - p.i - p.e;
        nothing;
    end
end

## Construct the diffuse initialisation version of the SEIR model.
# Here, we replace the Mi! of the original model with the auxiliary proposal (Q!).
# lMi is identity, because it is not required in the model.
SEIR_diff = let basemodel = build_SEIR(variant = SEIR_VARIANT), SEIR_Q! = SEIR_Q!
    GenericSSM(SEIRParticle,
               SEIR_Q!, identity,
               basemodel.M!, basemodel.lM,
               basemodel.lGi, basemodel.lG);
end;

## Define data for the model.
# Load the COVID data.
d = get_covid_data(; uusimaa_only = uusimaa_only, download = download_data);
if max_date > maximum(d[!, :Date])
    msg = string("Warning: specified maximum date for data is greater than ",
                 "what exists in data. Using data maximum instead.");
    println(msg);
    println(string("Maximum date in data is ", maximum(d[!, :Date]), "."));
    max_date = maximum(d[!, :Date]);
end
d = filter(r -> r[:Date] <= max_date, d);

# Data passed to functions making up the model.
# `Q` goes here for the time being.
data = (y = d[:, :Cases],
        dt = vcat(1.0, Float64.(getfield.(diff(d[:, :Date]), :value))),
        N = Npop);

## Build the SSM instance: a wrapper for model, storage and data.
ts_length = length(data.y);
storage = ParticleStorage(SEIRParticle, npar, ts_length);
ssm = SSMInstance(SEIR_diff, storage, data, ts_length);

## Build the adaptive MCMC object for updating θ.
log_prior = let SEIR_VARIANT = SEIR_VARIANT
    function(θ)
        p = 0.0;
        p += logpdf(Normal(-2.0, 0.3), θ[:log_σ]);
        if SEIR_VARIANT == :negativebinomial
            p += logpdf(Normal(0.0, 10.0), θ[:logit_p]);
        end
        p;
    end
end

# Function for returning logdensity of the sum of M's and G's.
pg_lp = build_pg_lp(ssm, θ, log_prior; fully_diffuse = true);

# MCMC object for updating parameters.
ram = let θ_dim = θ_dim, ram_gamma = ram_gamma, ram_max_eta = ram_max_eta
    ropt = RAMOptions{Float64, θ_dim}(gamma = ram_gamma, max_eta = ram_max_eta);
    RAM{Float64, θ_dim, θ_dim * θ_dim}(o = ropt, recalculate = true);
end;
θ_mcmc = SimpleMCMCProblem(pg_lp, ram);

## Build a storage object for the output from sampling.
# Output object.
cpfout = DiffuseCPFOutput(SEIRParticle, ts_length = ts_length, nsim = nsim);
θ_out = let θ_dim = θ_dim, nsim = nsim
    [zero(MVector{θ_dim, Float64}) for i in 1:nsim];
end;
out = (cpfout, θ_out);

## Run particle Gibbs with fully diffuse initialisation.
@time fdi_pg!(out, ssm, rwmkernel, θ_mcmc, θ;
              resampling = MultinomialResampling(),
              burnin = burnin, thin = thin);

## Save results.
mkpath(outfolder);
jldopen(outpath, "w") do file
    # Model variant.
    file["variant"] = SEIR_VARIANT;

    # Traces of estimated parameters.
    file["paramnames"] = keys(estimated(θ));
    file["params"] = permutedims(hcat(Vector.(out[2])...));
    file["fixedparams"] = fixed(θ);

    # Simulation settings.
    file["npar"] = npar;
    file["iterations"] = iterations;
    file["nsim"] = nsim;
    file["burnin"] = burnin;
    file["thin"] = thin;

    # The dataset used.
    file["data"] = d;

    # Simulations of state variables.
    file["s"] = map(y -> y.s, out[1].x);
    file["e"] = map(y -> y.e, out[1].x);
    file["i"] = map(y -> y.i, out[1].x);
    file["r"] = map(y -> y.r, out[1].x);
    rho = map(y -> y.ρ, out[1].x);
    file["rho"] = rho;
    file["R0"] = map(x -> get_r0(x, θ), rho);

    # Simulations of x0 state variables.
    file["x0_e"] = map(y -> y.e, out[1].x0);
    file["x0_i"] = map(y -> y.i, out[1].x0);
    file["x0_rho"] = map(y -> y.ρ, out[1].x0);
end
println("Output saved to: $outpath");
