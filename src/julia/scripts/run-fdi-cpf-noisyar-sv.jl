## Script to run the noisy RW or SV simulations with varying parameters
# with the adaptive FDI-CPF.
# Run `julia run-fdi-cpf-noisyar-sv.jl --help` for help.
include("../../../config.jl");

## Parse arguments.
include(joinpath(CONFIG_PATH, "parse-settings.jl"));
args = parse_fdi_cpf_noisyar_sv_args(ARGS);

## Get sampling settings.
modelname = args["model"]; # Name of model.
@assert modelname in ("SV", "NOISYAR") "`modelname` must be 'SV' or 'NOISYAR'.";
modelpath = joinpath(MODELS_PATH, modelname, modelname * "_model.jl");
adaptation = args["adaptation"]
@assert adaptation in ("am", "aswam") "`adaptation` must be 'am' or 'aswam'";
ts_length = 50; # Length of time series.
n_particle_vec = args["npar"]; # Particle counts.
nsim = args["iterations"]; # Total simulations after burnin.
burnin = args["burnin"]; # Number of burnin iterations.
thin = args["thin"];
gamma = args["gamma"]; # Adaptation parameter gamma.
max_eta = args["max_eta"]; # Adaptation parameter max_eta.
sigma_y_vec = args["sigma_y"]; # Sigma y values.
sigma_x_vec = args["sigma_x"]; # Sigma x values.
target = args["target"]; # Target value for MCMC.
verbose = args["verbose"];
outfolder = args["outfolder"];

## Load model code and model.
include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "diffuse-initialisation.jl"));
include(joinpath(LIB_PATH, "Parameter.jl"));
include(joinpath(LIB_PATH, "data-functions.jl"));
include(modelpath);
using StaticArrays
using Dates: today
using JLD2

## Some derived values.
prefix = "fdi-cpf-" * adaptation * "-" * modelname;
n_combinations = prod(map(length, (sigma_x_vec, sigma_y_vec, n_particle_vec)));

## Define output location.
if outfolder == ""
    # If outfolder is empty, use this default.
    outfolder = joinpath(RESULTS_PATH, prefix);
end;
if adaptation == "aswam"
    outfile = string(prefix, "-", string(today()), "-target",
                     round(Int, 100 * target),
                     ".jld2");
else
    outfile = string(prefix, "_", string(today()),
                     ".jld2");
end
outpath = joinpath(outfolder, outfile);

## Build initial parameter object.
pt = (x1_mean = 0.0,
      log_sigma_x1 = -Inf,
      log_sigma_x = -Inf,
      log_sigma_y = -Inf,
      log_rho = log(1.0));
θ = Parameter(pt);

## Define model.
basemodel = eval(Symbol(modelname));
basemodel_simobs = eval(Symbol(modelname * "_simobs"));
ptyp = ptype(basemodel);
statedim = dimension(basemodel);

rwmkernel = let max_eta = max_eta, gamma = gamma, target = target, adaptation = adaptation
    function Qkernel(cur::SVector{1, Float64}, aobj)
        x = rand(Normal(cur[1], sqrt(aobj.cov[1, 1])));
        SVector{1, Float64}(x);
    end
    ac = AdaptConstants(max_eta = max_eta, gamma = gamma);

    if adaptation == "aswam"
        adapt_obj = ASWAM(zeros(MVector{1, Float64}), ones(MMatrix{1, 1, Float64, 1}),
                          ac = ac, target = target);
    else
        # AM with default scaling. c = 2.38 ^ 2 / d.
        adapt_obj = AM(zeros(MVector{1, Float64}), ones(MMatrix{1, 1, Float64, 1}),
                       ac);
    end
    RWMKernel(Qkernel, zeros(1), adapt_obj);
end
Mi!_diffuse = let rwmkernel = rwmkernel
    function Q!(p, data, θ)
        p.x = rand(Normal(rwmkernel.x0[1], sqrt(rwmkernel.o.cov[1, 1])));
        nothing;
    end
end
model = GenericSSM(ptyp, Mi!_diffuse, identity,
                   basemodel.M!, basemodel.lM,
                   basemodel.lGi, basemodel.lG);

## Build container for data.
y = zeros(ts_length); # Container for observations.
y_tmp = [zeros(1) for i in 1:ts_length];

data = (y = y,); # Data container.

# Note: this argument is never used, but must be set so that we can call
# the same simulate! method as in the other scripts, and therefore we will
# have the same datasets with same parameters.
x_tmp = [ptyp() for i in 1:ts_length];

## Run simulations.
model_id = modelname;
cpfout = DiffuseCPFOutput(ptyp; ts_length = ts_length,
                          nsim = nsim);

mkpath(outfolder);
verbose && println("Starting simulation..");
verbose && println("Model is $prefix");
out = Vector{Any}(undef, 0);
iteration = 1;
for npar in n_particle_vec
    global iteration;
    # Build resampling and storage.
    resampling = MultinomialResampling();
    storage = ParticleStorage(ptyp, npar, ts_length);

    # Build model instance.
    inst = SSMInstance(model, storage, data, ts_length);

    for sigma_x in sigma_x_vec, sigma_y in sigma_y_vec
        verbose && (start = time();)

        # Set parameters to current values.
        θ[:log_sigma_x] = log(sigma_x);
        θ[:log_sigma_y] = log(sigma_y);

        # Simulate a dataset with the parameters and record it to the data.
        # Note that the model instance holds a pointer to "data",
        # so this is enough.
        Random.seed!(DATA_SIM_SEED);
        simulate!(y_tmp, x_tmp, basemodel, basemodel_simobs, data, θ);
        for i in 1:length(y_tmp)
            data.y[i] = y_tmp[i][1];
        end

        # Initialise x0.
        rwmkernel.x0[1] = θ[:x1_mean];

        # Run simulation with diffuse normal initialisation.
        aai_cpf!(cpfout, inst, rwmkernel, θ;
                resampling = resampling,
                burnin = burnin);
        verbose && print_progress(time() - start, iteration, n_combinations);

        # Extract results.
        verbose && (start = time();)
        res = (model = model_id,
               adaptation = adaptation,
               burnin = burnin,
               thin = thin,
               nsim = nsim,
               gamma = gamma,
               max_eta = max_eta,
               ts_length = ts_length,
               keyparams = ("npar", "sigma_x", "sigma_y"),
               npar = npar,
               sigma_x = sigma_x,
               sigma_y = sigma_y,
               sigma_x1 = Inf,
               x0 = map(y -> y.x, cpfout.x0),
               x1 = map(y -> y.x, cpfout.x[1, :]));
        if adaptation == "aswam"
            push!(out, merge(res,
                 (target = target,
                  last_log_delta = rwmkernel.o.log_delta[],
                  last_sigmamat = rwmkernel.o.sigma[1, 1])));
        else
            push!(out, merge(res,
                             (last_cov = rwmkernel.o.cov[1, 1],
                              )));
        end
        iteration += 1;
    end
end;
verbose && println("Simulations finished.")
verbose && (start = time();)
jldopen(outpath, "w") do file
    file["out"] = out;
end
if verbose
    msg = string("Saving to disk took ", round(time() - start, digits = 1), " s.");
    println(msg);
end
verbose && println(string("Output saved to ", outpath));
