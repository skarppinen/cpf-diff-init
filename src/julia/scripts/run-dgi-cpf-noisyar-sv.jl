## Run the simulations for the noisy RW model or SV model with the diffuse normal
# initialisation with various parameter combinations and numbers of
# particles.
# Run `julia run-dgi-cpf-noisyar-sv.jl --help` for help.

## Parse arguments.
include("../../../config.jl");
include(joinpath(CONFIG_PATH, "parse-settings.jl"));
args = parse_dgi_cpf_noisyar_sv_args(ARGS);

## Get sampling settings.
modelname = args["model"];
@assert modelname in ("SV", "NOISYAR") "`modelname` must be SV or NOISYAR.";
modelpath = joinpath(MODELS_PATH, modelname, modelname * "_model.jl");
ts_length = 50; # Length of time series.
n_particle_vec = args["npar"];
nsim = args["iterations"]; # Total simulations after burnin.
burnin = args["burnin"];
thin = args["thin"];
nreps = args["nreps"];
target = args["target"];
beta_init = args["beta_init"];
sigma_x1_vec = args["sigma_x1"];
sigma_y_vec = args["sigma_y"];
sigma_x_vec = args["sigma_x"];
verbose = args["verbose"];
outfolder = args["outfolder"];
noadapt = args["noadapt"];
n_combinations = prod(map(length, (sigma_x1_vec, sigma_x_vec, sigma_y_vec, 1:nreps,
                                   n_particle_vec)));

## Load code and model.
include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "diffuse-initialisation.jl"));
include(joinpath(LIB_PATH, "Parameter.jl"));
include(joinpath(LIB_PATH, "data-functions.jl"));
include(modelpath);
using Dates: today
using JLD2

## Define output location.
prefix = if noadapt
    "dgi-cpf-betafix-" * modelname;
else
    "dgi-cpf-" * modelname;
end
if outfolder == ""
    # Default outfolder.
    outfolder = joinpath(RESULTS_PATH, prefix);
end
outfile = if noadapt
    string(prefix, "-", string(today()),
           "-beta-", replace(string(beta_init), '.' => '_'), ".jld2");
else
    string(prefix, "-", string(today()),
           "-target", round(Int, 100 * target), ".jld2");
end
outpath = joinpath(outfolder, outfile);

## Define model and random walk kernel for sampling x0.
basemodel = eval(Symbol(modelname));
ptyp = ptype(basemodel);
basemodel_simobs = eval(Symbol(modelname * "_simobs"));
statedim = dimension(basemodel);

# Define RWMKernel for the problem. This samples x0 | x1.
rwmkernel = let target = target, beta_init = beta_init, noadapt = noadapt
    function Qkernel(cur::SVector{1, Float64}, acn)
        beta = invlogit(acn.logit_beta[]);
        mu = acn.mu[1];
        sd = sqrt(acn.cov[1, 1]);

        z = rand(Normal(0.0, sd));
        out = sqrt(1.0 - beta * beta) * (cur[1] - mu) + beta * z + mu;
        SVector{1, Float64}(out);
    end
    if noadapt
        ac = AdaptConstants(n_adapt = 0);
    else
        ac = AdaptConstants();
    end
    acn = AdaptiveCN(MVector{1, Float64}(-Inf), MMatrix{1, 1, Float64, 1}(-Inf);
                     beta = beta_init, target = target, ac = ac);
    RWMKernel(Qkernel, zeros(1), acn);
end

# Q for SSM. This samples x1 | x0. This function reads a pointer to `rwmstate`.
Mi!_diffuse = let rwmkernel = rwmkernel
    function Q!(p::ptyp, data, θ)
        beta = invlogit(rwmkernel.o.logit_beta[]);
        mu = rwmkernel.o.mu[1];
        sd = sqrt(rwmkernel.o.cov[1, 1]);
        _x = rwmkernel.x0[1];

        z = rand(Normal(0.0, sd));
        p.x = sqrt(1.0 - beta * beta) * (_x - mu) + beta * z + mu;
        nothing;
    end
end
model = GenericSSM(ptyp,
                   Mi!_diffuse, x -> x,
                   basemodel.M!, basemodel.lM,
                   basemodel.lGi, basemodel.lG);

## Define data and additional objects passed to the filter.
y = zeros(ts_length); # Container for observations.
y_tmp = [zeros(1) for i in 1:ts_length];
x_tmp = [ptyp() for i in 1:ts_length];

# Data object.
data = (y = y,);

## Build initial parameter object.
θ = Parameter((x1_mean = 0.0,
               log_sigma_x1 = -Inf,
               log_sigma_x = -Inf,
               log_sigma_y = -Inf,
               log_rho = log(1.0)));

# Parameter object for simulations.
θsim = Parameter((x1_mean = θ[:x1_mean],
                  log_sigma_x1 = log(0.0),
                  log_sigma_x = θ[:log_sigma_x],
                  log_sigma_y = θ[:log_sigma_y],
                  log_rho = θ[:log_rho]));
# Fields in \thetasim that should be the same as in \theta.
tracked_params = (:x1_mean, :log_sigma_x, :log_sigma_y, :log_rho);

## Run simulations.
model_id = modelname;
cpfout = DiffuseCPFOutput(ptyp, ts_length = ts_length, nsim = nsim);

verbose && println("Starting simulation..");
verbose && println("Model is $prefix");
mkpath(outfolder);
out = Vector{Any}(undef, 0);
iteration = 1;
for npar in n_particle_vec
    global iteration;
    # Build resampling and storage.
    resampling = MultinomialResampling();
    storage = ParticleStorage(ptyp, npar, ts_length);

    # Build model instance.
    inst = SSMInstance(model, storage, data, ts_length);
    for sigma_x1 in sigma_x1_vec, sigma_x in sigma_x_vec, sigma_y in sigma_y_vec

        verbose && (start = time();)

        # Set parameters to current values.
        θ[:log_sigma_x1] = log(sigma_x1);
        θ[:log_sigma_x] = log(sigma_x);
        θ[:log_sigma_y] = log(sigma_y);
        copy!(θsim, θ, tracked_params);

        # Simulate a dataset with the parameters and record it to the data.
        # Note that the SSMInstance "inst" holds a pointer to "data",
        # so this is enough.
        Random.seed!(DATA_SIM_SEED);
        simulate!(y_tmp, x_tmp, basemodel, basemodel_simobs, data, θsim);
        for i in 1:length(y_tmp)
            @inbounds data.y[i] = y_tmp[i][1];
        end

        for rep in 1:nreps
            # Initialise RWM kernel.
            rwmkernel.x0[1] = x_tmp[1].x;
            rwmkernel.o.target[] = target;
            rwmkernel.o.logit_beta[] = logit(beta_init);

            # Set the desired initial distribution for the model.
            rwmkernel.o.mu[1] = θ[:x1_mean];
            rwmkernel.o.cov[1, 1] = exp(2.0 * θ[:log_sigma_x1]);

            # Run simulation with diffuse normal initialisation.
            aai_cpf!(cpfout, inst, rwmkernel, θ;
                     resampling = resampling,
                     thin = thin,
                     burnin = burnin);
            verbose && print_progress(time() - start, iteration, n_combinations);

            # Extract results.
            verbose && (start = time();)
            base_nt = (model = model_id,
                        burnin = burnin,
                        nsim = nsim,
                        thin = thin,
                        ts_length = ts_length,
                        target = target,
                        last_beta = invlogit(rwmkernel.o.logit_beta[]),
                        keyparams = ("npar", "sigma_x1", "sigma_x", "sigma_y"),
                        npar = npar,
                        sigma_x1 = sigma_x1,
                        sigma_x = sigma_x,
                        sigma_y = sigma_y,
                        x0 = map(y -> y.x, cpfout.x0),
                        x1 = map(y -> y.x, cpfout.x[1, :]));
            if nreps > 1
                out_nt = merge(base_nt, (rep = rep,));
            else
                out_nt = base_nt;
            end
            push!(out, out_nt);
            iteration += 1;
        end
    end
end
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
