## Script to run the diffuse particle Gibbs (DPG-BS) for the noisy RW or SV
# model with various parameter values and numbers of particles.
# Run `julia run-dpg-cpf-noisyar-sv.jl --help` for help.

## Parse arguments.
include("../../../config.jl");
include(joinpath(CONFIG_PATH, "parse-settings.jl"));
args = parse_dpg_cpf_noisyar_sv_args(ARGS);

## Simulation settings.
modelname = args["model"];
@assert modelname in ("SV", "NOISYAR") "`modelname` must be SV or NOISYAR.";
modelpath = joinpath(MODELS_PATH, modelname, modelname * "_model.jl");
ts_length = 50;
nsim = args["iterations"];
burnin = args["burnin"];
thin = args["thin"];
n_particle_vec = args["npar"];
sigma_x1_vec = args["sigma_x1"]; #[2.0, 5.0, 10.0, 20.0, 50.0];
sigma_y_vec = args["sigma_y"]; #[0.1, 0.2, 0.5, 0.7, 1.0];
sigma_x_vec = args["sigma_x"]; #[0.1, 0.2, 0.5, 0.7, 1.0, 2.0];
fully_diffuse = args["fully_diffuse"];
gamma = args["gamma"]; # Adaptation parameter gamma.
max_eta = args["max_eta"]; # Adaptation parameter max_eta.
verbose = args["verbose"];
outfolder = args["outfolder"];

## Load code and model.
include(joinpath(LIB_PATH, "RAM.jl"));
include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "diffuse-initialisation.jl"));
include(joinpath(LIB_PATH, "Parameter.jl"));
include(joinpath(LIB_PATH, "data-functions.jl"));
include(modelpath);
using Dates: today
using JLD2

if fully_diffuse
    # If this is the fully diffuse case, set sigma_x1 to Inf.
    sigma_x1_vec = [Inf];
end
n_combinations = prod(map(length, (sigma_x1_vec, sigma_x_vec,
                                   sigma_y_vec, n_particle_vec)));
## Define output location.
# If outfolder is empty, use a default.
if fully_diffuse
    prefix = "dpg-cpf-" * modelname;
    outfile = string(prefix * "-", string(today()), ".jld2");
else
    prefix = "dnpg-cpf-" * modelname;
    outfile = string(prefix * "-", string(today()), ".jld2");
end
outfolder = outfolder == "" ? joinpath(RESULTS_PATH, prefix) : outfolder;
outpath = joinpath(outfolder, outfile);

## Define initial parameters.
θ = Parameter((x1_mean = 0.0,
               log_sigma_x1 = log(0.0),
               log_sigma_x = log(1.0),
               log_sigma_y = log(1.0),
               log_rho = log(1.0)));

# Parameter object for simulations.
θsim = Parameter((x1_mean = θ[:x1_mean],
                  log_sigma_x1 = log(0.0),
                  log_sigma_x = θ[:log_sigma_x],
                  log_sigma_y = θ[:log_sigma_y],
                  log_rho = θ[:log_rho]));
tracked_params = (:x1_mean, :log_sigma_x, :log_sigma_y, :log_rho);

## Define model.
basemodel = eval(Symbol(modelname));
basemodel_simobs = eval(Symbol(modelname * "_simobs"));
ptyp = ptype(basemodel);
model = shift(basemodel);

## Define data container, temporaries for data simulation,
#  and output from simulations.
y = zeros(ts_length); # Container for observations.
ytmp = [zeros(1) for i in 1:ts_length];
xtmp = [ptyp() for i in 1:ts_length];
data = (y = y, x1 = ptyp(0.0));
cpfout = [ptyp() for i in 1:ts_length, j in 1:nsim];

## Run simulations.
model_id = modelname;
verbose && println("Starting simulation..");
verbose && println("Model is $prefix");
mkpath(outfolder);
iteration = 1;
out = Vector{Any}(undef, 0);
for npar in n_particle_vec
    global iteration;

    # Build resampling and storage.
    resampling = MultinomialResampling();
    stor = ParticleStorage(ptyp, npar, ts_length);
    #GC.gc(); # Run garbage collector to flush unused pointers.

    # Build model instance.
    ssm = SSMInstance(model, stor, data, ts_length - 1);

    # Define logposterior function for RAM.
    lp = let basemodel = basemodel, ssm = ssm, data = data, θ = θ,
             fully_diffuse = fully_diffuse, x = ptyp()
        function(y::AVec{<: AFloat})
            copy!(x, y);
            xref = ssm.storage.X[ssm.storage.ref[1], 1];
            s = basemodel.lGi(x, data, θ) +
                basemodel.lM(xref, x, 2, data, θ) +
                basemodel.lG(x, xref, 2, data, θ);
            if !fully_diffuse
                s += basemodel.lMi(x, data, θ);
            end
            s;
        end
    end;

    for sigma_x in sigma_x_vec, sigma_y in sigma_y_vec, sigma_x1 in sigma_x1_vec
        verbose && (start = time();)

        # Set parameters to current values.
        θ[:log_sigma_x] = log(sigma_x);
        θ[:log_sigma_y] = log(sigma_y);
        θ[:log_sigma_x1] = log(sigma_x1);
        copy!(θsim, θ, tracked_params);

        # Simulate data and latent state.
        Random.seed!(DATA_SIM_SEED);
        simulate!(ytmp, xtmp, basemodel, basemodel_simobs, data, θsim);
        for i in 1:length(ytmp) data.y[i] = ytmp[i][1]; end
        data.x1.x = xtmp[1].x;

        # Define MCMC problem object.
        r = let gamma = gamma, max_eta = max_eta
            ropt = RAMOptions{Float64, 1}(gamma = gamma, max_eta = max_eta);
            RAM{Float64, 1, 1}(o = ropt, recalculate = true);
        end;
        θ_mcmc = SimpleMCMCProblem(lp, r);

        # Run diffuse particle Gibbs.
        dpg_bs!(cpfout, ssm, θ_mcmc, θ,
                 resampling = resampling,
                 burnin = burnin,
                 thin = thin);
        verbose && print_progress(time() - start, iteration, n_combinations);

        # Extract results.
        verbose && (start = time();)
        push!(out, (model = model_id,
                    burnin = burnin,
                    thin = thin,
                    gamma = gamma,
                    max_eta = max_eta,
                    nsim = nsim,
                    ts_length = ts_length,
                    fully_diffuse = fully_diffuse,
                    keyparams = ("npar", "sigma_x", "sigma_y", "sigma_x1"),
                    npar = npar,
                    sigma_x = sigma_x,
                    sigma_y = sigma_y,
                    sigma_x1 = sigma_x1,
                    x1 = map(y -> y.x, cpfout[1, :])));
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
