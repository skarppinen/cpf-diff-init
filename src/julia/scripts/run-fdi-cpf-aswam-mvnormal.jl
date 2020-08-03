## Script to run the multivariate normal model simulations with varying parameters
# with the FDI-CPF and ASWAM adaptation.
# Run `julia run-fdi-cpf-aswam-mvnormal.jl --help` for help.
include("../../../config.jl");

## Parse arguments.
include(joinpath(CONFIG_PATH, "parse-settings.jl"));
args = parse_fdi_cpf_mvnormal_args(ARGS);

include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "diffuse-initialisation.jl"));
include(joinpath(LIB_PATH, "Parameter.jl"));
include(joinpath(MODELS_PATH, "MVNORMAL", "MVNORMAL_model.jl"));

using StaticArrays
using Dates: today
using JLD2

## Sampling settings.
ts_length = 1; # Length of time series.
statedim_vec = args["statedim"];
n_particle_vec = args["npar"];
nsim = args["iterations"]; # Total simulations after burnin.
burnin = args["burnin"];
thin = args["thin"];
gamma = args["gamma"]; # Adaptation parameter gamma.
max_eta = args["max_eta"]; # Adaptation parameter max_eta.
sigma_vec = args["sigma"];
target = args["target"];
verbose = args["verbose"];
outfolder = args["outfolder"];

## Define output location.
if outfolder == ""
    # If outfolder is empty, use this default.
    outfolder = joinpath(RESULTS_PATH, "fdi-cpf-aswam-mvnormal");
end;
outfile = string("fdi-cpf-aswam-mvnormal-",
                 string(today()),
                 "-target",
                 round(Int, 100 * target),
                 ".jld2"); # Name of output file.
outpath = joinpath(outfolder, outfile);

## Grid of values for parameters.
n_combinations = prod(map(length, (sigma_vec, n_particle_vec, statedim_vec)));

## Build initial parameter object.
pt = (log_sigma = -Inf,);
θ = Parameter(pt);

## Run simulations
iteration = 1;
resampling = MultinomialResampling();
mkpath(outfolder);
out = Vector{Any}(undef, 0);
for statedim in statedim_vec
    global iteration;

    ## Build model.
    rwmkernel = let max_eta = max_eta, gamma = gamma, target = target, statedim = statedim
        function Qkernel(cur::SVector, aswam)
             rand(Gaussian(cur, aswam.cov));
        end
        ac = AdaptConstants(max_eta = max_eta, gamma = gamma);
        adapt_obj = ASWAM(zero(MVector{statedim, Float64}),
                          one(MMatrix{statedim, statedim, Float64, statedim * statedim}),
                          ac = ac, target = target);
        RWMKernel(Qkernel, zeros(statedim), adapt_obj);
    end
    MVNORMAL_Q! = let rwmkernel = rwmkernel
        function(p::FloatParticle, data, θ)
            x = rwmkernel.Q(SVector(rwmkernel.x0), rwmkernel.o);
            for i in eachindex(p.x)
                @inbounds p.x[i] = x[i];
            end
            nothing;
        end
    end
    basemodel = build_MVNORMAL(statedim);
    model = GenericSSM(ptype(basemodel),
                       MVNORMAL_Q!, basemodel.lMi,
                       basemodel.M!, basemodel.lM,
                       basemodel.lGi, basemodel.lG);

    ## Run simulations.
    verbose && println("Starting simulation, dimension $statedim..");
    for npar in n_particle_vec

        # Build resampling and storage.
        storage = ParticleStorage(ptype(basemodel), npar, ts_length);

        # Build diffuse CPF output.
        cpfout = DiffuseCPFOutput(ptype(basemodel); nsim = nsim,
                                  ts_length = ts_length);

        # Build model instance.
        inst = SSMInstance(model, storage, nothing, ts_length);

        for sigma in sigma_vec
            verbose && (start = time();)
            if verbose
                println("State dimension: ", statedim, ", sigma: ", sigma, ", N: ", npar, ".");
            end

            # Set parameters to current values.
            θ[:log_sigma] = log(sigma);
            rwmkernel.x0 .= zero(typeof(rwmkernel.x0));

            # Run simulation with diffuse normal initialisation.
            aai_cpf!(cpfout, inst, rwmkernel, θ;
                     resampling = resampling,
                     burnin = burnin,
                     thin = thin);

            # Extract results.
            push!(out, (model = "MVNORMAL",
                        nsim = nsim,
                        burnin = burnin,
                        thin = thin,
                        npar = npar,
                        gamma = gamma,
                        max_eta = max_eta,
                        ts_length = ts_length,
                        target = target,
                        keyparams = ("statedim", "npar", "sigma"),
                        statedim = statedim,
                        sigma = sigma,
                        x1 = map(y -> SVector{statedim, Float64}(y), cpfout.x[1, :]),
                        last_log_delta = rwmkernel.o.log_delta[],
                        last_sigmamat = copy(rwmkernel.o.sigma)));
            if verbose
                elapsed = round(time() - start, digits = 1);
                msg = string("Finished parameter combination ", iteration, "/",
                             n_combinations, " (",
                             round(iteration / n_combinations * 100, digits = 1),
                             "%), (", elapsed, " s)");
                println(msg);
            end
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
