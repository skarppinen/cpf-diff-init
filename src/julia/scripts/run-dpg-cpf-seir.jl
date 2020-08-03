## Script to run the SEIR simulations with the DPG-BS.
# Run `julia run-dpg-cpf-seir.jl --help` for help.

include("../../../config.jl")
include(joinpath(CONFIG_PATH, "parse-settings.jl"));
args = parse_dpg_cpf_seir_args(ARGS);

using AdaptiveParticleMCMC
using LabelledArrays
using Distributions
using CSV
using Statistics
using DataFrames
using Dates
using LinearAlgebra
using FileIO
using JLD2

## Get arguments.
N = args["npar"];     # Number of particles
burnin = args["burnin"]; # Number of burnin iterations.
n = args["iterations"] + burnin; # Number of total PMCMC iterations
thin = args["thin"];  # Thinning.
uusimaa_only = args["uusimaa"];
data_max_day = args["data_max_day"];
data_max_month = args["data_max_month"];
download_data = args["download"];
outfolder = args["outfolder"];
verbose = args["verbose"];

## Some derived values.
max_date = Date(string("2020-", data_max_month, "-", data_max_day));
verbose && println(string("Maximum date used is ", max_date, "."));

# Define output location.
model_id = "dpg-cpf-seir";
if outfolder == ""
    # If outfolder is empty, use this default.
    outfolder = joinpath(RESULTS_PATH, "seir-dpg");
end;
outfile = string(model_id, "-", string(today()), ".jld2"); # Name of output file.
outpath = joinpath(outfolder, outfile);

## Load dataset.
include(joinpath(LIB_PATH, "data-functions.jl"));
data = get_covid_data(; uusimaa_only = uusimaa_only, download = download_data);
date_name = :Date; cases_name = :Cases;
if max_date > maximum(data[!, date_name])
    msg = string("Warning: specified maximum date for data is greater than ",
                 "what exists in data. Using data maximum instead.");
    println(msg);
    println(string("Maximum date in data is ", maximum(data[!, date_name]), "."));
    max_date = maximum(data[!, date_name]);
end
data = filter(r -> r[:Date] <= max_date, data);
start_date = minimum(data[:, date_name]);
end_date = maximum(data[:, date_name]);
T = nrow(data);

# Some constants.
Npop = uusimaa_only ? 1_638_469 : 5_513_000;
const seir_const = (
    incubation_time = 3.0, # Mean incubation time (days)
    recovery_time = 7.0,  # Mean recovery time (days)
    R0_max = 10.0,        # Maximum value of basic reproduction rate R0
    sampling_effort = 0.15, # Sampling effort -- mean number of observed vs. true cases
    population = Npop,
    date_name = date_name,
    cases_name = cases_name);


## Load the model definitions.
include(joinpath(MODELS_PATH, "SEIR", "seir_r0_model_negbin.jl"));

# Set up the SMC state (for adaptive_pg, which uses SequentialMonteCarlo.jl internally)
state = SMCState(T, N, SEIRParticle, SEIRScratch, set_param!,
                 lG_SEIR, M_SEIR!, lM_SEIR)

# Initial (transformed) parameter vector
i0 = data[1,seir_const.cases_name]/(seir_const.sampling_effort*(1.0-exp(-1/seir_const.recovery_time)))
theta0 = LVector(log_sigma = log(1.0), e = i0, i = i0,
                 trans_R0_init = 0.0, logit_p = 0.0)

# Run the adaptive particle Gibbs.
if verbose
    show_progress = 2;
else
    show_progress = false;
end

@time out = adaptive_pg(theta0, prior, state, n; b = burnin,
                  show_progress = show_progress,
                  save_paths = true, thin = thin);

## Save output.
# Get a matrix of simulations from the output object.
function get_trajectories(out, field::Symbol)
    hcat(map(x -> map(y -> getfield(y, field), x), out.X)...);
end

mkpath(outfolder);
jldopen(outpath, "w") do file

    # Simulation settings.
    file["npar"] = N;
    file["burnin"] = burnin;
    file["iterations"] = n;
    file["nsim"] = n - burnin;
    file["thin"] = thin;

    # The dataset used.
    file["data"] = data;

    # Traces of estimated parameters.
    file["paramnames"] = keys(theta0);
    file["params"] = permutedims(out.Theta);
    file["fixedparams"] = (r0_max = seir_const[:R0_max],
                           eff = seir_const[:sampling_effort],
                           γ = 1.0 / seir_const[:recovery_time],
                           a = 1.0 / seir_const[:incubation_time]);

    # Simulations of state variables.
    file["s"] = get_trajectories(out, :s);
    file["e"] = get_trajectories(out, :e);
    file["i"] = get_trajectories(out, :i);
    file["r"] = get_trajectories(out, :r);
    trans_R0 = get_trajectories(out, :trans_R0);
    file["trans_R0"] = trans_R0;
    file["R0"] = R0_transform.(trans_R0);

end
println("Output saved to: $outpath");
