## Script to run a small simulation experiment to illustrate how the
# performance of CPF + BS deteriorates as the initial distribution becomes
# increasingly diffuse.

include("../../../config.jl");
include(joinpath(CONFIG_PATH, "parse-settings.jl"));
args = parse_diffinit_poor_mixing_args(ARGS);

include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "Parameter.jl"));
include(joinpath(LIB_PATH, "asymptotic-variance.jl"));
include(joinpath(MODELS_PATH, "NOISYAR", "NOISYAR_model.jl"));
using Statespace
using Random
using StatsBase
using JLD2
using Dates: today

Random.seed!(3);
log_sigma_x1_vec = [log(10.0), log(100.0), log(1000.0)];

outfolder = args["outfolder"];
filename = "diffinit-poor-mixing-example-summary-" * string(today()) * ".jld2";
if outfolder == ""
    outpath = joinpath(RESULTS_PATH, "summaries",
                       filename);
else
    outpath = joinpath(outfolder, filename);
end

## Define parameters.
# Parameter object for running CPF-BS.
θ = Parameter((log_sigma_x = log(0.5),
               log_rho = log(0.8),
               log_sigma_y = log(0.5),
               x1_mean = 0.0,
               log_sigma_x1 = NaN));

# Parameters for simulating the dataset.
θsim = Parameter((log_sigma_x = θ[:log_sigma_x],
                 log_rho = θ[:log_rho],
                 log_sigma_y = θ[:log_sigma_y],
                 x1_mean = θ[:x1_mean],
                 log_sigma_x1 = -Inf));

## Simulate dataset.
T = 50;
y_tmp = [zeros(1) for i in 1:T];
data = (y = zeros(T),);
simulate!(y_tmp, NOISYAR, NOISYAR_simobs, nothing, θsim);
for i in 1:length(data.y) data.y[i] = y_tmp[i][1]; end

## Run CPF-BS.
out = Vector{Any}(undef, 0);
N = 16;
M = 6000;
burnin = 1000;
thin = 1;

# Build model.
storage = ParticleStorage(NoisyarParticle, N, T);
inst = SSMInstance(NOISYAR, storage, data, T);
cpfout = [NoisyarParticle() for t in 1:T, i in 1:M];
resampling = MultinomialResampling();

# Run simulation.
for log_sigma_x1 in log_sigma_x1_vec
    # Set parameters.
    θ[:log_sigma_x1] = log_sigma_x1;

    # CPF-BS.
    cpf_bs!(cpfout, inst, θ; resampling = resampling, burnin = burnin,
            thin = thin);

    # Save results.
    push!(out, (x1 = map(y -> y.x, cpfout[1, :]),
                sigma_x1 = exp(log_sigma_x1),
                sigma_x = exp(θ[:log_sigma_x]),
                sigma_y = exp(θ[:log_sigma_y]),
                rho = exp(θ[:log_rho]),
                M = M,
                burnin = burnin,
                N = N));
end

## Save results.
mkpath(outfolder);
jldopen(outpath, "w") do file
    file["out"] = out;
end
println(string("Output saved to: ", outpath, "."));
