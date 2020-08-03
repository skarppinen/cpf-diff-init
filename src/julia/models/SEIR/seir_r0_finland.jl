# Before running this script, install Julia; I recommend following these instructions:
# http://docs.junolab.org/stable/man/installation/index.html
# After running the first time, turn to "true"
PACKAGES_INSTALLED=true
if !PACKAGES_INSTALLED
    using Pkg
    for p in ("Distributions", "LabelledArrays", "CSV", "DataFrames","Plots",
              "StatsPlots", "FileIO", "Measures",
              PackageSpec(url="https://github.com/mvihola/AdaptiveParticleMCMC.jl.git"))
        Pkg.add(p)
    end
end

using AdaptiveParticleMCMC, LabelledArrays, Distributions, CSV, Statistics, DataFrames, Dates, LinearAlgebra, FileIO

# Sampling parameters:
N = 64;     # Number of particles
n = 110_000; # Number of PMCMC iterations
burnin = 10_000;
thin = 10;  # Thinning (because paths take memory...)

population = 5_513_000

lockdown_start_date = Date("2020-03-16")
uusimaa_close_date = Date("2020-03-24")
restaurants_closed_date = Date("2020-04-04")
uusimaa_open_date = Date("2020-04-15")

# Data set:
case = "Finland"
if case == "Finland"
    data = CSV.read(joinpath(@__DIR__,"thl_data.csv"))
    date_name = :Aika; cases_name = Symbol("Kaikki Alueet")
    start_date = Date("2020-03-01") # After start of epidemic
    end_date = maximum(data[:, date_name]) - Dates.Day(3) # Reporting delay...
    sampling_effort = 0.15
elseif case == "HUS"
    population = 1_638_469
    data = CSV.read(joinpath(@__DIR__,"thl_data.csv"))
    date_name = :Aika; cases_name = Symbol("Helsingin ja Uudenmaan SHP")
    start_date = Date("2020-03-01") # After start of epidemic
    end_date = maximum(data[:, date_name]) - Dates.Day(3) # Reporting delay...
    sampling_effort = 0.15
end

# Selected dates only:
data = filter(r -> (start_date <= r[date_name] <= end_date), data)
T = size(data)[1]

# Some constants for seir_r0_model.jl
const seir_const = (
incubation_time = 3.0, # Mean incubation time (days)
recovery_time = 7.0,  # Mean recovery time (days)
R0_max = 10.0,        # Maximum value of basic reproduction rate R0
sampling_effort = sampling_effort, # Sampling effort -- mean number of observed vs. true cases
population = population,
date_name = date_name,
cases_name = cases_name
)


##################################################
# Load the model definitions
include("seir_r0_model_negbin.jl")
#include("seir_r0_model.jl");#_negbin.jl")

# Set up the SMC state (for adaptive_pg, which uses SequentialMonteCarlo.jl internally)
state = SMCState(T, N, SEIRParticle, SEIRScratch, set_param!, lG_SEIR, M_SEIR!, lM_SEIR)

# Initial (transformed) parameter vector
i0 = data[1,seir_const.cases_name]/(seir_const.sampling_effort*(1.0-exp(-1/seir_const.recovery_time)))
theta0 = LVector(logit_sigma = 0.0, e = i0, i = i0, trans_R0_init=0.0, logit_p = 10.0)

# Run the adaptive particle Gibbs (with backward sampling)
out = adaptive_pg(theta0, prior, state, n; b = burnin,
                  show_progress=2, save_paths=true, thin=thin);

# Visualisation:
include("seir_r0_plots.jl")
function add_restrictions!(p)
    vspan!(p, [lockdown_start_date, data.Aika[end]], color=:yellow, alpha=0.2)
    vspan!(p, [restaurants_closed_date, data.Aika[end]], color=:red, alpha=0.2)
    vline!(p, [data.Aika[1]], alpha=0.0) # Fixes y axis (workaround for a bug in vspan)
end
p_infected = path_plot(out, :i); plot!(p_infected, ylabel="Infected"); add_restrictions!(p_infected)
p_r0 = path_plot(out, :trans_R0, R0_transform); plot!(p_r0, ylabel="R0"); add_restrictions!(p_r0)
p_pred = posterior_predictive(out); plot(p_pred, title="Data (blue) and posterior predictive samples (red)")
#p_theta, p_infected, p_r0 = show_out(out)
#save("output/$(case)_$(end_date).jld2", "out", out) # Save output
savefig(p_infected, "$(case)_$(end_date)_infected.pdf")
savefig(p_r0, "$(case)_$(end_date)_R0.pdf")
savefig(p_pred, "$(case)_$(end_date)_predictive.pdf")
