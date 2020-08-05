## Functions for postprocessing simulation experiments.
# Each postprocessing function should take the path to the folder
# as input and return a DataFrame.

include("../../../config.jl");
include(joinpath(LIB_PATH, "asymptotic-variance.jl"));
include(joinpath(LIB_PATH, "data-functions.jl"));
include(joinpath(LIB_PATH, "math-functions.jl"));
using JLD2
using DataFrames
using StatsBase
using ArgParse
using Dates: today

# Parsing a string vector with ArgParse. Both , and . are valid delimiters.
function ArgParse.parse_item(::Type{Vector{String}}, x::AbstractString)
    strip_chars = [',', '.'];
    string.(split(x, strip_chars));
end
function parse_postprocess_args(ARGS)
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--experiments", "-e"
            help = string("a string vector of experiments indicating which ",
                          "experiments to postprocess. the possible experiment ",
                           "names can be found in the source code of this script.")
            arg_type = Vector{String}
            required = true
        "--indirpath", "-i"
            help = "the absolute path to the folder where to look for data"
            arg_type = String
            default = pwd()
        "--outdirpath", "-o"
            help = "the absolute path to the output folder where to save data"
            arg_type = String
            default = joinpath(pwd(), "postprocessed-" * string(today()))
        "--verbose", "-v"
            help = "print information about progress?"
            action = :store_true
    end;
    ArgParse.parse_args(ARGS, s);
end

## Function to get performance statistics.
function perf_stats(r)
    if eltype(r.x1) <: AbstractVector
        xfirst = map(first, r.x1);
    else
        xfirst = r.x1;
    end
    SV_sigma2 = estimateSV(xfirst);
    ac1 = autocor(xfirst, [1])[1];
    ire = SV_sigma2 * r[:npar];
    IACT = iact(xfirst);

    (emp_acc = prop_of_changes(r.x1),
     IACT = IACT,
     log_IACT = log(IACT),
     ac1_IACT = (1.0 + ac1) / (1.0 - ac1),
     ac1 = ac1,
     esjd = esjd(r.x1),
     SV_sigma2 = SV_sigma2,
     BM_sigma2 = estimateBM(xfirst),
     ire = ire,
     log_ire = log(ire));
end

## Function for reading simulations with the FDI-CPF.
function read_fdi_cpf_sim(folder::AbstractString)
    filepaths = joinpath.(folder, readdir(folder));
    d = DataFrame();
    adaptation = jldopen(filepaths[1], "r") do f
        f["out"][1].adaptation;
    end;
    for file in filepaths
        o = jldopen(file, "r") do f
            f["out"];
        end;
        for r in o
            base_nt = (model = r.model,
                       adaptation = adaptation,
                       npar = r.npar,
                       nsim = r.nsim,
                       burnin = r.burnin,
                       ts_length = r.ts_length,
                       sigma_x1 = r.sigma_x1,
                       sigma_x = r.sigma_x,
                       sigma_y = r.sigma_y);
            ps_nt = perf_stats(r);
            if adaptation == "aswam"
                nt = (target = r.target,
                      last_log_delta = r.last_log_delta);
            else
                nt = (last_cov = r.last_cov[1, 1],);
            end
            push!(d, merge(base_nt, ps_nt, nt));
        end
    end
    if adaptation == "aswam"
        # Add ranks.
        d = by(d, [:npar, :sigma_x, :sigma_y]) do gdf
            (target = gdf.target,
             ac1_rank = ordinalrank(gdf.ac1),
             iact_rank = ordinalrank(gdf.IACT));
         end |> x -> join(d, x, on = [:target, :npar, :sigma_x, :sigma_y],
                          kind = :left);
    end
    d;
end

## FDI-CPF mvnormal
using StaticArrays # Needed because the JLD files contain StaticArrays.
function read_fdi_cpf_mvnormal_sim(folder::AbstractString)
    filepaths = read_dir(folder, join = true);
    d = DataFrame();
    for filepath in filepaths
        o = jldopen(filepath, "r") do file
            file["out"];
        end
        for r in o
            ps_nt = perf_stats(r);
            base_nt = (model = r[:model],
                       statedim = r[:statedim],
                       npar = r[:npar],
                       sigma = r[:sigma],
                       target = r[:target],
                       nsim = r[:nsim],
                       burnin = r[:burnin],
                       thin = r[:thin])
            push!(d, merge(base_nt, ps_nt));
        end
    end
    d;
end

## Function for reading CPF-BS simulations.
function read_cpf_bs_sim(folder::AbstractString)
    path = read_dir(folder, join = true)[];
    d = DataFrame();
    o = jldopen(path, "r") do file
        file["out"]
    end
    for nt in o
        nt_ps = perf_stats(nt);
        base_nt = (model = "cpf-bs-" * nt.model,
                  nsim  = nt.nsim,
                  burnin = nt.burnin,
                  ts_length = nt.ts_length,
                  npar = nt.npar,
                  sigma_x = nt.sigma_x,
                  sigma_x1 = nt.sigma_x1,
                  sigma_y = nt.sigma_y);
        push!(d, merge(base_nt, nt_ps));
    end
    d;
end

## Function for reading DPG-CPF simulations.
function read_dpg_sim(folder::AbstractString)
    filepath = read_dir(folder, join = true)[];
    d = DataFrame();
    jldopen(filepath, "r") do file
        o = file["out"];
        for nt in o
            nt_ps = perf_stats(nt);
            base_nt = (model = "dpg-cpf-" * nt.model,
                      nsim  = nt.nsim,
                      burnin = nt.burnin,
                      ts_length = nt.ts_length,
                      npar = nt.npar,
                      sigma_x = nt.sigma_x,
                      sigma_x1 = nt.sigma_x1,
                      sigma_y = nt.sigma_y);
            push!(d, merge(base_nt, nt_ps));
        end
    end
    d;
end

## Function for reading diffuse normal simulations.
function read_dgi_cpf_sim(folder::AbstractString; variable::Symbol = :target)
    filepaths = joinpath.(folder, readdir(folder));
    d = DataFrame();
    for file in filepaths
        o = jldopen(file, "r") do f
            f["out"];
        end
        for nt in o
            ps_nt = perf_stats(nt);
            base_nt = (model = nt.model,
                       npar = nt.npar,
                       nsim = nt.nsim,
                       burnin = nt.burnin,
                       #thin = nt.thin,
                       target = nt.target,
                       last_beta = nt.last_beta,
                       sigma_x1 = nt.sigma_x1,
                       sigma_x = nt.sigma_x,
                       sigma_y = nt.sigma_y);
            push!(d, merge(base_nt, ps_nt));
        end
    end
    if variable == :target
        dsumm = by(d, [:npar, :sigma_x1, :sigma_x]) do gdf
            (ac1_rank = ordinalrank(gdf.ac1),
            iact_rank = ordinalrank(gdf.IACT),
            target = gdf.target);
        end;
        return join(d, dsumm, on = [:npar, :sigma_x1, :sigma_x, :target],
                    kind = :left);
    else
        rename!(d, :last_beta => :beta);
        select!(d, Not(:target));
        dsumm = by(d, [:npar, :sigma_x1, :sigma_x]) do gdf
            (ac1_rank = ordinalrank(gdf.ac1),
            iact_rank = ordinalrank(gdf.IACT),
            beta = gdf.beta);
        end;
        return join(d, dsumm, on = [:npar, :sigma_x1, :sigma_x, :beta],
                             kind = :left);
    end
end

function read_dgi_cpf_reps(folder::AbstractString)
    d = DataFrame();
    for filename in read_dir(folder, join = true)
        o = jldopen(filename, "r") do file
            file["out"];
        end
        for nt in o
            ps_nt = perf_stats(nt);
            base_nt = (model = nt.model,
                       npar = nt.npar,
                       nsim = nt.nsim,
                       burnin = nt.burnin,
                       thin = nt.thin,
                       rep = nt.rep,
                       target = nt.target,
                       beta = nt.last_beta,
                       sigma_x1 = nt.sigma_x1,
                       sigma_x = nt.sigma_x,
                       sigma_y = nt.sigma_y);
            push!(d, merge(base_nt, ps_nt));
        end
    end
    d;
end

## Functions for reading SEIR output.
function read_fdi_pg_seir(folder::AbstractString)
    filepath = read_dir(folder, join = true)[];
    fdi_pg_seir = jldopen(filepath, "r") do file
         fixed = file["fixedparams"];
         params = let
             v = map(i -> file["params"][:, i], 1:size(file["params"], 2));
             (; zip(file["paramnames"], v)...);
         end
         sim = (s = file["s"],
                e = file["e"],
                i = file["i"],
                r = file["r"],
                ρ = file["rho"],
                R0 = file["R0"],
                x0_e = file["x0_e"],
                x0_i = file["x0_i"],
                x0_ρ = file["x0_rho"]);

            (sim = sim, params = params, fixed = fixed,
             data = file["data"], npar = file["npar"],
             burnin = file["burnin"], thin = file["thin"]);
    end
    fdi_pg_seir;
end

function read_dpg_cpf_seir(folder::AbstractString)
    filepath = read_dir(folder, join = true)[];
    dpg_cpf_seir = jldopen(filepath, "r") do file
        fixed = file["fixedparams"];
        params = let
            v = map(i -> file["params"][:, i], 1:size(file["params"], 2));
            (; zip(file["paramnames"], v)...);
        end
        sim = (s = file["s"],
               e = file["e"],
               i = file["i"],
               r = file["r"],
               ρ = file["trans_R0"],
               R0 = file["R0"],);

        (sim = sim, params = params, fixed = fixed,
         data = file["data"], npar = file["npar"],
         burnin = file["burnin"], thin = file["thin"]);
    end
    dpg_cpf_seir;
end
