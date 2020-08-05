## File controlling the command line interfaces for scripts in
# src/julia/scripts. The ArgParse package is used for defining the
# interfaces.

import ArgParse
"""
A function to allow ArgParse to parse a numeric vector passed
from the command line.
"""
function ArgParse.parse_item(::Type{Vector{T}}, x::AbstractString) where T <: Real
    strip_chars = ['[', ']', ','];
    split_x = split(x, [' ', ',']) |> (x -> filter(y -> y != "", x));
    char_array = map(y -> strip(y, strip_chars), split_x);
    return parse.(T, char_array)
end

function parse_fdi_cpf_noisyar_sv_args(ARGS)
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--model", "-m"
            help = "the model to run, SV or NOISYAR"
            arg_type = String
            required = true
        "--npar", "-p"
            help = "the numbers of particles"
            arg_type = Vector{Int}
            required = true
        "--iterations", "-i"
            help = "amount of iterations after burnin"
            arg_type = Int
            required = true
        "--burnin", "-b"
            help = "amount of burnin iterations"
            arg_type = Int
            required = true
        "--thin", "-t"
            help = "the amount of thinning. thin = 5 means every fifth iteration is saved."
            arg_type = Int
            default = 1
        "--adaptation"
            help = "the adaptation algorithm to use. 'aswam' or 'am'."
            arg_type = String
            default = "aswam"
        "--target", "-a"
            help = "the target acceptance rate. ignored if adaptation = 'am'."
            arg_type = Float64
            required = true
        "--sigma_x"
            help = "vector of state standard deviations to run simulations on"
            arg_type = Vector{Float64}
            required = true
        "--sigma_y"
            help = "vector of observation standard deviations to run simulations on"
            arg_type = Vector{Float64}
            required = true
        "--gamma"
            help = "adaptation parameter gamma"
            arg_type = Float64
            default = 2.0/3.0
        "--max_eta"
            help = "adaptation parameter max_eta"
            arg_type = Float64
            default = 0.5
        "--verbose", "-v"
            help = "should information regarding progress of simulations be printed"
            action = :store_true
        "--outfolder", "-o"
            help = "the output folder where to save data. if empty, use a default computed in the script"
            arg_type = String
            default = ""
    end;
    ArgParse.parse_args(ARGS, s);
end

function parse_fdi_cpf_mvnormal_args(ARGS)
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--npar", "-p"
            help = "the numbers of particles"
            arg_type = Vector{Int}
            required = true
        "--iterations", "-i"
            help = "amount of iterations after burnin"
            arg_type = Int
            required = true
        "--burnin", "-b"
            help = "amount of burnin iterations"
            arg_type = Int
            required = true
        "--thin", "-t"
            help = "the amount of thinning. thin = 5 means every fifth iteration is saved."
            arg_type = Int
            default = 1
        "--target", "-a"
            help = "the target value to tune towards to"
            arg_type = Float64
            required = true
        "--sigma"
            help = "vector of state standard deviations to run simulations on"
            arg_type = Vector{Float64}
            required = true
        "--statedim"
            help = "vector of state dimensions"
            arg_type = Vector{Int}
            required = true
        "--gamma"
            help = "ASWAM adaptation parameter gamma"
            arg_type = Float64
            default = 2.0/3.0
        "--max_eta"
            help = "ASWAM adaptation parameter max_eta"
            arg_type = Float64
            default = 0.5
        "--verbose", "-v"
            help = "should information regarding progress of simulations be printed"
            action = :store_true
        "--outfolder", "-o"
            help = "the output folder where to save data. if empty, use a default computed in the script"
            arg_type = String
            default = ""
    end;
    ArgParse.parse_args(ARGS, s);
end

function parse_dgi_cpf_noisyar_sv_args(ARGS)
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--model", "-m"
            help = "the model to run, SV or NOISYAR"
            arg_type = String
            required = true
        "--npar", "-p"
            help = "the numbers of particles"
            arg_type = Vector{Int}
            required = true
        "--iterations", "-i"
            help = "amount of iterations after burnin"
            arg_type = Int
            required = true
        "--burnin", "-b"
            help = "amount of burnin iterations"
            arg_type = Int
            required = true
        "--nreps"
            help = string("integer denoting how many replicates of the algorithm should be",
                          " repeated for each dataset simulated")
            arg_type = Int
            default = 1
        "--thin", "-t"
            help = "the amount of thinning. thin = 5 means every fifth iteration is saved."
            arg_type = Int
            default = 1
        "--target"
            help = "the target acceptance rate for adaptation"
            arg_type = Float64
            required = true
        "--beta_init"
            help = "the initial Crank-Nicolson beta value."
            arg_type = Float64
            default = 0.5
        "--sigma_x1"
            help = "a vector of initial state sds to run simulations on"
            arg_type = Vector{Float64}
            required = true
        "--sigma_x"
            help = "vector of state sds to run simulations on"
            arg_type = Vector{Float64}
            required = true
        "--sigma_y"
            help = "vector of observation sds to run simulations on"
            arg_type = Vector{Float64}
            required = true
        "--noadapt"
            help = "if given, the algorithm is run without adapting beta at all,
                    and the value of `beta_init` is used throughout the sampling.
                    in this case, the value of target is ignored."
            action = :store_true
        "--verbose", "-v"
            help = "print information about progress?"
            action = :store_true
        "--outfolder", "-o"
            help = "the output folder where to save data. if empty, use a default computed in the script"
            arg_type = String
            default = ""
    end;
    ArgParse.parse_args(ARGS, s);
end

function parse_dpg_cpf_noisyar_sv_args(ARGS)
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--model", "-m"
            help = "the model to run, SV or NOISYAR"
            arg_type = String
            required = true
        "--npar", "-p"
            help = "the numbers of particles"
            arg_type = Vector{Int}
            required = true
        "--iterations", "-i"
            help = "amount of iterations after burnin"
            arg_type = Int
            required = true
        "--burnin", "-b"
            help = "amount of burnin iterations"
            arg_type = Int
            required = true
        "--thin", "-t"
            help = "the amount of thinning. thin = 5 means every fifth iteration is saved."
            arg_type = Int
            default = 1
        "--sigma_x1"
            help = "a vector of initial state sds to run simulations on"
            arg_type = Vector{Float64}
            required = true
        "--sigma_x"
            help = "vector of state sds to run simulations on"
            arg_type = Vector{Float64}
            required = true
        "--sigma_y"
            help = "vector of observation sds to run simulations on"
            arg_type = Vector{Float64}
            required = true
        "--fully_diffuse", "-f"
            help = "should M1 be fully diffuse or normal. if given, sigma_x1 is ignored"
            action = :store_true
        "--gamma"
            help = "RAM adaptation parameter gamma"
            arg_type = Float64
            default = 2.0/3.0
        "--max_eta"
            help = "RAM adaptation parameter max_eta"
            arg_type = Float64
            default = 0.5
        "--verbose", "-v"
            help = "should information regarding progress of simulations be printed"
            action = :store_true
        "--outfolder", "-o"
            help = "the output folder where to save data. if empty, use a default computed in the script"
            arg_type = String
            default = ""
    end;
    ArgParse.parse_args(ARGS, s);
end

function parse_fdi_pg_seir_args(ARGS)
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--variant"
            help = "the type of observation model, negativebinomial or betabinomial"
            arg_type = String
            default = "negativebinomial"
        "--npar", "-p"
            help = "the number of particles"
            arg_type = Int
            default = 64
        "--iterations", "-i"
            help = "amount of iterations after burnin"
            arg_type = Int
            required = true
        "--burnin", "-b"
            help = "amount of burnin iterations"
            arg_type = Int
            required = true
        "--thin", "-t"
            help = "amount of thinning. must divide `iterations` evenly. thin = 1 corresponds to no thinning."
            arg_type = Int
            required = true
        "--ram_gamma"
            help = "RAM adaptation parameter gamma"
            arg_type = Float64
            default = 2.0/3.0
        "--ram_max_eta"
            help = "RAM adaptation parameter max_eta"
            arg_type = Float64
            default = 0.5
        "--aux_target"
            help = "target rate for auxiliary distribution adaptation"
            arg_type = Float64
            default = 0.8
        "--aux_gamma"
            help = "adaptation parameter gamma for auxiliary distribution"
            arg_type = Float64
            default = 2.0/3.0
        "--aux_max_eta"
            help = "adaptation parameter max_eta for auxiliary distribution"
            arg_type = Float64
            default = 0.5
        "--uusimaa", "-u"
            help = "run simulation only for Uusimaa region?"
            action = :store_true
        "--download", "-d"
            help = "download latest data from THL?"
            action = :store_true
        "--data_max_day"
            help = "used together with `data_max_month` to specify the maximum date in data used."
            arg_type = Int
            required = true
        "--data_max_month"
            help = "used together with `data_max_day` to specify the maximum date in data used."
            arg_type = Int
            required = true
        "--outfolder", "-o"
            help = "the output folder where to save data. if empty, use a default computed in the script."
            arg_type = String
            default = ""
        "--verbose", "-v"
            help = "display messages?"
            action = :store_true
    end;
    ArgParse.parse_args(ARGS, s);
end

function parse_dpg_cpf_seir_args(ARGS)
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--npar", "-p"
            help = "the number of particles"
            arg_type = Int
            default = 64
        "--iterations", "-i"
            help = "amount of iterations after burnin"
            arg_type = Int
            required = true
        "--burnin", "-b"
            help = "amount of burnin iterations"
            arg_type = Int
            required = true
        "--thin", "-t"
            help = "amount of thinning. must divide `iterations` evenly. thin = 1 corresponds to no thinning."
            arg_type = Int
            required = true
        "--uusimaa", "-u"
            help = "run simulation only for Uusimaa region?"
            action = :store_true
        "--download", "-d"
            help = "download latest data from THL?"
            action = :store_true
        "--data_max_day"
            help = "used together with `data_max_month` to specify the maximum date in data used."
            arg_type = Int
            required = true
        "--data_max_month"
            help = "used together with `data_max_day` to specify the maximum date in data used."
            arg_type = Int
            required = true
        "--outfolder", "-o"
            help = "the output folder where to save data. if empty, use a default computed in the script."
            arg_type = String
            default = ""
        "--verbose", "-v"
            help = "display progress bar and messages?"
            action = :store_true
    end;
    ArgParse.parse_args(ARGS, s);
end

function parse_diffinit_poor_mixing_args(ARGS)
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--outfolder", "-o"
            help = string("the output folder where to save data. ",
                          "if empty, use the default in the script file.")
            arg_type = String
    end;
    ArgParse.parse_args(ARGS, s);
end

function print_progress(delta::AFloat, i::Integer, tot::Integer)
    elapsed = round(delta, digits = 1);
    msg = string("Finished parameter combination ", i, "/",
                 tot, " (", round(i / tot * 100, digits = 1),
                 "%), (", elapsed, " s)");
    println(msg);
end
