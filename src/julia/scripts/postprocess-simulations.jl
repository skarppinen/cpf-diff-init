## A script to process raw simulation output to DataFrames or other data
# structures. Do not include to open code.
#
include("../../../config.jl");
include(joinpath(LIB_PATH, "postprocessing-functions.jl"));
args = parse_postprocess_args(ARGS);

# Get command line arguments.
indirpath = args["indirpath"]; # Absolute path to the directory where to look for data.
experiments = unique(args["experiments"]); # Vector of experiment names to process.
outdirpath = args["outdirpath"]; # Absolute path to directory where to save files.
verbose = args["verbose"];

if verbose
    println(string("Input directory is ", indirpath));
    println(string("Output directory is ", outdirpath));
    println(string("Experiments are ", experiments));
end

# Mappings from simulation experiment to postprocessing function.
# The function should take the path to the folder where the raw data is as the
# only input and return the postprocessed dataset.
const EF_MAP = ["fdi-cpf-am-noisyar" => read_fdi_cpf_sim,
 "fdi-cpf-am-sv" => read_fdi_cpf_sim,
 "fdi-cpf-aswam-noisyar" => read_fdi_cpf_sim,
 "fdi-cpf-aswam-sv" => read_fdi_cpf_sim,
 "fdi-cpf-aswam-mvnormal" => read_fdi_cpf_mvnormal_sim,
 "fdi-pg-seir" => read_fdi_pg_seir,
 "dgi-cpf-betafix-reps-sv" => read_dgi_cpf_reps,
 "dpg-cpf-noisyar" => read_dpg_sim,
 "dpg-cpf-sv" => read_dpg_sim,
 "dpg-cpf-seir" => read_dpg_cpf_seir,
 "dgi-cpf-noisyar" => x -> read_dgi_cpf_sim(x, variable = :target),
 "dgi-cpf-sv" => x -> read_dgi_cpf_sim(x, variable = :target),
 "dgi-cpf-betafix-noisyar" => x -> read_dgi_cpf_sim(x, variable = :beta),
 "dgi-cpf-betafix-sv" => x -> read_dgi_cpf_sim(x, variable = :beta)];

# Check that experiments passed as argument are valid.
valid_experiments = map(first, EF_MAP);
for e in experiments
    if !(e in valid_experiments)
        msg = string("Experiment ", e, " not in valid experiments. ",
                     "Valid experiments are: ", valid_experiments, ".");
        throw(ArgumentError(msg));
    end
    if !isdir(joinpath(indirpath, e))
        msg = string("Could not find folder named ", e, " in the directory ",
                      indirpath, ".", " Place all related raw simulation files to a ",
                      "folder with this name.");
        throw(ArgumentError(msg));
    end
end
verbose && println("Requested experiments ", experiments, " are valid, starting ",
                   "postprocessing..");

# Get a Dict mapping experiment to function for requested experiments.
dict_ef = Dict(filter(x -> first(x) in experiments, EF_MAP));

# Postprocess.
mkpath(outdirpath);
for name in keys(dict_ef)
    d = dict_ef[name](joinpath(indirpath, name));
    outpath = joinpath(outdirpath, name * "-" * "summary.jld2");
    jldopen(outpath, "w") do file
        file["out"] = d;
    end
    verbose && println("Finished postprocessing experiment ", name, ".");
end
verbose && println("Finished.");
