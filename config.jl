## File containing settings and important paths related to the project.

# Shorthands for frequently used types.
const AVec = AbstractVector;
const AMat = AbstractMatrix;
const AFloat = AbstractFloat;

# Globals for referencing certain paths in the project.
const PROJECT_ROOT = @__DIR__;
const PROJECT_NAME = basename(PROJECT_ROOT);
const SRC_PATH = joinpath(PROJECT_ROOT, "src");
const LIB_PATH = joinpath(SRC_PATH, "julia", "lib");
const MODELS_PATH = joinpath(SRC_PATH, "julia", "models");
const OUTPUT_PATH = joinpath(PROJECT_ROOT, "output");
const PLOTS_PATH = joinpath(OUTPUT_PATH, "plots");
const RESULTS_PATH = joinpath(OUTPUT_PATH, "results");
const CONFIG_PATH = joinpath(SRC_PATH, "julia", "config");
const DATA_PATH = joinpath(PROJECT_ROOT, "data");
const COVID_DATA_PATH = joinpath(DATA_PATH, "covid");
mkpath(COVID_DATA_PATH);

# URLs.
const THL_DATA_URL = "https://sampo.thl.fi/pivot/prod/" *
                     "fi/epirapo/covid19case/" *
                     "fact_epirapo_covid19case.csv?" *
                     "row=hcdmunicipality2020-" *
                     "445222&column=%20dateweek2020010120201231-443702L";
const THL_TESTS_URL = "https://sampo.thl.fi/pivot/prod/" *
                      "api/epirapo/covid19case/" *
                      "fact_epirapo_covid19case.csv?&" *
                      "row=dateweek2020010120201231-" *
                      "443702L&column=measure-444833L#";
const SIMULATION_DATA_URL = "https://nextcloud.jyu.fi/index.php/s/zjeiwDoxaegGcRe/" *
                            "download?path=%2F&amp;files=simulation-results-and-data-jld2.zip";

# A seed number used in all scripts that run an experiment
# that needs to simulate data with the `simulate!` function.
# This ensures that there is no additional variance from
# sampling and all variability is in the parameters.
const DATA_SIM_SEED = 75399237;

# Activate the environment related to the project (if not already activated).
import Pkg
if PROJECT_NAME != basename(dirname(Base.active_project()))
    Pkg.activate(".");
end

# Setup Julia to only load code from standard library and active environment,
# and LIB_PATH.
empty!(LOAD_PATH);
push!(LOAD_PATH, "@");
push!(LOAD_PATH, "stdlib");
push!(LOAD_PATH, LIB_PATH);

nothing;
