## A script to download postprocessed simulation data used to produce
# the results in the article.
include("../../../config.jl");
using InfoZIP

outfolder = joinpath(RESULTS_PATH, "summaries");
mkpath(outfolder);
println(string("URL is ", SIMULATION_DATA_URL));
println("Downloading simulation data, please wait. This may take a few minutes..");
filepath = download(SIMULATION_DATA_URL, joinpath(outfolder, ".temp.zip"));
println("Finished downloading simulation data. Unzipping archive..");

InfoZIP.unzip(filepath, outfolder);
rm(joinpath(outfolder, ".temp.zip"));
InfoZIP.unzip(joinpath(outfolder, "cpf-diff-init", "simulation-results-and-data-jld2.zip"),
              outfolder);
rm(joinpath(outfolder, "cpf-diff-init"), recursive = true);
println(string("The simulation data is in the directory ", outfolder));
println("See the readme file in the directory for details.");
