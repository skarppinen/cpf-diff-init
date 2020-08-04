## Functions related to manipulating data, input and output etc.
include("../../../config.jl");
using DataFrames, CSV, Dates

"""
Load a summary DataFrame corresponding to a certain simulation.
"""
function load_summary(filename::AbstractString)
    inputfolder = joinpath(RESULTS_PATH, "summaries");
    filepath = joinpath(inputfolder, filename);
    jldopen(filepath, "r") do file
        file["out"];
    end
end

"""
Helper function for reading multiple files.
"""
function read_dir(folder::AbstractString; join::Bool = false)
    if join
        return joinpath.(folder, readdir(folder));
    else
        return readdir(folder);
    end
end

"""
Return filepath to the latest COVID datafile.
"""
function latest_covid_filepath()
    last(sort(read_dir(COVID_DATA_PATH, join = true)));
end


"""
Function returns a DataFrame with Finnish COVID-19 data from THL.

Optional arguments:
* `filename`: Attempt to read data file with this name from the `COVID_DATA_PATH`.
* `start_date`: The first date that should be in the returned data.
* `uusimaa_only`: Return data only for the Uusimaa region?
* `download`: Download latest data from THL website? Default is false,
which means that latest data is read from disk and returned. If download is true,
`filename` is ignored. Downloading also happens automatically, if there are
no datafiles on disk.
"""
function get_covid_data(; filename::AbstractString = "",
                        start_date::Date = Date("2020-03-01"),
                        uusimaa_only::Bool = false, download::Bool = false)

    # Load the raw data file.
    filepaths = sort(read_dir(COVID_DATA_PATH, join = true));
    if download || isempty(filepaths)
        # File is requested for download, or there is no COVID data.
        download_covid_data();
        filepath = latest_covid_filepath();
        data = CSV.read(filepath, DataFrame; copycols = true);
    else
        if filename == ""
            filepath = latest_covid_filepath();
        else
            filepath = joinpath(COVID_DATA_PATH, filename);
        end
        data = CSV.read(filepath, DataFrame; copycols = true);
    end

    # Data has been loaded, subset to include only the region requested.
    # The data is subsetted to start from `start_date`.
    cases_name = uusimaa_only ? Symbol("Helsingin ja Uudenmaan SHP") :
                                Symbol("Kaikki Alueet");
    out = select(data, :Aika, cases_name);
    out = filter(r -> r[:Aika] >= start_date, out);
    rename!(out, :Aika => :Date, cases_name => :Cases);
    out[!, :Region] .= uusimaa_only ? :uusimaa : :finland;
    out;
end


"""
Function downloads the latest Finnish COVID data from THL to
`COVID_DATA_PATH` (see file /config.jl). The downloaded data gets the name
"thl-covid-data-*date-today*.csv".
"""
function download_covid_data()
    # Get data about tests.
    tests = CSV.read(download(THL_TESTS_URL), DataFrame);
    tests = unstack(tests, :Aika, :Mittari, :val);
    select!(tests, [:Aika, :Testausmäärä]);
    dropmissing!(tests);

    # Get data about cases.
    data = CSV.read(download(THL_DATA_URL), DataFrame);
    data = unstack(data, :Aika, :Alue, :val);
    _data = dropmissing(data);
    first_record = _data.Aika[1];
    last_record = _data.Aika[end];
    data = filter(r -> first_record <= r.Aika <= last_record, data);

    # Join datasets and set all missing values to zero.
    data = outerjoin(data, tests, on = :Aika, makeunique = false,
                     indicator = nothing, validate = (false, false));
    data = coalesce.(data, 0);

    # Save file.
    str_last_record = string(last_record);
    filename = "thl-covid-data-" * str_last_record * ".csv";
    filepath = joinpath(COVID_DATA_PATH, filename);
    CSV.write(filepath, data);
    println("COVID data with last record from $str_last_record saved to: $filepath");
end
