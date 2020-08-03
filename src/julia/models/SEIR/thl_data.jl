using CSV, DataFrames, Dates, Plots

show_window = Dates.Day(28)
reporting_window = Dates.Day(4)

# Testausmäärät:
tests_url = "https://sampo.thl.fi/pivot/prod/api/epirapo/covid19case/fact_epirapo_covid19case.csv?&row=dateweek2020010120201231-443702L&column=measure-444833L#"
tests = CSV.read(download(tests_url))
tests = unstack(tests, :Aika, :Mittari, :val)
select!(tests, [:Aika, :Testausmäärä])
dropmissing!(tests)
#Not(:Asukaslukumäärä))

# Tapausmäärät:
url = "https://sampo.thl.fi/pivot/prod/fi/epirapo/covid19case/fact_epirapo_covid19case.csv?row=hcdmunicipality2020-445222&column=%20dateweek2020010120201231-443702L"
# Alueet:
data = CSV.read(download(url))
data = unstack(data, :Aika, :Alue, :val)
_data = dropmissing(data)
first_record = _data.Aika[1]; last_record = _data.Aika[end]
data = filter(r -> first_record <= r.Aika <= last_record, data)
#rename!(data, [:Date, :Cases, :Tests])
#start_date = minimum(data.Aika[.!ismissing.(data.val)])
data = outerjoin(data, tests, on=:Aika)
data = coalesce.(data, 0)
data.Muut = data[:,Symbol("Kaikki Alueet")] - data[:,Symbol("Helsingin ja Uudenmaan SHP")]

#end_date = Dates.today()
#start_date = end_date - show_window
report_date = last_record - reporting_window

#show_data = dropmissing(filter(r->r.Aika >= start_date, data))

#shp = unique(data.Alue)
function show_cases(data, col; summary=sum,
    show_summary=true, title="$(col)")
    col = Symbol(col)
    s = summary(data[:,col])
    p = plot(ylabel=title * (show_summary ? ": $(s)" : ""))
    bar!(p, data.Aika, data[:,col], legend=false, tickdir=:out)
    vspan!(p, [report_date, last_record+Dates.Day(1)], color=:grey, fillalpha=0.5,
    linealpha=0.0)
    #, xticks=data.Date, xrotation=45)
    p=vline!(p,[data.Aika[1]], label="",alpha=0.0)
    p
end
#p = Dict()
#for a in shp
#    p[a] = show_cases(data, a)
#end
#display(p["Kaikki sairaanhoitopiirit"])

#data.Ratio = data.Cases ./ data.Tests

CSV.write("thl_data.csv", data)

plot(show_cases(data, "Helsingin ja Uudenmaan SHP"; title="HUS"),
     show_cases(data, "Muut"; title="Muut"),
     show_cases(data, "Testausmäärä"; title="Testit"),
     layout=(3,1))

#all = dropmissing(filter(r->(r.Alue=="Kaikki sairaanhoitopiirit" &&
#                 r.Aika>=start_date && r.Aika<=report_date), data))
#all = dropmissing(filter(r->(r.Alue=="Kaikki sairaanhoitopiirit" &&
#                 r.Aika>=start_date && r.Aika<=report_date), data))
