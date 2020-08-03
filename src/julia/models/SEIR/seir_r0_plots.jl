using StatsPlots, Statistics, Measures
function quantile_plot!(plt, S, p=0.95; x=1:size(S)[1])
    alpha = (1-p)/2
    qs = [alpha, 0.5, 1-alpha]
    Q = mapslices(x->quantile(x, qs), S, dims=2)
    plot!(plt, x, Q[:,2], ribbon=(Q[:,2]-Q[:,1], Q[:,3]-Q[:,2]), color=:black,
    legend=:false, fillalpha=0.2)
end
function path_plot(out, v, f=+)
    _getfields(o, v) = (typeof(v) == Symbol) ? getfield(o, v) : map(_v->getfield(o, _v), v)
    S = [f(_getfields(out.X[j][i], v)...) for i=1:length(out.X[1]), j=1:length(out.X)]
    p_paths = plot(xlabel="Time", ylabel="$v",
    legend=false,  xrotation=45, xticks = data[:,date_name],
    bottom_margin=2mm, left_margin=1mm,
    title="Posterior median, 50% and 95% credible intervals")
    quantile_plot!(p_paths, S, 0.95; x=data[:,date_name])
    quantile_plot!(p_paths, S, 0.5; x=data[:,date_name])
    p_paths
end
function parameter_plot(out; title="")
    labels = String.(keys(out.theta0))
    corrplot(out.Theta', label=[labels...])
end
function posterior_predictive(out, max_samples=100)
    date = DateTime.(data[:,seir_const.date_name])
    cases = data[:,seir_const.cases_name]
    n = length(out.X); n_samples = min(n, max_samples)
    T = length(cases)
    plt = bar(date, cases, legend=false)
    ind = sample(1:n, n_samples; replace=false)
    Y = zeros(Int, T,n_samples)
    D = Matrix{DateTime}(undef, T, n_samples)
    #t = zeros(set_param!()
    theta = deepcopy(out.theta0)
    scratch = SEIRScratch()
    for j = 1:n_samples
        d_jitter = Dates.Minute(sample(-360:360))
        for k = 1:T
            j_ = ind[j]
            i_ = out.X[j_][k].i
            theta .= out.Theta[:,j_]
            set_param!(scratch, theta)
            D[k,j] = date[k]+d_jitter
            Y[k,j] = G_SEIR(k, out.X[j_][k], scratch)
        end
    end
    scatter!(plt, D, Y, color=:red, markerstrokewidth=0, alpha=0.5)
end
