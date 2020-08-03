## Script to produce plots and summary statistics comparing the fully diffuse
# version of the SEIR model (FDI-PG) with the diffuse particle Gibbs
# version (DPG-BS). This script assumes that summaries of the SEIR experiments
# are in /output/results/summaries.

include("../../../config.jl");
include(joinpath(LIB_PATH, "data-functions.jl"));
include(joinpath(LIB_PATH, "asymptotic-variance.jl"));
include(joinpath(LIB_PATH, "math-functions.jl"));
include(joinpath(LIB_PATH, "r-plot-helpers.jl"));
using JLD2
using StatsBase
using Distributions

## Generate the plots path.
mkpath(PLOTS_PATH);
println(string("Producing plots to ", PLOTS_PATH, "."));

## Load data.
dpg_cpf_seir = load_summary("dpg-cpf-seir-summary.jld2");
fdi_pg_seir = load_summary("fdi-pg-seir-summary.jld2");

const COVID_DATA = fdi_pg_seir.data;
const DATES = COVID_DATA[:, :Date];

# Produce DataFrames of initial states and parameters.
fdi_params = DataFrame(;fdi_pg_seir.params...);
fdi_init_states = let nm = (:s, :e, :i, :R0)
    map(nm) do s
        fdi_pg_seir.sim[s][1, :]
    end |> x -> (; zip(nm, x)...);
end;
fdi_variables = hcat(fdi_params, DataFrame(; fdi_init_states...));
for v in ((:log_σ, exp), (:logit_p, invlogit))
    name = v[1]; f = v[2];
    x = fdi_variables[!, name];
    x .= f.(x);
end
rename!(fdi_variables, :log_σ => :sigma, :logit_p => :p);
rename!(fdi_variables,
        map(x -> x => Symbol(string(x) * "_init"), (:s, :e, :i, :R0))...);
fdi_variables[!, :method] .= "fdi";
fdi_variables[!, :iteration] = 1:nrow(fdi_variables);

# DPG
dpg_variables = hcat(DataFrame(; dpg_cpf_seir.params...),
                  DataFrame(s = dpg_cpf_seir.sim[:s][1, :]),
                  DataFrame(R0 = dpg_cpf_seir.sim[:R0][1, :]));
select!(dpg_variables, Not(:trans_R0_init))
for v in ((:log_sigma, exp), (:logit_p, invlogit),
          (:e, x -> ceil(x)), (:i, x -> ceil(x)))
    name = v[1]; f = v[2];
    x = dpg_variables[!, name];
    x .= f.(x);
end
for s in (:e, :i)
    dpg_variables[!, s] = Int.(dpg_variables[!, s]);
end
rename!(dpg_variables, :log_sigma => :sigma, :logit_p => :p);
rename!(dpg_variables, map(x -> x => Symbol(string(x) * "_init"), (:s, :e, :i, :R0))...);
dpg_variables[!, :method] .= "dpg";
dpg_variables[!, :iteration] = 1:nrow(dpg_variables);

# Joint data.
vardf = vcat(fdi_variables, dpg_variables);

# Relative IACT per time and state variable.
fdi_dpg_rel_iact = let methods = [("FDI-PG", fdi_pg_seir), ("DPG-BS", dpg_cpf_seir)],
                       vars = [:s, :e, :i, :r, :R0]
    map(methods) do o
        nm = o[1];
        dat = o[2];
        X = map(vars) do sym
            mapslices(iact, dat[1][sym]; dims = 2)[:, 1]
        end |> x -> hcat(x...);

        d = DataFrame(X);
        rename!(d, vars);
        d[!, :date] = DATES;
        d = stack(d, vars, value_name = :IACT);
        d[!, :method] .= nm;
        d;
    end |> x -> vcat(x...) |>
    x -> unstack(x, :method, :IACT) |>
    x -> begin
        x[!, :rel_IACT] .= x[!, Symbol("FDI-PG")] ./ x[!, Symbol("DPG-BS")];
        x[!, :variable] = string.(x[!, :variable]);
        x;
    end
end

## TABLES

# Mixing statistics.
mix_stats = let
    d = DataFrame();
    α = quantile(Normal(0, 1), 0.975);
    for m in unique(vardf[!, :method])
        data = filter(r -> r[:method] == m, vardf);
        for s in setdiff(names(data), [:method, :iteration])
            x = data[!, s];

            # Neff confint.
            IACT = iact(x);
            n = length(x);
            mean_x = mean(x);
            neff = n / IACT;
            neff_ci_width = α * std(x) / sqrt(neff);

            # BM confint.
            BM_sigma2 = estimateBM(Float64.(x));
            BM_ci_width = α * sqrt(BM_sigma2) / sqrt(n);

            # SV confint.
            SV_sigma2 = estimateSV(Float64.(x));
            SV_ci_width = α * sqrt(SV_sigma2) / sqrt(n);

            append!(d, DataFrame(id = m,
                                 variable = s,
                                 ac1 = autocor(x, [1])[],
                                 IACT = IACT,
                                 neff = neff,
                                 mean = mean_x,
                                 var = var(x),
                                 BM_sigma2 = BM_sigma2,
                                 SV_sigma2 = SV_sigma2,
                                 neff_ci = (mean_x - neff_ci_width, mean_x + neff_ci_width),
                                 BM_ci = (mean_x - BM_ci_width, mean_x + BM_ci_width),
                                 SV_ci = (mean_x - SV_ci_width, mean_x + SV_ci_width)));
        end
    end
    d;
end |> x -> sort!(x, [:id, :variable]);

mkpath(joinpath(RESULTS_PATH, "summaries"));
let filename = "fdi-vs-dpg-seir-variable-mixing-summary.jld2"
    outpath = joinpath(RESULTS_PATH, "summaries", filename);
    jldopen(outpath, "w") do file
        file["out"] = mix_stats;
    end;
end;

dpg_fdi_tab_values = let
    var_ord = [:e_init, :i_init, :R0_init, :sigma, :p];

    d = select(mix_stats, [:id, :variable, :IACT, :neff, :neff_ci]) |>
    x -> filter(r -> r[:variable] in var_ord, x) |>
    x -> stack(x, [:IACT, :neff, :neff_ci],
               variable_name = :stat, value_name = :value) |>
    x -> begin
        x[!, :statid] .= string.(x[!, :stat]) .* "_" .* x[!, :id];
        x
    end |>
    x -> select!(x, Not([:stat, :id])) |>
    x -> unstack(x, :statid, :value) |>
    x -> select(x, [:variable, :IACT_fdi, :IACT_dpg,
                    :neff_fdi, :neff_dpg,
                    :neff_ci_fdi, :neff_ci_dpg]) |>
    x -> begin
    ind = map(z -> findfirst(y -> y == z, x[!, :variable]), var_ord);
    select(x[ind, :], Not(:variable));
    #x[ind, :]
    end
    format(x::Real) = string(round(x, digits = 3))
    format(x::Tuple) = "(" * format(x[1]) * ", " * format(x[2]) * ")";

    map(x -> format(x), Matrix(d)) |>
    x -> mapslices(s -> reduce((x, y) -> x * " & " * y, s), x, dims = 2) |>
    x -> reduce((y, z) -> y * "\\\\ \n" * z, x);
end;

## Plots
# Traceplots of initial states and parameters per method.
function plot_seir_traces(; show::Bool = false, locations::Vector{String},
                            type::String = "pdf")
    R"""
    library(tidyr)
    library(ggplot2)
    library(patchwork)
    library(dplyr)

    # Get data.
    additional_thin <- 10
    thin_ind <- seq(additional_thin, max($fdi_variables[["iteration"]]),
                    by = additional_thin)
    d_fdi <- filter($fdi_variables, iteration %in% thin_ind) %>%
             gather(key = "param", value = "value", -method, -iteration)
    d_dpg <- filter($dpg_variables, iteration %in% thin_ind) %>%
             gather(key = "param", value = "value", -method, -iteration)
    plot_data <- bind_rows(d_fdi, d_dpg) %>% filter(param != "s_init")

    expand <- rep(0.007, 2)
    param_names <- c(#s_init = "S[1]",
    e_init = "E[1]",
    i_init = "I[1]",
    R0_init = "R0[1]",
    sigma = "sigma",
    p = "p")
    method_names <- c(fdi = "plain(FDI-PG)", dpg = "plain(DPG-BS)")
    plot_data[["param"]] <- factor(param_names[plot_data[["param"]]],
                                   levels = unname(param_names))
    plot_data[["method"]] <- factor(method_names[plot_data[["method"]]],
                                    levels = unname(method_names))

    xticks <- seq(0, 50000, by = 10000)
    xticknames <- paste0(xticks / 100, "k")

    plot_traces <- function(d) {
        ggplot(d, aes(x = iteration, y = value)) +
        geom_line(alpha = 0.8) +
        facet_grid(param ~ method, switch = "y", scales = "free_y",
                   labeller = label_parsed) +
        scale_x_continuous(expand = expand, breaks = xticks, labels = xticknames) +
        labs(x = "Iteration") +
        theme_faceted +
        theme(axis.text.x = element_text(size = SIZE_DEFAULTS[["axis_text_size"]] - 1,
                                         angle = 30, hjust = 1),
        axis.text.y = element_blank(),
        axis.title.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.ticks.length.y = unit(0, "pt"),
        panel.grid = element_blank(),
        strip.text.y.left = element_text(angle = 0),
        panel.spacing.x = unit(4, "pt"),
        plot.margin = unit(c(0, 3, 0, 2), "pt"))
    }
    plt <- plot_traces(plot_data)

        filename <- "seir-traces"
        save_plot(plt, filename, w = 6, h = 3, show = $show,
                  locations = $locations, type = "pdf")
    """
end
plot_seir_traces(show = true, locations = ["default"])


## Plots of autocorrelation per model and parameter.
function plot_ac_per_method_and_param(; show::Bool = false, locations)
    let VARS = [:sigma, :p, :s_init, :e_init, :i_init, :R0_init]
        data = stack(vardf, VARS; variable_name = :variable, value_name = :value)
        data[!, :variable] .= string.(data[!, :variable]);

        R"""
        library(dplyr)
        library(purrr)
        library(tidyr)
        library(ggplot2)
        library(latex2exp)

        expand <- rep(0.02, 2)
        method_names <- c(fdi = "FDI-PG", dpg = "DPG-BS")
        variable_names <- c(s_init = "S[1]",
                            e_init = "E[1]",
                            i_init = "I[1]",
                            R0_init = "R0[1]",
                            sigma = "sigma",
                            p = "p")

        plot_data <- group_by($data, variable, method) %>%
        summarise(ac_list = list(acf(value, plot = FALSE, lag.max = 50))) %>%
        ungroup %>%
        mutate(ac = map(ac_list, ~ as.numeric(.x[["acf"]])),
               lag = map(ac_list, ~ as.numeric(.x[["lag"]]))) %>%
        select(-ac_list) %>%
        unnest(c(ac, lag)) %>%
        filter(lag != 0) %>%
        mutate(method = factor(unname(method_names[method]),
                              levels = unname(method_names)),
               variable = factor(unname(variable_names[variable]),
                                 levels = unname(variable_names))) %>%
        filter(variable != "S[1]")

        plt <- ggplot(plot_data, aes(x = lag, y = ac)) +
        geom_line(aes(linetype = method)) +
        facet_wrap(~ variable, labeller = label_parsed) +
        scale_x_continuous(breaks = seq(0, 50, by = 10), expand = expand) +
        scale_y_continuous(breaks = seq(0.0, 1.0, by = 0.2), expand = expand) +
        labs(y = "Autocorrelation", x = "Lag") +
        theme_faceted +
        theme(legend.position = "top",
              panel.spacing.x = unit(3, "pt"),
              legend.title = element_blank(),
              legend.margin = margin(0, 0, -10, 0))

        filename <- "seir-ac-per-method-and-param.pdf"
        save_plot(plt, filename, w = 5, h = 3, show = $show, locations = $locations)
        """
    end
end
plot_ac_per_method_and_param(show = true, locations = ["default"])

## Plots of density estimates per model and parameter.
function plot_dens_per_method_and_param(; show::Bool = false, locations)
    let VARS = [:sigma, :p, :s_init, :e_init, :i_init, :R0_init]
        data = stack(vardf, VARS; variable_name = :variable, value_name = :value)
        data[!, :variable] .= string.(data[!, :variable]);

        R"""
        library(dplyr)
        library(ggplot2)
        library(latex2exp)

        expand <- 0.01 * rep(1, 2)
        method_names <- c(fdi = "FDI-PG", dpg = "DPG-BS")
        variable_names <- c(s_init = "S[1]",
        e_init = "E[1]",
        i_init = "I[1]",
        R0_init = "R0[1]",
        sigma = "sigma",
        p = "p")

        plot_data <- mutate($data,
            method = factor(unname(method_names[method]),
            levels = unname(method_names)),
            variable = factor(unname(variable_names[variable]),
            levels = unname(variable_names))) %>%
            filter(variable != "S[1]")

            plt <- ggplot(plot_data, aes(x = value)) +
            stat_density(aes(linetype = method),
                         geom = "line", position = "identity") +
            facet_wrap(~ variable, labeller = label_parsed, scales = "free") +
            scale_x_continuous(expand = expand) +
            theme_bw() +
            theme(legend.position = "top",
            legend.title = element_blank(),
            legend.text = element_text(size = SIZE_DEFAULTS[["legend_text_size"]]),
            axis.title = element_blank(),
            legend.margin = margin(0, 0, -10, 0),
            axis.text = element_text(size = 7),
            strip.text = element_text(size = SIZE_DEFAULTS[["strip_text_size"]]))

            filename <- "seir-dens-per-method-and-param.pdf"
            save_plot(plt, filename, w = 6, h = 3, show = $show, locations = $locations)
            """
    end
end
plot_dens_per_method_and_param(; show = true, locations = ["default"])

# Plot relative IACT of FDI-PG wrt that of DPG-BS for state variables wrt time.
function plot_rel_iact_fdi_dpg(; show::Bool = false, locations)

    R"""
    library(ggplot2)
    library(dplyr)

    d <- $fdi_dpg_rel_iact %>%
        filter(variable != "s")
    varnames <- c(#s = "S",
                  e = "E", i = "I", r = "R", R0 = "R0")
    d[["variable"]] <- factor(varnames[d[["variable"]]], levels = unname(varnames))
    daybreaks <- seq(min(d[["date"]]), max(d[["date"]]), by = 7)
    expand <- rep(0.005, 2)

    plt <- ggplot(d, aes(x = date, y = rel_IACT)) +
    geom_line(alpha = 0.6) +
    geom_hline(aes(yintercept = 1), linetype = "dashed") +
    scale_x_date(expand = expand, breaks = daybreaks) +
    #scale_y_continuous(breaks = seq(0.0, 1.0, by = 0.1)) +
    facet_grid(variable ~ .) +
    labs(y = "Relative IACT (FDI-PG / DPG-BS)") +
    theme_faceted +
    theme(axis.title.x = element_blank(),
    axis.text.x = element_text(angle = 30, hjust = 1),
    strip.text.y.right = element_text(angle = 0))

    filename <- "seir-rel-iact-fdi-dpg.pdf"
    save_plot(plt, filename, w = 5, h = 4, show = $show, locations = $locations)
    """
end
plot_rel_iact_fdi_dpg(show = true, locations = ["default"]);

## Plot distribution of R0 (1) and posterior predictive check (2) for
# in the fully diffuse case.

include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(MODELS_PATH, "SEIR", "SEIR_model.jl"));

"""
Make a "posterior predictive check", i.e return datasets
from the SEIR model such that each simulation is conditional on the
simulated states and parameters obtained when the model was fit.
A matrix of datasets with each simulation on the columns is returned.
"""
function SEIR_post_pred(out)
    n = size(out.sim.s, 1);
    M = size(out.sim.s, 2);
    y_ = [0]; y = zeros(Int, n, M);

    for j in 1:M
        for t in 1:n
            par = SEIRParticle(out.sim.s[t, j], out.sim.e[t, j], out.sim.i[t, j],
                               out.sim.r[t, j], out.sim.ρ[t, j]);
            θ = (log_σ = out.params[:log_σ][j],
                 logit_p = out.params[:logit_p][j],
                 eff = out.fixed[:eff],
                 γ = out.fixed[:γ]);
            SEIR_simobs!(y_, par, t, nothing, θ);
            y[t, j] = y_[];
        end
    end
    y;
end

function quantiles(mat::AMat{<: Real},
                   qs::Vector{<: AFloat} = [0.025, 0.25, 0.5, 0.75, 0.975])
    mapslices(x -> quantile(x, qs), mat, dims = 2);
end

function plot_R0_and_postpred(; show::Bool = false, locations)
    # Dataset of quantiles.
    R0_data = let
        fdi_R0_qs = quantiles(fdi_pg_seir.sim.R0);
        dpg_R0_qs = quantiles(dpg_cpf_seir.sim.R0);
        map([("fd", fdi_R0_qs), ("dpg", dpg_R0_qs)]) do qs
            name = qs[1];
            q = qs[2];
            nm = [:q025, :q25, :q50, :q75, :q975];
            d = DataFrame(q);
            rename!(d, nm);
            d[!, :method] .= name;
            d[!, :time] = DATES;
            d;
        end |> x -> vcat(x...);
    end

    # Build dataset from the posterior predictive check.
    post_pred_data = let every = 500
        y = SEIR_post_pred(fdi_pg_seir);
        plot_data = DataFrame(y[:, 1:every:end]);
        plot_data[!, :time] = DATES;
        plot_data = stack(plot_data, Not(:time));
        rename!(plot_data, :variable => :sim);
        plot_data[!, :sim] = begin
            ms = map(x -> match(r"(\d+)$", x), string.(plot_data[!, :sim]));
            parse.(Int, map(x -> x[1], ms));
        end;
        plot_data
    end

    R"""
    library(dplyr)
    library(ggplot2)
    library(patchwork)

    add_event <- function(date, label, tick_height, text_height = tick_height - 0.4,
                          text_size = 2) {
        d <- data.frame(date = date, label = label)
        list(geom_segment(data = d, mapping = aes(x = date, xend = date),
                     y = 0, yend = tick_height, alpha = 0.7, inherit.aes = FALSE),
        annotate(geom = "text", label = label,
                 x = date, y = text_height, hjust = 0, size = text_size))
    }

    # Some constants.
    R0_data <- $R0_data
    expand <- c(0.01, 0.01)
    xbreaks <- seq(min(R0_data[["time"]]), max(R0_data[["time"]]), by = 7)
    test_only_in_risk <- as.Date("2020-03-13") # OK
    lockdown_starts <- as.Date("2020-03-18") # OK
    uusimaa_closes <- as.Date("2020-03-28") # OK
    restaurants_close <- as.Date("2020-04-04") # OK
    uusimaa_opens <- as.Date("2020-04-15") # OK
    test_suspected <- as.Date("2020-04-15") # OK
    schools_open <- as.Date("2020-05-14") # OK
    restaurants_open <- as.Date("2020-06-01") # OK
    restrictions <- c(lockdown_starts, uusimaa_closes, restaurants_close,
                      uusimaa_opens)

    theme_first <- theme_bw() +
                   theme(axis.title.x = element_blank(),
                         axis.text.x = element_blank(),
                         axis.title.y = element_text(size = SIZE_DEFAULTS[["axis_title_size"]]),
                         axis.text.y = element_text(size = SIZE_DEFAULTS[["axis_text_size"]]),
                         axis.ticks.x = element_blank(),
                         axis.ticks.length.x = unit(0, "pt"),
                         plot.margin = margin(0, 0, 2, 0)) +
                         theme(legend.title = element_blank(),
                         legend.position = "top")

    theme_last <- theme_bw() +
                  theme(axis.title.x = element_blank(),
                        axis.title.y = element_text(size = SIZE_DEFAULTS[["axis_title_size"]]),
                        axis.text.y = element_text(size = SIZE_DEFAULTS[["axis_text_size"]]),
                        axis.text.x = element_text(angle = 30, hjust = 1,
                                                   size = SIZE_DEFAULTS[["axis_text_size"]]),
                        plot.margin = margin(0, 0, 0, 0))

    text_size <- 2
    plt_r0_fd <- filter(R0_data, method == "fd") %>%
        ggplot(aes(x = time)) +
            ## Changes in timeline:
            add_event(test_only_in_risk, " Only risk\n groups tested",
                      tick_height = 4.55, text_size = text_size) +
            # Uusimaa closed:
            annotate(geom = "rect", xmin = uusimaa_closes,
                     xmax = uusimaa_opens, ymin = 0, ymax = Inf,
                     fill = "black", alpha = 0.3) +
            annotate(geom = "text", label = "Uusimaa region closed\ndue to COVID-19",
                     x = mean(c(uusimaa_opens, uusimaa_closes)), y = 5.0, size = text_size) +
            add_event(lockdown_starts, " Lockdown\n begins",
                      tick_height = 3.25, text_size = text_size) +
            add_event(test_suspected, " All suspected of\n COVID-19 tested",
                      tick_height = 2.35, text_size = text_size) +
            add_event(restaurants_close, " Restaurants\n close",
                      tick_height = 2.35, text_size = text_size) +
            add_event(schools_open, " Schools\n open",
                      tick_height = 1.8, text_size = text_size) +
            add_event(restaurants_open, " Restaurants\n open",
                      tick_height = 1.7, text_size = text_size) +

            geom_line(aes(y = q50)) +
            geom_ribbon(aes(ymin = q025, ymax = q975), alpha = 0.5) +
            geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.75) +
            scale_x_date(expand = expand, breaks = xbreaks) +
            scale_y_continuous(expand = expand) +
            labs(y = "R0") +
            theme_first

    post_pred_data <- $post_pred_data
    orig_data <- $COVID_DATA

    plt_post_pred <- ggplot(post_pred_data, aes(x = time, y = value, group = sim)) +
        geom_point(color = "gray", alpha = 0.4, size = 1) +
        geom_point(data = orig_data, aes(x = Date, y = Cases), fill = "black",
                   inherit.aes = FALSE, size = 1, alpha = 0.6) +
        scale_x_date(breaks = xbreaks, expand = expand) +
        scale_y_continuous(breaks = seq(0, 500, by = 50), expand = expand) +
        labs(y = "No. new positive cases") +
        theme_last

    plt <- (plt_r0_fd / plt_post_pred) + theme(plot.margin = unit(c(0, 0, 0, 0), "pt"))

    filename <- "seir-fd-r0-and-postpred.pdf"
    save_plot(plt, filename, h = 3, w = 6, show = $show, locations = $locations)
    """
end
plot_R0_and_postpred(show = true, locations = ["default"])
