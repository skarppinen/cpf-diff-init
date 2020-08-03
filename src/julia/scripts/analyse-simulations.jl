## Script to produce plots in the paper about the simulation experiments
# with DGI-CPF, FDI-CPF and DPG-BS. This script assumes that summaries of the
# simulation experiments are in /output/results/summaries.
include("../../../config.jl");
include(joinpath(LIB_PATH, "data-functions.jl"));
include(joinpath(LIB_PATH, "math-functions.jl"));
include(joinpath(LIB_PATH, "r-plot-helpers.jl"));
using JLD2
using DataFrames
using StatsBase
using Statistics

## Generate plots folder.
mkpath(PLOTS_PATH);

## Data load.
# Load noisyar and sv summaries with fully diffuse initialisation.
fdi_cpf_am = let
    fdi_cpf_am_noisyar = load_summary("fdi-cpf-am-noisyar-summary.jld2");
    fdi_cpf_am_sv = load_summary("fdi-cpf-am-sv-summary.jld2");
    vcat(fdi_cpf_am_noisyar, fdi_cpf_am_sv);
end
fdi_cpf_aswam = let
    fdi_cpf_aswam_noisyar = load_summary("fdi-cpf-aswam-noisyar-summary.jld2");
    fdi_cpf_aswam_sv = load_summary("fdi-cpf-aswam-sv-summary.jld2");
    d = vcat(fdi_cpf_aswam_noisyar, fdi_cpf_aswam_sv);
    d[!, :ire] .=  d[!, :SV_sigma2] .* d[!, :npar];
    d[!, :log_ire] .= log.(d[!, :ire]);
    d;
end
dpg_cpf = let
    dpg_cpf_noisyar = load_summary("dpg-cpf-noisyar-summary.jld2");
    dpg_cpf_sv = load_summary("dpg-cpf-sv-summary.jld2");
    d = vcat(dpg_cpf_noisyar, dpg_cpf_sv);
    replace!(d[!, :model], "dpg-cpf-NOISYAR" => "NOISYAR", "dpg-cpf-SV" => "SV");
    d;
end
fd_configs = let d = fdi_cpf_aswam
    keyparams = [:sigma_x];
    config_df = unique(d[:, keyparams]);
    config_df[!, :config] = 1:nrow(config_df);
    config_df;
end
fdi_cpf_am = join(fdi_cpf_am, fd_configs, kind = :left, on = [:sigma_x]);
fdi_cpf_aswam = join(fdi_cpf_aswam, fd_configs, kind = :left, on = [:sigma_x]);
dpg_cpf = join(dpg_cpf, fd_configs, kind = :left, on = [:sigma_x]);

# Load mvnormal summary with fully diffuse initialisation
fdi_cpf_mvnormal = load_summary("fdi-cpf-aswam-mvnormal-summary.jld2");

# Load noisyar and sv summaries with diffuse normal initialisation.
dgi_paramnames = [:sigma_x1, :sigma_x];
dgi_cpf_sim = let
    dgi_noisyar = load_summary("dgi-cpf-noisyar-summary.jld2");
    dgi_sv = load_summary("dgi-cpf-sv-summary.jld2");
    vcat(dgi_noisyar, dgi_sv);
end
dgi_cpf_sim[!, :penalty] .= dgi_cpf_sim[!, :SV_sigma2] .* dgi_cpf_sim[!, :npar];
dgi_configs = let
    params = [:sigma_x1, :sigma_x];
    config_df = unique(dgi_cpf_sim[:, params]);
    config_df[!, :config] = 1:nrow(config_df);
    config_df;
end
dgi_cpf_sim = join(dgi_cpf_sim, dgi_configs, on = dgi_paramnames, kind = :left);
dgi_cpf_betafix_sim = let
    dgi_noisyar = load_summary("dgi-cpf-betafix-noisyar-summary.jld2");
    dgi_sv = load_summary("dgi-cpf-betafix-sv-summary.jld2");
    vcat(dgi_noisyar, dgi_sv);
end
dgi_cpf_betafix_sim = join(dgi_cpf_betafix_sim, dgi_configs,
                           on = dgi_paramnames, kind = :left);

dgi_cpf_reps_sv = load_summary("dgi-cpf-betafix-reps-sv-summary.jld2");

# Load CPF-BS results.
cpf_bs_sim = let
    filter(r -> r[:beta] == 1.00, dgi_cpf_betafix_sim);
end

# Load poor mixing example data.
poor_mixing_ex = let
    raw = load_summary("diffinit-poor-mixing-example-summary.jld2");
    map(raw) do o
        DataFrame(iteration = 1:o.M, sigma_x1 = o.sigma_x1,
                  value = o.x1);
    end |> x -> vcat(x...);
end

## Plots for paper

# Plot the example which shows how mixing of CPF-BS deteriorates with increasing
# diffuseness of the initial distribution.
function plot_poor_mixing_example(locations::Vector{<: AbstractString} = String[];
                                  show::Bool = true)
    R"""
    library(ggplot2)
    d <- $poor_mixing_ex
    d[["sigma_x1"]] <- make_numeric_factor(d[["sigma_x1"]], add_str = "sigma[1] == ")
    expand <- rep(0.01, 2)

    plt <- ggplot(d, aes(x = iteration, y = value)) +
    geom_line(size = 0.3) +
    facet_grid(sigma_x1 ~ ., labeller = label_parsed) +
    scale_x_continuous(expand = expand, breaks = seq(0, 6000, by = 1000)) +
    scale_y_continuous(expand = expand) +
    labs(x = "Iteration") +
    theme_faceted +
    theme(axis.title.y = element_blank())

    save_plot(plt, "diffinit-poor-mixing-example",
    h = 3, w = 5, locations = $locations, show = $show)
    """

end
plot_poor_mixing_example(["default"])

# Comparing FDI-CPF (AM) and FDI-CPF (ASWAM) with DPG-BS.
# The parameter configuration (in this case sigma_x) on the x-axis,
# log(IACT) on y-axis and a separate line for each algorithm for each number
# of particles (panels).
function plot_fdi_cpf_vs_dpg(model::AbstractString; locations = String[],
                             show::Bool = false)

    plot_data = let
        d1 = filter(r -> r[:model] == model, copy(fdi_cpf_am)) |>
        x -> select!(x, Not([:adaptation, :last_cov]));
        d1[!, :alg] .= "FDI-CPF (AM)";

        d2 = filter(r -> r[:model] == model, copy(dpg_cpf));
        d2[!, :alg] .= "DPG-BS";

        d3 = filter(r -> r[:model] == model && r[:iact_rank] == 1,
        copy(fdi_cpf_aswam)) |>
        x -> select!(x, Not([:adaptation, :target, :last_log_delta, :ac1_rank,
                             :iact_rank]));
        d3[!, :alg] .= "FDI-CPF (ASWAM, optimal tuning)";
        vcat(d1, d2, d3);
    end

    R"""
    library(ggplot2)
    library(latex2exp)
    d <- $plot_data
    d[["npar_fact"]] <- make_numeric_factor(d[["npar"]], "N = ")
    d[["alg"]] <- factor(d[["alg"]],
    levels = c("DPG-BS", setdiff(unique(d[["alg"]]), "DPG-BS")))

    plt <- ggplot(d, aes(x = factor(sigma_x), y = log(IACT), group = alg, linetype = alg)) +
    geom_line() +
    scale_linetype_manual(values = c("solid", "dashed", "dotted")) +
    facet_wrap(~ npar_fact, nrow = 2) +
    labs(y = TeX("$\\log{(IACT)}$"), x = TeX("$\\sigma_x$")) +
    theme_faceted +
    theme(legend.position = "top",
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 40, hjust = 0.7,
                               size = SIZE_DEFAULTS[["axis_text_size"]] - 1),
    legend.margin = margin(0, 0, -10, 0))

    w <- 5; h <- 2.75
    filename <- paste0($model, "-fdi-cpf-vs-dpg-cpf.pdf")
    save_plot(plt, filename, w = w, h = h, locations = $locations,
              show = $show)
    plt;
    """

end
plot_fdi_cpf_vs_dpg("NOISYAR", locations = ["default"], show = true);
plot_fdi_cpf_vs_dpg("SV", locations = ["default"], show = true);

# Compare FDI-CPF and DPG-BS by plotting mean log(IACT) (y-axis) wrt. target (x-axis).
# Show model with linetype and number of particles with panels.
# Overlay performance of DPG-BS with horizontal lines.
function log_IACT_wrt_tuning(; locations, show::Bool = false)
    plot_data = let
        dpg = by(dpg_cpf, [:model, :npar]) do gdf
                  (mean_log_iact = mean(log.(gdf.IACT)), );
              end;
        fdi = by(fdi_cpf_aswam, [:model, :npar, :target]) do gdf
                  (mean_log_iact = mean(log.(gdf.IACT)), );
              end;
        fdi, dpg;
    end

    R"""
    library(ggplot2)

    d_fdi <- $(plot_data[1])
    d_dpg <- $(plot_data[2])
    model_nm <- c(NOISYAR = "Noisy random walk", SV = "Stochastic volatility")
    d_fdi[["npar_fact"]] <- make_numeric_factor(d_fdi[["npar"]], "N = ")
    d_dpg[["npar_fact"]] <- make_numeric_factor(d_dpg[["npar"]], "N = ")
    d_dpg[["model"]] <- factor(model_nm[d_dpg[["model"]]], levels = unname(model_nm))
    d_fdi[["model"]] <- factor(model_nm[d_fdi[["model"]]], levels = unname(model_nm))

    expand <- 0.01 * rep(1, 2)
    xbreaks <- seq(0.0, 1.0, by = 0.2)
    xlabels <- c("0", as.character(seq(0.2, 0.8, by = 0.2)), "1")
    alpha <- ifelse(d_fdi[["model"]] == "Stochastic volatility", 1.0, 0.7)

    # Make plot.
    plt <- ggplot(d_fdi, aes(x = target, y = mean_log_iact)) +
    geom_line(aes(linetype = model, alpha = model), size = 1) +
    geom_hline(data = d_dpg, aes(yintercept = mean_log_iact, linetype = model),
               show.legend = FALSE) +
    labs(x = "Target", y = "Mean log(IACT)") +
    facet_wrap(~ factor(npar_fact), nrow = 2, ncol = 4) +
    scale_y_continuous(expand = expand) +
    scale_x_continuous(breaks = xbreaks, labels = xlabels,
                       expand = expand) +
    scale_alpha_manual(values = c(0.6, 1.0)) +
    scale_linetype_manual(values = c("solid", "dotted")) +
    theme_faceted +
    theme(legend.position = "top",
          legend.margin = margin(0, 0, -10, 0),
          legend.title = element_blank())

    filename <- paste0("fdi-dpg-log-iact-wrt-tuning.pdf")
    save_plot(plt, filename, w = 5, h = 2.75, locations = $locations,
              show = $show)
    """
end
log_IACT_wrt_tuning(locations = ["default"], show = true);

# Plot log(IACT) (y-axis) with respect to parameter configuration (x-axis).
# Show CPF-BS and DGI-CPF with optimal beta with different linetypes.
# Show differing numbers of particles with panels.
function plot_cpf_vs_opt_beta(model::AbstractString; locations, show = false)
    plot_data = let
        cpf_data = filter(r -> r[:model] == model, cpf_bs_sim);
        cpf_data[!, :log_IACT] .= log.(cpf_data[!, :IACT]);
        cpf_data[!, :alg] .= "CPF-BS";
        #select!(cpf_data, Not(:ts_length));

        dgi_data = filter(r -> r[:model] == model && r[:iact_rank] == 1,
                          dgi_cpf_betafix_sim); #|>
                   #x -> select!(x, Not([:ac1_rank, :iact_rank, :beta]));
        dgi_data[!, :log_IACT] .= log.(dgi_data[!, :IACT]);
        dgi_data[!, :alg] .= "DGI-CPF";
        vcat(cpf_data, dgi_data);
    end
    # Get order of configurations s.t CPF-BS struggles on average more with
    # increasing configuration.
    ordering = let
        d = filter(r -> r[:alg] == "CPF-BS", plot_data);
        d = by(d, :config) do gdf
            (iact_summ = mean(log.(gdf.IACT)),)
        end
        Int.(sort(d, :iact_summ)[:, :config]);
    end

    R"""
    library(ggplot2)
    library(latex2exp)

    d <- $plot_data
    d[["alg"]] <- factor(d[["alg"]], levels = c("CPF-BS", "DGI-CPF"))
    npar_sorted <- sort(unique(d[["npar"]]))
    npar_levels <- structure(paste0("N = ", npar_sorted), .Names = npar_sorted)
    d[["npar_fact"]] <- factor(npar_levels[as.character(d[["npar"]])],
                                levels = unname(npar_levels))

    plt <- ggplot(d, aes(x = factor(config, levels = $ordering), y = log_IACT,
                            linetype = alg)) +
    geom_line(aes(group = alg), alpha = 0.7) +
    facet_wrap(~ npar_fact, ncol = 4, nrow = 2) +
    scale_linetype_manual(labels = list(TeX("CPF-BS"),
                                        TeX("DGI-CPF (best $\\beta$)")),
                          values = c("solid", "dotted")) +
    labs(x = "Parameter configuration", y = "log(IACT)") +
      theme_faceted +
      theme(legend.position = "top",
            legend.title = element_blank(),
            axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            panel.grid = element_blank(),
            legend.margin = margin(c(0, 0, -10, 0)),
            legend.box.margin = unit(rep(0, 4), "pt"),
            legend.spacing = unit(rep(0, 4), "pt"))
     filename <- paste0("dgi-cpf-opt-vs-cpf-bs-", $model, ".pdf")
     save_plot(plt, filename, w = 5, h = 3.25, show = $show, locations = $locations)
    """
end
plot_cpf_vs_opt_beta("NOISYAR"; show = true, locations = ["default"])
plot_cpf_vs_opt_beta("SV"; show = true, locations = ["default"]);


# Plot mean log IACT (y-axis) with respect to target (x-axis) and
# number of particles (panels). Show different models with linetypes.
# Show performance of cpfbs with vertical lines.
function plot_mean_log_iact_wrt_target(; show::Bool = false,
                                         locations)
    plot_data = let
        d_dgi = filter(r -> r[:emp_acc] > 0, dgi_cpf_sim) |>
            x -> by(x, [:model, :target, :npar]) do gdf
            (mean_log_IACT = mean(log.(gdf.IACT)),)
        end;
        d_cpfbs = filter(r -> r[:emp_acc] > 0, cpf_bs_sim) |>
            x -> by(x, [:model, :npar]) do gdf
            (mean_log_IACT = mean(log.(gdf.IACT)),);
        end;
        (d_dgi, d_cpfbs);
    end

    R"""
        library(ggplot2)

        d_dgi <- $(plot_data[1])
        d_cpfbs <- $(plot_data[2])
        model_nm <- c(NOISYAR = "Noisy random walk", SV = "Stochastic volatility")
        d_dgi[["npar_fact"]] <- make_numeric_factor(d_dgi[["npar"]], "N = ")
        d_cpfbs[["npar_fact"]] <- make_numeric_factor(d_cpfbs[["npar"]], "N = ")
        d_dgi[["model"]] <- factor(model_nm[d_dgi[["model"]]], levels = unname(model_nm))
        d_cpfbs[["model"]] <- factor(model_nm[d_cpfbs[["model"]]], levels = unname(model_nm))

        expand <- 0.01 * c(1, 1)
        xbreaks <- seq(0.0, 1.0, by = 0.2)
        xticks <- c("0", as.character(seq(0.2, 0.8, by = 0.2)), "1")

        plt <- ggplot(d_dgi, aes(x = target, y = mean_log_IACT)) +
        geom_line(aes(linetype = model, alpha = model), size = 1) +
        geom_hline(data = d_cpfbs, aes(yintercept = mean_log_IACT, linetype = model),
                   show.legend = FALSE) +
        facet_wrap(~ npar_fact, nrow = 2) +
        scale_x_continuous(expand = expand, breaks = xbreaks, labels = xticks) +
        scale_linetype_manual(values = c("solid", "dotted")) +
        scale_alpha_manual(values = c(0.6, 1.0)) +
        labs(x = "Target", y = "Mean log(IACT)") +
        theme_faceted +
        theme(legend.position = "top",
              legend.margin = margin(0, 0, -10, 0),
              legend.title = element_blank())

        filename <- paste0("dgi-cpf-vs-cpf-bs-target-mean-log-iact.pdf")
        save_plot(plt, filename, w = 5, h = 2.75, show = $show, locations = $locations)
        """
end
plot_mean_log_iact_wrt_target(; show = true, locations = ["default"])

## Plot heatmap with sigma_x and sigma_x1 on the axises, and log(IACT) mapped to
#  fill. Plot CPF-BS and DGI-CPF on separate heatmaps.
function log_iact_heatmap(model::AbstractString, npar::Int = 256; show::Bool = false,
                          locations)

    plot_data = let
        cpfbs = filter(r -> r[:npar] == npar && r[:sigma_x] <= 10.0 &&
                            r[:model] == model, cpf_bs_sim) |>
                x -> select!(x, Not(:beta));
        dgicpf = filter(r -> r[:iact_rank] == 1 && r[:npar] == npar &&
                             r[:sigma_x] <= 10.0 && r[:model] == model, dgi_cpf_sim) |>
                x -> select!(x, Not([:last_beta, :penalty, :target]));
        cpfbs[!, :alg] .= "CPF-BS";
        dgicpf[!, :alg] .= "DGI-CPF";
        vcat(cpfbs, dgicpf);
    end

    R"""
    library(ggplot2)
    library(latex2exp)

    d <- $plot_data
    d[["sigma_x1"]] <- factor(d[["sigma_x1"]])
    d[["sigma_x"]] <- factor(d[["sigma_x"]])

    alg_names <- c(`CPF-BS` = "CPF-BS", `DGI-CPF` = "DGI-CPF~(best~beta)")
    d[["alg"]] <- factor(alg_names[d[["alg"]]], levels = unname(alg_names))

    plt <- ggplot(d, aes(x = sigma_x, y = sigma_x1)) +
    geom_tile(aes(fill = log(IACT))) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    scale_fill_gradient(low = "white", high = "black", name = "log(IACT)") +
    facet_wrap(~ alg, nrow = 1, ncol = 2, labeller = label_parsed) +
    coord_fixed() +
    labs(x = TeX("$\\sigma_x$"), y = TeX("$\\sigma_1$")) +
    guides(fill = guide_colorbar(barheight = 0.5, barwidth = 4,
                                 title.vjust = 1)) +
    theme_faceted +
    theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
          legend.position = "top",
          legend.margin = margin(0, 0, -10, 0),
          legend.text = element_text(size = 6),
          legend.key = element_rect(size = 0.5))

    filename <- paste0("cpf-bs-dgi-cpf-", $model, "-log-iact-heatmap-", "npar",
                       $npar, ".pdf")
    save_plot(plt, filename, h = 2, w = 6, show = $show, locations = $locations)
    """
end
log_iact_heatmap("NOISYAR"; show = true, locations = ["default"]);
log_iact_heatmap("SV"; show = true, locations = ["default"]);

# Plot for the mvnormal model the target rate vs. the mean log IACT (over parameter
# configs) with respect to the number of particles (separate lines) and
# dimension (panels).
function plot_mvnormal_logiact_vs_target(; show::Bool = false,
                                         locations)

    plot_data = by(fdi_cpf_mvnormal, [:statedim, :npar, :target]) do gdf
        (mean_log_iact = mean(log.(gdf.IACT)),);
    end;

    R"""
    library(ggplot2)
    library(RColorBrewer)

    df <- $plot_data
    xbreaks <- ybreaks <- seq(0.0, 1.0, by = 0.2)
    expand <- 0.01 * c(1, 1)
    xlabels <- c("0", as.character(seq(0.2, 0.8, by = 0.2)), "1")
    num_npars <- length(unique(df[["npar"]]))
    colorfun <- colorRampPalette(c("#C6C6C6", "black"))
    nparcolors <- colorfun(num_npars)

    # Set factor leveles for dimension.
    dim <- df[["statedim"]]
    dim_text <- paste0("d = ", dim)
    df[["dim_factor"]] <- factor(dim_text, levels = paste0("d = ",
                            sort(unique(dim))))

    # Make plot.
    plt <- ggplot(df, aes(x = target, y = mean_log_iact)) +
    geom_line(aes(color = factor(npar)), size = 0.5) +
    labs(x = "Target", y = "Mean log(IACT)") +
    facet_wrap(~ dim_factor, nrow = 2, ncol = 5) +
    scale_color_manual(values = nparcolors, name = "N") +
    scale_x_continuous(breaks = xbreaks, labels = xlabels, expand = expand) +
    scale_y_continuous(expand = expand) +
    theme_faceted +
    theme(legend.position = "top",
          axis.text = element_text(size = 6),
          legend.margin = margin(0, 0, -10, 0))

    filename <- "fdi-cpf-aswam-mvnormal-target-vs-mean-log-iact.pdf"
    save_plot(plt, filename, w = 5, h = 3, show = $show, locations = $locations)
    """
end
plot_mvnormal_logiact_vs_target(; show = true, locations = ["default"])

# Plot the optimal alpha (logit(alpha)) wrt the number of particles (x-axis)
# and dimension (panels) based on the mvnormal model analysis.
function plot_alphaopt_wrt_dim_npar(; show::Bool = false,
                                         locations)
    plot_data = let
        by(fdi_cpf_mvnormal, [:statedim, :npar, :sigma]) do gdf
            (iact_rank = ordinalrank(gdf.IACT),
            target = gdf.target);
        end |>
        x -> join(fdi_cpf_mvnormal, x, kind = :left, on = [:statedim, :npar, :sigma, :target]) |>
        x -> filter(r -> r[:iact_rank] == 1 && r[:target] < 1.0, x);
    end

    R"""
    library(ggplot2)
    library(latex2exp)

    d <- $plot_data
    d[["dim_fact"]] <- make_numeric_factor(d[["statedim"]], "d = ")
    logit <- function(x) {
        log(x / (1 - x))
    }
    expand <- rep(0.02, 2)

    plt <- ggplot(d, aes(x = log(npar), y = logit(target))) +
    geom_point(alpha = 0.6, size = 0.5) +
    facet_wrap(~ dim_fact, nrow = 2) +
    #xlim(c(0, NA)) +
    scale_x_continuous(expand = expand, limits = c(0, NA)) +
    scale_y_continuous(expand = expand) +
    labs(x = TeX("$\\log{(N)}$"), y = TeX("$logit(\\alpha_{opt})$")) +
    theme_faceted +
    theme(plot.margin = unit(c(0, 2, 0, 0), "pt"))

    filename <- "alphaopt-vs-log-npar-dim.pdf"
    save_plot(plt, filename, h = 2.75, w = 5, show = $show, locations = $locations)
    """
end
plot_alphaopt_wrt_dim_npar(; show = true, locations = ["default"]);

function plot_dgi_cpf_beta_reps(;show::Bool = false,
                                         locations)
    R"""
    library(ggplot2)
    library(dplyr)
    library(latex2exp)

    d <- $dgi_cpf_reps_sv %>%
    group_by(beta) %>%
    summarise(mean_log_iact = mean(log_IACT))

    d_cpfbs <- filter($dgi_cpf_reps_sv, beta == 1.0) %>%
        group_by(beta) %>%
        summarise(mean_log_iact = mean(log_IACT))
    expand <- rep(0.01, 2)

    plt <- ggplot(d, aes(x = beta, y = mean_log_iact)) +
    geom_point(alpha = 0.6, size = 1) +
    geom_hline(data = d_cpfbs, aes(yintercept = mean_log_iact)) +
    scale_x_continuous(expand = expand) +
    scale_y_continuous(expand = expand) +
    labs(x = TeX("$\\beta$"), y = TeX("Mean $\\log{(IACT)}")) +
    theme_bw() +
    theme(plot.margin = unit(c(0, 2, 0, 0), "pt"),
    axis.text = element_text(size = SIZE_DEFAULTS[["axis_text_size"]]),
    axis.title = element_text(size = SIZE_DEFAULTS[["axis_title_size"]]))

    filename <- paste0("dgi-cpf-mean-log-iact-beta-reps-npar-128-sigmax-1_0",
                       "-sigmax1-50", "-", "sv", ".pdf")
    save_plot(plt, filename, w = 3, h = 2, show = $show, locations = $locations)
    """
end
plot_dgi_cpf_beta_reps(show = true, locations = ["default"])
