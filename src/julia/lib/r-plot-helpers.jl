# Some R code with interpolated Julia values for helping make plots
# in src/julia/scripts

using RCall
R"""
## Some themes and helpers for making plots.
library(ggplot2)

# Defaults for various sizes in plots.
SIZE_DEFAULTS <- list(
  linewidth = 0.25,
  pointsize = 0.5,
  axis_text_size = 5,
  axis_title_size = 5,
  strip_text_size = 5,
  legend_text_size = 5,
  legend_title_size = 5
)

# Default theme for faceted plot.
theme_faceted <- with(SIZE_DEFAULTS, {
    theme_bw() +
    theme(panel.spacing = unit(1, "pt"),
          plot.margin = unit(c(0, 2, 0, 2), "pt"),
          axis.text = element_text(size = axis_text_size),
          axis.title = element_text(size = axis_title_size),
          strip.text = element_text(size = strip_text_size),
          legend.title = element_text(size = legend_title_size),
          legend.text = element_text(size = legend_text_size))
})

# Make a numeric factor out of numeric vector `x`.
# `add_str` can be used to add a certain string to each level of the factor.
make_numeric_factor <- function(x, add_str = "") {
   sorted_uniq_names <- paste0(add_str, sort(unique(x)))
   ordered_names <- paste0(add_str, x)
   factor(ordered_names, levels = sorted_uniq_names)
}
"""
