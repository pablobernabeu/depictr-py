# R-derived values pinned in tests/test_estimation_tables.py
# (R_LAYOUT, the numeric groups and the captions).
#
# Run from the root of depictr-py against a checkout of the R package that lines
# up the estimation plot's columns:
#
#   Rscript dev/goldens/estimation.R ../depictr
#
# For each call, it prints the reference estimation_plot() takes, the columns of
# the upper panel and the tick labels of the lower one. It then prints the
# caption at the default number of resamples and at 100,000. Needs pkgload,
# ggplot2 and patchwork.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
grDevices::pdf(NULL)
suppressMessages(pkgload::load_all(r_twin, quiet = TRUE))

tick_labels <- function(panel) {
  ggplot2::ggplot_build(panel)$layout$panel_params[[1]]$x$get_labels()
}
quoted <- function(x) paste0("[", paste0('"', x, '"', collapse = ", "), "]")
show_layout <- function(heading, p) {
  cat(heading, "\n")
  cat("  reference:", attr(p, "differences")$reference[1], "\n")
  cat("  upper columns:", quoted(tick_labels(p[[1]])), "\n")
  cat("  lower labels: ", quoted(tick_labels(p[[2]])), "\n")
}

calls <- list(
  list("wellbeing_survey", "life_satisfaction", "region", NULL),
  list("wellbeing_survey", "life_satisfaction", "region", "North"),
  list("crop_yield", "yield", "treatment", NULL),
  list("lexical_decision", "RT", "condition", NULL),
  list("lexical_decision", "RT", "modality", NULL)
)
for (call in calls) {
  # do.call() hands the column names over as strings, which is how
  # estimation_plot() reads a name that is not written out in the call.
  set.seed(1)
  p <- do.call(estimation_plot,
               list(get(call[[1]]), call[[2]], call[[3]],
                    reference = call[[4]], n_boot = 50))
  show_layout(sprintf("%s, %s, %s, reference = %s", call[[1]], call[[2]],
                      call[[3]], if (is.null(call[[4]])) "NULL" else call[[4]]),
              p)
}

# A numeric group, whose levels factor() sorts by value.
numeric_groups <- data.frame(y = seq_len(12), g = rep(c(20, 5, 10), 4))
set.seed(1)
p <- estimation_plot(numeric_groups, y, g, n_boot = 50)
show_layout("numeric groups 20, 5, 10 in that row order, reference = NULL", p)

set.seed(1)
p <- estimation_plot(crop_yield, yield, treatment)
cat("caption at the defaults:", p[[2]]$labels$caption, "\n")
# A group of one skips the bootstrap, so 100,000 resamples cost nothing.
tiny <- data.frame(y = c(1, 2, 3), g = c("a", "a", "b"))
p <- suppressWarnings(estimation_plot(tiny, y, g, n_boot = 1e5))
cat("caption at 1e5 resamples:", p[[2]]$labels$caption, "\n")
