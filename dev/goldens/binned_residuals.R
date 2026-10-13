# R-derived values pinned in tests/test_diagnostics.py.
#
# Run from the root of depictr-py against a checkout of the R package whose
# binned_residual_plot() takes its band from the model (change set CS28):
#
#   Rscript dev/goldens/binned_residuals.R ../depictr
#
# Prints the per-bin summary that binned_residual_plot() attaches to its plot
# for four fits. The first two are the R twin's documented example and a model
# whose fitted values are tied within each arm. The other two fit a grouped
# model with a (successes, failures) response, as a binomial model and as a
# quasibinomial one, whose band carries the estimated dispersion. Each row
# gives the mean fitted value, the mean residual, the count and the half-width
# of the band (se2) under the default band = "model". The rows of the first
# fit add the half-width under band = "empirical". Needs pkgload and ggplot2.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
grDevices::pdf(NULL)
suppressMessages(pkgload::load_all(r_twin, quiet = TRUE))

show_bins <- function(label, fit, empirical = FALSE) {
  bd <- attr(binned_residual_plot(fit), "bins")
  if (empirical) {
    bd$se2_empirical <- attr(binned_residual_plot(fit, band = "empirical"),
                             "bins")$se2
  }
  cat(label, "\n")
  for (i in seq_len(nrow(bd))) {
    values <- c(bd$fitted[i], bd$resid[i], bd$n[i], bd$se2[i],
                if (empirical) bd$se2_empirical[i])
    cat(sprintf("    (%s),\n", paste(sprintf("%.10f", values), collapse = ", ")))
  }
}

show_bins("adverse_event ~ biomarker + age + arm: fitted, resid, n, se2, se2_empirical",
          glm(adverse_event ~ biomarker + age + arm, data = clinical_trial,
              family = binomial),
          empirical = TRUE)

# Two fitted values, one per arm, so ranks within an arm follow row order.
show_bins("adverse_event ~ arm: fitted, resid, n, se2",
          glm(adverse_event ~ arm, data = clinical_trial, family = binomial))

# A grouped response built with integer arithmetic, which the Python test
# repeats exactly.
i <- 1:40
grouped <- data.frame(x = (i - 20.5) / 10, trials = 4 + i %% 9)
grouped$successes <- (grouped$trials * (i %% 11)) %/% 10
show_bins("cbind(successes, trials - successes) ~ x: fitted, resid, n, se2",
          glm(cbind(successes, trials - successes) ~ x, data = grouped,
              family = binomial))

# The Python test fits this one with fit(scale="X2").
show_bins("quasibinomial cbind(successes, trials - successes) ~ x: fitted, resid, n, se2",
          glm(cbind(successes, trials - successes) ~ x, data = grouped,
              family = quasibinomial))
