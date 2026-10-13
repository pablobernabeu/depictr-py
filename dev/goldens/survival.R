# R-derived values pinned in tests/test_plots.py (the Kaplan-Meier limit
# parity tests).
#
# Run from the root of depictr-py against a checkout of the R package whose
# survival_plot() has the conf_type argument (change set CS37):
#
#   Rscript dev/goldens/survival.R ../depictr
#
# Prints the lower and upper confidence limits that the R twin's
# survival_plot() draws for each clinical_trial arm at months 6, 12, 18, 24 and
# 30: the log-log limits at 95%, the default, and the log and plain limits at
# 90%. It then prints the 95% limits at times 0 to 5 of a small curve censored
# at time 1, before any event, whose last event, at time 5, leaves nobody at
# risk. Each value is the limit in force at that time on the drawn step curve.
# Needs pkgload and ggplot2, but not the survival package.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
grDevices::pdf(NULL)
suppressMessages(pkgload::load_all(r_twin, quiet = TRUE))

values <- function(x) paste(sprintf("%.10f", x), collapse = ", ")

# The limits in force at `times` on a drawn step curve: those of the last row
# at or before each time.
in_force <- function(curve, times) {
  curve <- curve[order(curve$time), , drop = FALSE]
  i <- findInterval(times, curve$time)
  list(lower = curve$lower[i], upper = curve$upper[i])
}

months <- c(6, 12, 18, 24, 30)
specs <- list(list("log-log", 0.95), list("log", 0.9), list("plain", 0.9))
for (spec in specs) {
  p <- survival_plot(clinical_trial$time, clinical_trial$event,
                     group = clinical_trial$arm, conf_type = spec[[1]],
                     conf_level = spec[[2]])
  cat(sprintf("%s, conf_level = %s\n", spec[[1]], spec[[2]]))
  for (arm in levels(p$data$group)) {
    lim <- in_force(p$data[p$data$group == arm, ], months)
    cat("  ", arm, " lower: ", values(lim$lower), "\n", sep = "")
    cat("  ", arm, " upper: ", values(lim$upper), "\n", sep = "")
  }
}

cat("ends of a curve, conf_level = 0.95, times 0 to 5\n")
for (type in c("log-log", "log", "plain")) {
  p <- survival_plot(c(1, 2, 3, 4, 5), c(0, 1, 0, 1, 1), conf_type = type)
  lim <- in_force(p$data, 0:5)
  cat("  ", type, " lower: ", values(lim$lower), "\n", sep = "")
  cat("  ", type, " upper: ", values(lim$upper), "\n", sep = "")
}
