# R-derived values pinned in tests/test_plots.py (the gains, lift and
# calibration parity tests).
#
# Run from the root of depictr-py against a checkout of the R package:
#
#   Rscript dev/goldens/classification.R ../depictr
#
# Both score vectors are built from integers by the same few floating-point
# operations as in the Python tests, so those tests rebuild them exactly
# without a random number generator. The script prints the R twin's gains
# table and lift points for a vector with four tied scores, then the bins and
# Wilson limits that calibration_plot() draws for a continuous and for the
# tied vector. Needs pkgload and ggplot2, because gain_table() is internal to
# the R package.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
grDevices::pdf(NULL)
suppressMessages(pkgload::load_all(r_twin, quiet = TRUE))

values <- function(x) paste(sprintf("%.10f", x), collapse = ", ")

# Four risk bands, as a rounded or banded score gives: 23, 41, 57 and 79 rows
# at 0.85, 0.55, 0.30 and 0.10, holding 17, 22, 18 and 7 of the 64 positives.
# A fixed stride then spreads each band through the rows, row i taking row
# (77 * (i - 1)) %% 200 + 1 of the banded layout.
band_n <- c(23, 41, 57, 79)
band_pos <- c(17, 22, 18, 7)
tied_score <- rep(c(0.85, 0.55, 0.30, 0.10), band_n)
tied_actual <- unlist(Map(function(n, k) rep(c(1, 0), c(k, n - k)),
                          band_n, band_pos))
stride <- (77 * (seq_len(200) - 1)) %% 200 + 1
tied_score <- tied_score[stride]
tied_actual <- tied_actual[stride]

g <- gain_table(tied_actual, tied_score)
cat("gains, tied vector\n")
cat("  population:", values(g$population), "\n")
cat("  captured:  ", values(g$captured), "\n")
lift_layer <- Filter(function(l) "lift" %in% names(l$data),
                     lift_plot(tied_actual, tied_score)$layers)[[1]]
cat("lift, tied vector\n")
cat("  population:", values(lift_layer$data$population), "\n")
cat("  lift:      ", values(lift_layer$data$lift), "\n")

# A continuous score that crowds towards zero, as the predicted probabilities
# of a rare outcome do, with an outcome drawn by comparing each score with a
# fixed sequence of fractions in place of random draws.
i <- seq_len(300)
x <- i / 300
cont_score <- 0.9 * x * x
cont_actual <- as.integer(((37 * i) %% 101) / 101 < cont_score)

show_bins <- function(heading, p) {
  d <- p$data
  cat(heading, "\n")
  cat("  predicted:", values(d$predicted), "\n")
  cat("  observed: ", values(d$observed), "\n")
  cat("  successes:", paste(d$successes, collapse = ", "), "\n")
  cat("  n:        ", paste(d$n, collapse = ", "), "\n")
  cat("  conf.low: ", values(d$conf.low), "\n")
  cat("  conf.high:", values(d$conf.high), "\n")
}
show_bins("calibration, continuous vector, bins = 10, conf_level = 0.95",
          calibration_plot(cont_actual, cont_score))
show_bins("calibration, continuous vector, bins = 4, conf_level = 0.9",
          calibration_plot(cont_actual, cont_score, bins = 4, conf_level = 0.9))
show_bins("calibration, tied vector, bins = 10, conf_level = 0.95",
          calibration_plot(tied_actual, tied_score))
