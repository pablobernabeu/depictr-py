# R-derived values pinned in tests/test_timeseries.py.
#
# Run from the root of depictr-py against a checkout of the R package:
#
#   Rscript dev/goldens/timeseries.R ../depictr
#
# Prints AirPassengers, which the Python test holds as a literal, and the
# partial autocorrelations at lags 1 to 21 that the R twin's
# acf_plot(type = "partial") draws for it with stats::pacf(). It then prints
# the R twin's centred moving averages of 1:10 over windows of 4 and 3. Needs
# pkgload, because moving_average() is internal to the R package.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
pkgload::load_all(r_twin, quiet = TRUE)

air <- as.numeric(AirPassengers)
cat(sprintf("AirPassengers: %d values summing to %d\n", length(air),
            as.integer(sum(air))))
for (year in split(air, rep(seq_len(length(air) / 12), each = 12))) {
  cat(paste(year, collapse = ", "), ",\n", sep = "")
}

pacf_drawn <- acf_plot(AirPassengers, type = "partial")$data$acf
cat(sprintf("Partial autocorrelations at lags 1 to %d:\n", length(pacf_drawn)))
cat(paste0(sprintf("%.15f", pacf_drawn), ",", collapse = "\n"), "\n", sep = "")

for (window in c(4, 3)) {
  ma <- moving_average(1:10, window)
  cat(sprintf("moving_average(1:10, %d): %s\n", window,
              paste(ifelse(is.na(ma), "NA", sprintf("%g", ma)), collapse = " ")))
}
