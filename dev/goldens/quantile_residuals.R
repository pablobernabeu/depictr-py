# R-derived values pinned in tests/test_diagnostics_panels.py.
#
# Run from the root of depictr-py against a checkout of the R package:
#
#   Rscript dev/goldens/quantile_residuals.R ../depictr
#
# Prints the quantile residuals that the R twin's quantile_residuals() gives a
# weighted Gaussian glm and a Gamma glm fitted to crop_yield. Neither family is
# randomised, so the Python port has to reproduce them. Needs pkgload, because
# quantile_residuals() is internal to the R package.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
pkgload::load_all(r_twin, quiet = TRUE)

cy <- crop_yield
# Weights 1, 2, 3, 1, 2, 3, ..., which np.resize() builds in the Python test.
cy$w <- rep_len(1:3, nrow(cy))
# At glm()'s default epsilon of 1e-8, the Gamma fit stops six iterations in,
# about 1e-5 short of the estimate statsmodels converges to. A tighter
# criterion takes both twins to the same estimate, so the values below test
# the residuals alone.
tight <- glm.control(epsilon = 1e-14, maxit = 100)
fits <- list(
  "gaussian, weights 1:3" = glm(yield ~ rainfall + fertiliser, data = cy,
                                weights = w, control = tight),
  "Gamma, log link" = glm(yield ~ rainfall + fertiliser, data = cy,
                          family = Gamma(link = "log"), control = tight)
)
for (label in names(fits)) {
  r <- as.numeric(quantile_residuals(fits[[label]]))
  cat(sprintf("%s: first five %s\n", label,
              paste(sprintf("%.10f", r[1:5]), collapse = ", ")))
  cat(sprintf("%s: sum of squares %.10f\n", label, sum(r^2)))
}
