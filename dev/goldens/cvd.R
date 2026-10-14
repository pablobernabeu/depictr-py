# R-derived colours pinned in tests/test_palette_cvd.py.
#
# Run from the root of depictr-py against a checkout of the R package:
#
#   Rscript dev/goldens/cvd.R ../depictr
#
# Prints the default palette as the R twin's simulate_cvd() shows it at
# severities 0.25, 0.5 and 0.75, the PARTIAL_SEVERITY_GOLDEN of the Python
# test. When colorspace is installed, the script first confirms that its
# simulation gives the same colours.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
pkgload::load_all(r_twin, quiet = TRUE)

# The Okabe-Ito colours in the order depictr_palette() gives them by default.
okabe_ito <- c("#005b96", "#e69f00", "#009e73", "#d55e00", "#cc79a7",
               "#56b4e9", "#f0e442", "#999999")
for (severity in c(0.25, 0.5, 0.75)) {
  for (deficiency in c("protan", "deutan", "tritan")) {
    ours <- simulate_cvd(okabe_ito, deficiency, severity = severity)
    if (requireNamespace("colorspace", quietly = TRUE)) {
      simulate <- getExportedValue("colorspace", deficiency)
      stopifnot(identical(ours, tolower(simulate(okabe_ito,
                                                 severity = severity))))
    }
    cat(sprintf("(\"%s\", %s): [%s],\n", deficiency, format(severity),
                paste0("\"", ours, "\"", collapse = ", ")))
  }
}
