# R-derived values pinned in tests/test_diagnostics.py.
#
# Run from the root of depictr-py against a checkout of the R package:
#
#   Rscript dev/goldens/vif.R ../depictr
#
# Prints the generalised VIFs that the R twin's gvif_terms(), like car::vif(),
# gives three models: an lm of wellbeing_survey with two factors, the same lm
# with weights and a logistic glm of clinical_trial. Needs pkgload, because
# gvif_terms() is internal to the R package.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
pkgload::load_all(r_twin, quiet = TRUE)

ws <- wellbeing_survey
# Weights 1, 2, 3, 1, 2, 3, ..., which np.resize() builds in the Python test.
ws$w <- rep_len(1:3, nrow(ws))
# At glm()'s default epsilon of 1e-8, the fit stops a step short of the
# estimate statsmodels converges to, which moves the VIFs in the sixth decimal
# place. A tighter criterion takes both twins to the same estimate.
tight <- glm.control(epsilon = 1e-14, maxit = 100)
fits <- list(
  "lm" = lm(life_satisfaction ~ stress + region + education, data = ws),
  "lm, weights 1:3" = lm(life_satisfaction ~ stress + region + education,
                         data = ws, weights = w),
  "binomial glm" = glm(adverse_event ~ biomarker + age + arm,
                       family = binomial, data = clinical_trial,
                       control = tight)
)
for (label in names(fits)) {
  res <- gvif_terms(fits[[label]])
  for (i in seq_len(nrow(res))) {
    cat(sprintf("%s: %s df %d GVIF %.11f GVIF^(1/(2 df)) %.11f\n", label,
                res$term[i], res$df[i], res$gvif[i], res$gvif_adj[i]))
  }
}
