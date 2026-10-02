# R-derived values pinned in tests/test_data.py and tests/test_mixed.py.
#
# Run from the root of depictr-py against a checkout of the R package at the
# commit whose data/*.rda were exported:
#
#   Rscript dev/goldens/datasets.R ../depictr
#
# It first checks that every CSV under src/depictr/data reads back equal to the
# R data it was exported from, then prints the summaries the Python tests pin.
# Needs the survival and lme4 packages.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
load_rda <- function(name) {
  env <- new.env()
  load(file.path(r_twin, "data", paste0(name, ".rda")), envir = env)
  env[[name]]
}

datasets <- c("crop_yield", "wellbeing_survey", "lexical_decision",
              "clinical_trial", "monthly_sales")
for (name in datasets) {
  df <- load_rda(name)
  csv <- utils::read.csv(file.path("src", "depictr", "data", paste0(name, ".csv")),
                         na.strings = "", stringsAsFactors = FALSE)
  for (col in names(df)) {
    original <- df[[col]]
    if (is.factor(original) || inherits(original, "Date")) {
      original <- as.character(original)
    }
    stopifnot(isTRUE(all.equal(csv[[col]], original, tolerance = 0,
                               check.attributes = FALSE)))
  }
  cat(sprintf("%s: %d x %d\n", name, nrow(df), ncol(df)))
  numeric <- vapply(df, is.numeric, logical(1))
  sums <- vapply(df[numeric], function(x) sum(x, na.rm = TRUE), numeric(1))
  cat("  column sums:", paste(sprintf("%s=%.6f", names(sums), sums),
                              collapse = ", "), "\n")
  missing <- vapply(df, function(x) sum(is.na(x)), integer(1))
  if (any(missing > 0)) {
    cat("  missing:", paste(sprintf("%s=%d", names(missing)[missing > 0],
                                    missing[missing > 0]), collapse = ", "), "\n")
  }
}

ct <- load_rda("clinical_trial")
lr <- survival::survdiff(survival::Surv(time, event) ~ arm, data = ct)
cat(sprintf("clinical_trial: %d events, log-rank chi-squared by arm %.4f\n",
            sum(ct$event), lr$chisq))

ld <- load_rda("lexical_decision")
fit <- lme4::lmer(RT ~ condition + (1 | participant), data = ld, REML = TRUE)
vc <- as.data.frame(lme4::VarCorr(fit))
cat(sprintf("lexical_decision: participant intercept variance (REML) %.4f\n",
            vc$vcov[vc$grp == "participant"]))
