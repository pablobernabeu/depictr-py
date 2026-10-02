# Export the R package's bundled datasets for the Python twin.
#
# Run from the root of depictr-py, naming a checkout of the R package:
#
#   Rscript dev/export_r_datasets.R ../depictr
#
# The R twin's data/*.rda files are the single source of the datasets. This
# script writes each one to src/depictr/data/<name>.csv and records the column
# types R holds, which a CSV cannot carry, in src/depictr/data/levels.json:
# the storage type of every column and, for each factor, its levels in order and
# whether it is ordered. depictr.data reads both files and rebuilds the same
# frames in pandas.
#
# The CSVs are written with "\n" line endings and an empty field for a missing
# value, so the files are byte-identical on every platform. tests/test_data.py
# pins their SHA-256, so rerun this script whenever the R data change and update
# the pins in the same commit.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
data_dir <- file.path(r_twin, "data")
if (!dir.exists(data_dir)) {
  stop("No data/ folder under ", normalizePath(r_twin, mustWork = FALSE),
       ". Pass the path to a checkout of the depictr R package.")
}
out_dir <- file.path("src", "depictr", "data")
if (!dir.exists(file.path("src", "depictr"))) {
  stop("Run this script from the root of depictr-py.")
}
dir.create(out_dir, showWarnings = FALSE)

datasets <- c("crop_yield", "wellbeing_survey", "lexical_decision",
              "clinical_trial", "monthly_sales")

column_spec <- function(x) {
  if (is.factor(x)) {
    list(type = "factor", levels = levels(x), ordered = is.ordered(x))
  } else if (inherits(x, "Date")) {
    list(type = "Date")
  } else if (is.integer(x)) {
    list(type = "integer")
  } else if (is.double(x)) {
    list(type = "double")
  } else if (is.character(x)) {
    list(type = "character")
  } else {
    stop("No CSV mapping for a column of class ", class(x)[1])
  }
}

schema <- list()
for (name in datasets) {
  env <- new.env()
  load(file.path(data_dir, paste0(name, ".rda")), envir = env)
  df <- env[[name]]
  schema[[name]] <- lapply(df, column_spec)
  # A character field that held the separator or a quote would need escaping
  # rules the loader does not apply, so refuse it here.
  text <- vapply(df, function(x) is.character(x) || is.factor(x), logical(1))
  for (col in names(df)[text]) {
    if (any(grepl("[\",\n]", as.character(df[[col]])))) {
      stop(name, "$", col, " holds a comma, quote or newline.")
    }
  }
  path <- file.path(out_dir, paste0(name, ".csv"))
  con <- file(path, open = "wb")
  utils::write.table(df, con, sep = ",", quote = FALSE, na = "",
                     row.names = FALSE, eol = "\n", fileEncoding = "UTF-8")
  close(con)
}

desc <- read.dcf(file.path(r_twin, "DESCRIPTION"), fields = "Version")
json <- jsonlite::toJSON(
  list(
    source = paste0("depictr ", desc[1, "Version"], " (R), data/*.rda"),
    datasets = schema
  ),
  auto_unbox = TRUE, pretty = TRUE
)
# toJSON unboxes a one-element vector, so a factor with a single level would
# lose its array. No dataset has one, and the check stops the export if one
# ever does.
stopifnot(all(unlist(lapply(schema, function(s) {
  vapply(s, function(col) is.null(col$levels) || length(col$levels) > 1, logical(1))
}))))
con <- file(file.path(out_dir, "levels.json"), open = "wb")
writeLines(json, con, sep = "\n", useBytes = TRUE)
close(con)

cat("Wrote", length(datasets), "datasets and levels.json to", out_dir, "\n")
