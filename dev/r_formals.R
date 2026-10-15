# Write the R twin's exported signatures for the parity manifest.
#
# Run from the root of depictr-py, naming a checkout of the R package:
#
#   Rscript dev/r_formals.R ../depictr
#
# Writes tests/data/r_formals.json: the version of the R package read and, for
# each function it exports, the names of its formal arguments in order. The
# checkout is loaded with pkgload, so the file describes the source named here
# whatever version of depictr is installed. tests/test_parity_manifest.py
# compares the Python signatures with this file through an explicit map of
# renamed arguments and lists of what only one twin has. Rerun the script
# whenever the R twin's signatures change, and update the manifest and the
# README's Status section in the same commit. Needs pkgload and jsonlite.

args <- commandArgs(trailingOnly = TRUE)
r_twin <- if (length(args) >= 1) args[[1]] else "../depictr"
if (!file.exists(file.path(r_twin, "DESCRIPTION"))) {
  stop("No DESCRIPTION under ", normalizePath(r_twin, mustWork = FALSE),
       ". Pass the path to a checkout of the depictr R package.")
}
if (!dir.exists(file.path("tests", "data"))) {
  stop("Run this script from the root of depictr-py.")
}
pkgload::load_all(r_twin, export_all = FALSE, helpers = FALSE, quiet = TRUE)

ns <- asNamespace("depictr")
# The radix method sorts in the C locale, so the order is the same everywhere.
exports <- sort(getNamespaceExports("depictr"), method = "radix")
functions <- list()
for (name in exports) {
  obj <- get(name, envir = ns)
  if (is.function(obj)) {
    # I() keeps a one-argument signature an array under auto_unbox.
    functions[[name]] <- I(as.character(names(formals(obj))))
  }
}

desc <- read.dcf(file.path(r_twin, "DESCRIPTION"), fields = "Version")
json <- jsonlite::toJSON(
  list(source = paste0("depictr ", desc[1, "Version"], " (R)"),
       functions = functions),
  auto_unbox = TRUE, pretty = TRUE
)
path <- file.path("tests", "data", "r_formals.json")
con <- file(path, open = "wb")
writeLines(json, con, sep = "\n", useBytes = TRUE)
close(con)

cat("Wrote the signatures of", length(functions), "functions to", path, "\n")
