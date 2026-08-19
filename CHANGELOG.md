# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`check_figure()`, an accessibility and honesty audit of the figure you are about
  to submit.** Until now the package could vouch for its palette and say nothing
  about a finished plot, which is the thing a reader actually sees. Give it anything
  a depictr function returns, including a plot extended afterwards with `+`, and it
  introspects the build and returns a tidy DataFrame: the separability of the
  encoding colours under each dichromacy and in greyscale, the smallest text size
  against a stated physical output width, the WCAG contrast of the text and of the
  geometry against their backgrounds, and whether any distinction is carried by
  colour alone. Every row carries the value it measured beside the threshold it was
  measured against, so a verdict can be argued with. The R twin gained the same
  function, with the same check names, thresholds, verdicts and measured numbers.

### Changed

- `simulate_cvd()` names the three deficiencies in its refusal message rather than
  printing a Python tuple, so the wording matches the R twin byte for byte.
- The sRGB-to-XYZ matrix behind the CIE Lab conversion is now given to seven decimal
  places instead of four. The rounded form shifted a Delta-E by a few thousandths,
  enough to round a reported distance differently from the R twin on the same
  colours. The distances `palette_safety()` reports for the default palette are
  unchanged at two decimal places.
- **The accessibility claim has been narrowed to what is true.** The default
  eight-colour palette clears every colour-vision check and fails the new greyscale
  check: its orange (`#e69f00`) and sky blue (`#56b4e9`) differ by 0.79 in CIE
  lightness, so a black-and-white printer renders them as the same grey. The
  Okabe-Ito guarantee is about hue confusion and was never a claim about greyscale.
  The threshold has been left where it is rather than moved so the package's own
  defaults pass, and the accessibility page now states the limitation where the
  claim is made.

### Fixed

- **`depictr_palette()` interpolated past its accessibility guarantee without saying so.**
  Beyond the eight Okabe-Ito base colours the palette is a ramp, and the
  colour-vision-deficiency guarantee that is this package's reason for existing stops
  holding; the interpolated palette fails the package's own `palette_safety()` check. It
  now warns at the point of interpolation. The R twin carried the same silence and was
  fixed with it.
- **`survival_plot()` drew a phantom arm for a group that does not exist.** A missing
  value in `group` became a level matching no observation. Missing groups are now
  dropped with a count, and an all-missing group is an error.
- **`interaction_plot()` drew a missing grouping value as a literal `nan` line and legend
  entry** — the same "a missing value stringified into a legitimate-looking value"
  defect fixed across the family this round.
- **`summary_table()` counted missing-group records in `Overall` and in no group column**,
  so the per-group sizes silently fell short of the headline N; they now get a `Missing`
  column. Its group columns are also ordered by level rather than by row appearance, so
  the table no longer depends on input row order.
- **`seasonal_plot()` ordered its cycles lexicographically**, sorting cycle 10 between 1
  and 2, and coloured them categorically rather than with the sequential ramp. An annual
  index also inferred a seasonal period of 1; an explicit `period` is now required.
- Degenerate inputs that returned a confident wrong answer now refuse or abstain:
  `estimation_plot(two_panel=True)` omits the interval for a single-observation group
  instead of drawing a zero-width one that implies perfect precision; `palette_safety()`
  rejects a palette of fewer than two colours rather than reporting it safe at infinite
  distance; `gain_plot()` and `lift_plot()` refuse a single-class outcome instead of
  substituting a denominator of 1 and drawing a flat curve; `survival_plot()` raises a
  clear error for a non-finite follow-up time instead of a bare `StopIteration`; and
  `correlation_heatmap()` drops zero-variance columns with a message and labels an
  undefined correlation `n/a` rather than rendering the string `nan` in the cell.
- **`acf_plot()`, `decompose_plot()` and `seasonal_plot()` silently dropped internal
  missing values**, closing up the gap: the ACF correlated values across it, the
  decomposition misaligned, and every post-gap observation landed on the wrong
  within-period position. An internal missing value is now an error, in the wording of
  `survival_plot()`'s refusal of a non-finite follow-up time; missing values at either
  end are still trimmed, since trimming only shortens the series.
- `acf_plot()` turns an all-missing or empty series away by name, as `seasonal_plot()`
  already did. Its default lag count took the base-10 logarithm of the length, so a
  series with nothing in it surfaced as an `OverflowError` from the surrounding `int()`,
  naming neither the argument nor the problem.
- **`roc_curve_plot()`, `pr_curve_plot()` and `threshold_plot()` accepted a single-class
  outcome**, annotating an AUC of nan or an average precision of 0.000 as if they
  measured something (scikit-learn computes both under a warning rather than refusing).
  They now raise the same refusal `gain_plot()` and `lift_plot()` already carry, with
  the R twin's wording per curve.
- **`dendrogram_plot()` needed scikit-learn despite its documented scipy-only
  contract.** Standardisation went through sklearn's `StandardScaler`; it is now a
  numpy z-score with identical semantics (population standard deviation, and a scale of
  1 substituted for a zero-variance column), so the dendrogram runs on the core install
  and the sklearn-backed plots see the same input as before.
- **The installation docs called statsmodels optional, but plotnine 0.15 requires it**,
  so every core install already ships it. The README and the docs now name scikit-learn
  and lifelines as the genuinely optional back-ends, with `depictr[models]` kept to pin
  the tested statsmodels floor.
- `summary_table()`'s documentation promised to sort the levels of an unordered
  categorical, when any categorical keeps its declared category order (the intended
  behaviour); the docs now say so.
- `posterior_plot()` warns when a `labels` key matches no parameter, instead of returning
  an unrelabelled figure.
- **Three declared dependency floors described a stack no install could produce.**
  `pandas>=2.0`, `matplotlib>=3.6` and `scipy>=1.7` all sat below what
  `plotnine>=0.15` already requires (2.2, 3.8 and 1.8 respectively), and the
  `models` extra named `statsmodels>=0.14` where plotnine forces 0.14.5. A
  resolver always took the higher bound, so no install was ever affected, but a
  floor is a claim about what has been tested and this one was false. The floors
  now state what plotnine forces, and a new CI job installs them.

### Added

- **CI tests what users install, not only the checkout.** The matrix gains Python
  3.14. A new job installs the built wheel into a bare environment outside the
  repository and imports it there, so packaged data and distribution metadata are
  exercised rather than masked by the source tree an editable install sits beside.
  A second new job installs the declared minimum dependency versions. A weekly
  schedule runs the suite when nobody has pushed, so upstream drift in the
  plotting stack surfaces as a dated red badge rather than a surprise.
- **Linting, matching the rest of the family.** This was the only Python package
  in the family running no linter, which is why it had quietly accumulated seven
  findings. The rule set is stated explicitly in `pyproject.toml` (`E, F, W, I,
  UP, B`, as in the scopusflow and theoryforge twins) rather than inherited from
  whatever ruff defaults to: an inherited default is what turned a sibling's
  green CI red without a line of that package changing. The job lints the whole
  repository rather than `src` and `tests` by name, which had left
  `app/streamlit_app.py` and `docs/_exec.py` covered by nothing; the app was
  carrying an over-long line that no gate could see. Naming directories leaves a
  new top-level script unlinted until somebody remembers to extend the list.

### Changed

- Two `zip()` calls now pass `strict=True`: the dendrogram builder pairs scipy's
  `icoord` with `dcoord`, and the forest plot pairs two columns of one frame. In
  both, unequal lengths are impossible by construction, so raising is the right
  response to the impossible happening rather than silently truncating a plot.
  Import blocks were sorted and one long line wrapped; no behaviour changed.

## [0.2.2] - 2026-07-23

### Added

- `calibration_plot` gained the Parameters and Returns sections it lacked,
  including a note that `y_score` must hold fitted probabilities.

### Fixed

- The `calibration_plot` example fits a logistic regression and passes its
  predicted probabilities. A reliability curve compares predicted probability
  with observed frequency, so the previous hand-written score misstated the
  calibration it was meant to demonstrate.
- The `seasonal_plot` example carries trend and noise. Its series was a
  noiseless sine, so all ten cycles coincided exactly and the figure showed one
  curve behind a ten-entry legend.

## [0.2.1] - 2026-07-15

### Added

- The gallery now covers every plot family, matching the R package's vignettes.

### Fixed

- Corrected the `palette_safety()` result shape shown in the README, which left
  out `worst_condition` and `worst_pair`. The function returns those as well as
  the elements already listed.

## [0.2.0] - 2026-07-10

### Changed

- Raised the plotnine dependency floor to 0.15, the first release with the
  plot composition operators (`|`, `/`) that `arrange_plots` uses.
- `arrange_plots`, and the multi-panel reports built on it, warn when a
  `title` is dropped because plotnine compositions cannot carry a figure-level
  title. Previously the argument was discarded silently.

### Fixed

- A missing value in a grouped column crashed at draw time. The default colour
  for missing (`NA`) levels was `grey80`, an R colour name that matplotlib
  rejects, and it is now the equivalent hex `#cccccc`.
- Error messages named nonexistent extras. The ImportError messages and module
  docstrings in the diagnostics, mixed-effects and multivariate modules pointed
  at `depictr[diagnostics]`, `depictr[mixed]` and `depictr[multivariate]`, none
  of which is defined. They now name `depictr[models]` and
  `depictr[classification]` (scipy is a core dependency).

## [0.1.1] - 2026-07-08

### Added

- Added an opt-in `legend_inside=False` parameter to `explore_distribution`,
  `ecdf_plot`, `dumbbell_plot`, `missingness_map` and `survival_plot`, plus the
  public `legend_inside()` theme helper, which places the legend inside the
  panel (over a light background) rather than in a right-hand margin.
- Added a 'Getting started' guide that walks through a short analysis end to end.

### Changed

- Rebuilt the number-at-risk table beneath `survival_plot(risk_table=True)`.
  The curves now use the full panel width (no left-hand gutter), the group names
  label the rows on the y-axis, and the counts are coloured to match the curves,
  forming a tidy strip under the curves rather than text floating in loosely
  spaced negative space.
- The log-rank *p*-value follows APA style (an italicised *p*, no leading zero,
  and *p* < .001 reported below that threshold). The colour legend and the
  risk-table rows now list the groups in the same order (a user-set categorical
  order, otherwise first appearance).
- The README and the PyPI project page now open with a gallery (a grouped
  density and Kaplan-Meier curves), and the documentation landing page gains the
  same hero plot and a PyPI install link.
- README image assets are kept out of the source distribution.

### Fixed

- Grouped histograms were invisible. `explore_distribution(kind="both"`
  or `"histogram")` with a `group` drew no bars at all, because
  `geom_histogram(fill=None)` made them fully transparent instead of deferring
  to the group colour mapping.
- Axis and legend titles leaked raw column names. Several plots that
  meant to leave a title blank (`labs(x=None, ...)`) instead showed the mapped
  column's literal name (`x`, `value`, `variable`, `term`, `metric`, ...) because
  this plotnine version treats `None` as unset rather than blank. Corrected 14
  call sites across `diagnostics`, `eda`, `estimation`, `mixed`, `models`,
  `multivariate`, `posterior`, `predictions`, `distributions_extra`,
  `timeseries` and `classification`.
- The grouped risk-table path on `survival_plot(risk_table=True)` added the
  colour scale twice, which plotnine warned about and silently replaced with an
  identical one. It is now added once.
- Corrected the Cook (1977) reference title and added DOIs to Cook (1977),
  Hedges (1981) and Allen et al. (2021).

## [0.1.0] - 2026-06-27

First release. depictr (Python) is a unified, colourblind-safe toolkit for
publication-ready statistical visualisation, built on plotnine and the Python
sibling of the depictr R package. It gives one consistent theme and calling
convention across the whole workflow, and every function returns a plotnine
object you can extend with `+`.

### Accessibility

- Okabe-Ito palette and the depictr theme and scales as the default look.
- A Machado-2009 colour-vision-deficiency simulator (`simulate_cvd`) and a
  CIE-Lab palette safety check (`palette_safety`), so the default palette is
  validated rather than asserted.

### Plotting functions, by family

The functions fall into nine families.

- Exploratory analysis has `explore_distribution`, `explore_categorical`,
  `explore_bivariate`, `scatter_trend`, `correlation_heatmap`,
  `missingness_map`, `ecdf_plot`, `ridgeline_plot`, `dumbbell_plot`,
  `outlier_plot`, `group_comparison_plot`, `explore_pairs` and
  `raincloud_plot`.
- Estimation and tables are served by `estimation_plot` (single-panel Cumming
  or two-panel Gardner-Altman) and `summary_table`.
- Model estimates are drawn by `coefficient_plot`, `tidy_estimates` (a fitted
  model or a tidy frame), `effects_plot`, `interaction_plot`, `compare_models`,
  `random_effects_plot`, `posterior_plot`, `frequentist_bayesian_plot` and
  `power_curve_plot`.
- Diagnostics are covered by `qq_plot`, `influence_plot`, `vif_plot`,
  `binned_residual_plot`, `residual_diagnostics_plot` and `model_report`.
- Classification is covered by `roc_curve_plot`, `pr_curve_plot`,
  `confusion_matrix_plot`, `calibration_plot`, `gain_plot`, `lift_plot` and
  `threshold_plot`.
- Multivariate analysis has `pca_plot`, `scree_plot`, `cluster_plot`,
  `dendrogram_plot` and `silhouette_plot`.
- Survival has `survival_plot`, with an optional number-at-risk table.
- Time series have `acf_plot`, `decompose_plot`, `seasonal_plot` and
  `timeseries_plot`.
- Composition has `arrange_plots` and `save_plot`.

### Design

- Computation is delegated to the specialist packages (scikit-learn, statsmodels,
  lifelines, scipy) and re-skinned under the shared theme. Each is an optional
  dependency installed via an extra (`depictr[classification]`, `depictr[models]`,
  `depictr[survival]`).
- Reproducibly simulated datasets (`crop_yield`, `wellbeing_survey`,
  `lexical_decision`, `clinical_trial`) and a Streamlit gallery app with a live
  colourblind-vision toggle.

### Known limitations

- plotnine compositions have no figure-level title, so multi-panel grids carry
  their titles on each panel.
- A handful of functions from the R package are not ported:
  `optimizer_fixef_plot` (there is no clean statsmodels equivalent of
  `lme4::allFit`), `k_diagnostic`, `palette_preview`, `model_fit_table` and
  `ts_forecast`.

[Unreleased]: https://github.com/pablobernabeu/depictr-py/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/pablobernabeu/depictr-py/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/pablobernabeu/depictr-py/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/pablobernabeu/depictr-py/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/pablobernabeu/depictr-py/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/pablobernabeu/depictr-py/releases/tag/v0.1.0
