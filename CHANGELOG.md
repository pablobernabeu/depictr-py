# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `monthly_sales()`, the time-series dataset that was R-only.
- `cluster_plot(clusters=)` draws a clustering computed elsewhere, as the R
  twin's `clusters` argument does, and every cluster plot carries its
  assignments as `p.clusters`, one per complete-case row of `data`.
- `power_curve_plot()` gains `target`, `lower`, `upper`, `ribbon` and `y_lab`
  after `title`, as in the R twin. The target line was fixed at 0.8, and no
  confidence band was drawn. It also takes a list of dicts, one per row, which
  is how pilotr returns a power curve in Python, and recognises `n_subject` and
  `<power>_lo`/`<power>_hi` columns, so pilotr's output plots as it is, with
  its band.

### Changed

- `CITATION.cff` names PyPI as the package's repository artifact, as the R
  twin's names CRAN, and says that its DOI is the Zenodo concept DOI of this
  repository, which holds the Python package alone, since PyPI assigns no DOI.
  Its release date, and the date of the 0.3.0 section below, are now
  2026-09-24, the day PyPI published 0.3.0 and the GitHub release appeared. Both
  gave 2026-08-21, the day the release was prepared.
- The home page gives the Zenodo concept DOI as a full link,
  <https://doi.org/10.5281/zenodo.21266311>, where it showed the Zenodo badge,
  and the About page links every DOI the same way. The BibTeX entry's `url`
  gives the PyPI page, so a style that prints both fields no longer shows the
  DOI twice. The README's DOI badge now ends the badge row and names the DOI in
  its text alternative, and the README's citation section gives the DOI.
- The bundled datasets are now the R package's own data, shipped as CSV files,
  so an example gives the same numbers in both twins and every column an R
  example uses exists here. Their shapes, columns and values change. `crop_yield` gains its
  field identifier and loses two impossible negative yields. `wellbeing_survey`
  gains respondent ids, exercise days and an ordered education factor.
  `lexical_decision` becomes the crossed design of 24 participants and 40 items,
  so the mixed-model example is no longer singular. The arms of
  `clinical_trial` are placebo and treatment. Factor columns are categorical in
  R's level order. The files are written from the R package by
  `dev/export_r_datasets.R`, and the tests pin their checksums and a set of
  values computed in R.
- The `random_effects_plot()` example and gallery entry fit participant
  intercepts to `lexical_decision`, whose participant variance statsmodels and
  lme4 estimate alike (about 6050). The old example grouped the wellbeing survey
  by region, whose variance was estimated at zero.
- The web app offers a column as a grouping variable only when it has 2 to 12
  distinct values, so an identifier such as `participant` or `patient` is never
  the default group, and it leaves identifiers out of the model predictors.
- `cluster_plot()` clustered the two-dimensional PCA projection, so its
  clusters could differ from `silhouette_plot()`'s and the R twin's for the
  same `k`. On the bundled data with their default columns, the two partitions
  agreed with an adjusted Rand index of 0.39 for `crop_yield` and 0.63 for
  `wellbeing_survey` at `k = 4`. It now clusters the full standardised data with
  `silhouette_plot()`'s settings and projects only for display, so cluster
  assignments change. Each centroid is drawn at the projection of its k-means
  centre, and the axes give each component's share of the variance, as
  `pca_plot()` and the R twin do.
- `power_curve_plot()` finds its columns as the R twin does when `n` and
  `power` are not given, so `n`, `nlevels`, `x`, `sample_size` or `n_subject`
  serves as the sample size and `power`, `mean` or `p` as the power. A table
  with several curves and no `group` was joined into a single zig-zag line. An
  `effect`, `param` or `term` column with more than one level now separates the
  curves, with a warning naming the columns used, and rows that share a sample
  size with no such column raise `ValueError`. Each curve takes a line type as
  well as a colour. The y-axis title is "Power" for the usual power columns and
  the column's name otherwise.
- `binned_residual_plot()` bins as the R twin does, so the twins draw the same
  points for a model. It took `round(sqrt(n))` bins where R takes
  `floor(sqrt(n))`, kept between 2 and `n // 2`, and it cut the ranks of the
  fitted values elsewhere, so the first bin of the clinical-trial example held
  18 rows to R's 17. Tied fitted values were left in whatever order numpy's
  sort gave them and now follow row order, as in R. The plot gains `band=`. Its
  default, `"model"`, keeps the band the plot drew, which the R twin now draws
  too, and `"empirical"` draws the band of R's `arm::binnedplot()`, from the
  spread of the residuals in each bin. The plot's data gain the R twin's `n`,
  `se2` and `outside` columns. An `n_bins` below 2 raises `ValueError`, as in
  R, and so does a model with fewer than four observations.

### Deprecated

- The `seed` argument of the dataset loaders, which no longer generate data.
  Passing a value raises a `FutureWarning` and has no effect.

### Fixed

- `coefficient_plot(order="ascending")` and `order="descending"` drew the terms
  the opposite way round from the docstring ("order of the terms up the axis")
  and from the R twin. They now order up the axis, so `"ascending"` puts the
  largest estimate at the top, as in R. Terms with tied estimates keep their
  model order, as R's `order()` keeps them.
- `tidy_estimates()` kept every entry of a MixedLM's `params`, so
  `coefficient_plot()` and `compare_models()` drew the random-effect variances
  and covariances beside the fixed effects as if they were coefficients. The
  `Group Var` entry is not even a variance: statsmodels divides it by the
  residual scale. Only the fixed effects are kept now, as in the R twin.
- A model fitted from arrays, without a formula, made `tidy_estimates()` raise
  `AttributeError`, and with it `coefficient_plot()`, `compare_models()` and
  `frequentist_bayesian_plot()`. Its terms are now named from the model's
  `exog_names`, as statsmodels' own summary names them.
- A model fitted on a data frame with numbered columns, as `pd.DataFrame()`
  makes from an array, gave `tidy_estimates()` integer terms, on which
  `coefficient_plot()` and `compare_models()` failed. Terms read from a model
  are now always strings, as in the R twin.
- `survival_plot()` read any non-zero `event` as an event. With the
  `survival::Surv()` 1/2 coding (1 = censored), which the R twin accepts, every
  subject became an event, and the curves and the log-rank test were wrong
  without a warning. A 0/0.5 coding was cast to integers and so read as all
  censored, and a missing event became a large negative integer, which lifelines
  counted as an event. The 0/1 and 1/2 codings are now accepted, read over all
  observations together as in R. Any other coding raises, and a missing event is
  dropped with a warning.
- The number-at-risk table rounded its times to one decimal, so short follow-up
  collapsed onto a single tick and quarter-unit breaks sat under the wrong
  labels. Times are kept exact and only the labels are formatted. The table also
  gained a column of zeros at a break past the longest follow-up, outside the
  time axis. Its breaks now stop at the last follow-up time, as the R twin's do.
- The web app no longer offers the time column as the event indicator. Its
  Kaplan-Meier pane offers the columns coded 0/1 or 1/2, opens on one named
  `event` or `status`, and says so when a dataset has none.
- `save_plot()` wrote files that journals reject. A PDF or EPS carried Type 3
  fonts, which Nature refuses. An EPS drew its translucent layers opaque, which
  matplotlib reported only in its log. A TIFF was uncompressed RGBA, 10.8 MiB at
  the default 7 by 4.5 inches, where PLOS requires LZW compression and no alpha
  channel. Fonts in a PDF or EPS are now embedded as TrueType (Type 42), for that
  save only, and an EPS with translucent layers warns and says how many there
  are. A TIFF is saved as LZW-compressed RGB at the requested resolution, with
  any transparent background composited onto white, unless `pil_kwargs` is
  passed. A missing output directory is now created, as in the R twin, where it
  raised `FileNotFoundError`.
- `effects_plot()` and `interaction_plot()` held covariates at means taken over
  every row of the source frame, including rows the fit dropped, and treated
  every column as a predictor, so an unrelated all-missing column raised
  `IndexError`. They now use the estimation rows and the columns the formula
  names. A column outside the formula is no longer accepted as `var`, `x` or
  `group`, where sweeping it drew a flat line, and a group level found only in
  dropped rows is no longer drawn.
- Offset and exposure models were predicted at an offset of zero. They are now
  predicted at the mean offset over the estimation rows, which for an exposure
  is its geometric mean, as in the R twin, and the y-axis label says so.
- A formula term computed from statistics of its rows, such as
  `I(x - x.mean())`, shifted the predicted line, since `get_prediction()`
  computed it afresh over the grid. Such a term is now refused with
  `ValueError`, as in the R twin, whether statsmodels builds the formula with
  patsy or with formulaic, the engine statsmodels 0.15 added. patsy's
  `center()` and `standardize()` keep the values of the fit and are unaffected.
- `estimation_plot()` drew the differences out of line with the groups they
  belong to. Its difference panel had one column fewer than there are groups
  and spread its columns across the full width, so with two groups the single
  difference sat midway between them, beneath neither. The default reference
  was the group of the first row, so shuffling the rows could change the
  reference and with it the sign of every difference, and the interval carried
  no label. In both layouts, every group now keeps its column, with an empty
  one for the reference, which comes first as in the R twin. The default
  reference is the first level of the group, in a categorical's declared order
  or else in sorted order, with numbers sorted by value. A `FutureWarning` names
  both groups wherever the 0.3.0 rule would have chosen another. A numeric
  group made the two-panel layout raise `TypeError` and is now drawn as in R. A
  caption names the interval ("Bootstrap 95% CI (5,000 resamples)") as in R,
  and the default `n_boot` rises from 2,000 to the R twin's 5,000. A `seed` now
  fixes the jitter of the raw data as well, so a seeded two-panel figure comes
  out the same every time. The layout is now called a Cumming estimation plot.
  Gardner-Altman names a layout that sets the difference beside the data, which
  this function does not draw.
- `check_figure()` measured all text against the plot background, so text on a
  coloured facet strip or legend box was judged against a background it is not
  drawn on: white strip text on a dark strip failed and unreadable grey strip text
  passed. Each text is now measured against its own background, with a
  translucent fill taken as the colour it shows over what lies beneath it. Large
  text (18 pt, or 14 pt bold, as printed) is held to the 3:1 that WCAG asks of it,
  where it used to be held to 4.5:1. The `text_contrast` row reports the text that
  falls furthest short of its own threshold and gives that threshold, and its
  detail names the text's printed size and weight. A translucent plot or panel
  background is also measured as the colour it shows. Its alpha used to be
  dropped, so `geometry_contrast` moves for such a figure. The
  relative-luminance code now cites the current WCAG 2.2 definition and its
  0.04045 breakpoint, which changes no ratio.
- With a blank panel, as under `theme_depictr()`, `check_figure()` measured
  geometry against the invisible white axes patch in place of the plot
  background, so a figure on a dark background passed in Python and failed in R.
  Both twins now report the same numbers for it.
- `check_figure()` measured translucent marks and text as if they were opaque,
  so a figure could pass contrast, colour-vision and greyscale checks that its
  drawn colours fail. The accessibility page's 'mended' example was one: its
  points, drawn at alpha 0.7, passed the geometry check at 3.87:1, and the
  vermillion a reader sees reaches 2.56:1. Colours are now composited over
  their background with the alpha each geom applies, outlines drawn opaque are
  credited, and the colours of each aesthetic are compared among themselves. A
  translucent colour or text is named in the detail with its alpha and the
  colour it is drawn as, and the separability rows say whether their closest
  pair are colours or fill colours. Verdicts on translucent figures can change
  in either direction. The page's example now draws its points opaque.
- `check_figure()` now treats a colour or fill mapped in several layers, one
  value per layer, as the encoding it is. Such figures were reported as
  encoding nothing by colour. With red and green drawn one group per layer,
  every colour row read "not applicable", whereas the same figure drawn in one
  layer fails the deuteranopia check at 0.37. Shape and line type split across
  layers count towards `redundant_encoding` in the same way. A colour set as a
  constant, as in `geom_point(colour="red")`, is still left out, as the
  docstring now says.
- `pca_plot()` stretched the loading arrows by a different factor on each axis,
  so the drawn directions were wrong, by up to 22 degrees on the
  `wellbeing_survey()` of 0.3.0. Both axes now share one factor, as in the R
  twin and in base R's `biplot()`. The aspect ratio is fixed as well, so each
  arrow points along its loading on a figure of any shape. The panel's height
  and width follow the ranges of its axes, so a figure of another shape gains
  blank space around the panel. Each label continues its arrow outward, as in R.
  Arrows and labels are drawn in a neutral dark grey (`#333333`) in place of
  the accent, which is also the second group colour and so tied the variables
  to the second group. The `KeyError` for a misspelt `group` now names the
  column and the argument, in the R twin's words.
- `residual_diagnostics_plot()` and `model_report()` raised `AttributeError`
  for every GLM, Logit and Probit fit, although documented for GLMs. Such a fit
  now gets a Q-Q panel of randomised quantile residuals, which are standard
  normal under a correct model whatever the family. A binomial fit gets
  `binned_residual_plot()` as its residual panel, as in the R twin, and other
  families plot their deviance residuals against the fitted values. Both
  functions gain `glm_panels=True`, where `False` draws the classic panels, and
  `seed=` for a reproducible randomisation. `binned_residual_plot()` now takes
  a Logit or Probit fit. Such a result's `fittedvalues` holds the linear
  predictor, so the panels read its fitted probabilities from `predict()`.
- `binned_residual_plot()` misdrew a grouped binomial model. With a two-column
  (successes, failures) response, statsmodels' response residuals count
  successes, so the points were drawn in counts against fitted probabilities,
  and the band took every row as a single trial, wherever the trials came from.
  The residuals are now proportions, as in the R twin, and each row's variance
  is divided by its number of trials, whether these come from the two columns
  or from `var_weights`. A GLM fitted with an estimated scale, as by
  `fit(scale="X2")`, has the variances multiplied by it, as the R twin
  multiplies them by a quasibinomial fit's dispersion. statsmodels' Pearson
  estimate leaves out the trials of a two-column response, so for such a fit
  the scale is recomputed over the trials, which gives R's dispersion.
- `scale_colour_depictr()` and `scale_fill_depictr()` drew a ninth group in the
  missing-value grey, indistinguishable from missing data, with no warning of
  depictr's own, and so did every plot that adds them, such as `ecdf_plot()`
  and `raincloud_plot()`. Without `n`, the scales now take as many colours as
  the plot has groups, so past eight groups they interpolate the palette and
  warn, as the R twin and the README describe. Figures with eight groups or
  fewer are drawn exactly as before.
- `group_comparison_plot()` drew mean ± 1.96 SE, which covers 81% at n = 3 and
  88% at n = 5 for a nominal 95%. It now uses the t distribution, as the R twin
  and `estimation_plot()` do, and gains `conf_level`. The intervals of small
  groups widen accordingly. A group with a single observation shows its mean
  only, with a warning in the R twin's words, where plotnine used to drop its
  interval with a generic message about missing values.
- `confusion_matrix_plot()` and `correlation_heatmap()` drew every cell label
  in the near-black `#1a1a1a`, although both colour ramps end in a dark blue.
  The labels were therefore least legible on the darkest cells. The largest
  count in the docstring's confusion matrix sat at 1.3:1 and the r = 1 diagonal
  of a correlation heatmap at 2.4:1, where WCAG asks 4.5:1 of text. Each label
  is now black or white, whichever has the higher contrast against its tile as
  the fill scale paints it, which is at least 4.58:1 on any colour. The `n/a` of
  an undefined correlation is chosen against the scale's grey in the same way.

## [0.3.0] - 2026-09-24

### Added

- `check_figure()`, an accessibility and honesty audit of the figure you are about
  to submit. Until now the package could vouch for its palette and say nothing
  about a finished plot, which is the thing a reader sees. Give it anything a
  depictr function returns, including a plot extended afterwards with `+`, and it
  introspects the build and returns a tidy DataFrame. The rows cover the
  separability of the encoding colours under each dichromacy and in greyscale, the
  smallest text size against a stated physical output width, the WCAG contrast of
  the text and of the geometry against their backgrounds, and whether any
  distinction is carried by colour alone. Every row carries the value it measured
  beside the threshold it was measured against, so a verdict can be argued with.
  The R twin gained the same function, with the same check names, thresholds,
  verdicts and measured numbers.
- CI now exercises the installed package as well as the checkout. The matrix gains
  Python 3.14. A new job installs the built wheel into a bare environment outside
  the repository and imports it there, so packaged data and distribution metadata
  are tested in the state a user meets them, and not masked by the source tree an
  editable install sits beside. A second new job installs the declared minimum
  dependency versions. A weekly schedule runs the suite when nobody has pushed, so
  upstream drift in the plotting stack shows up as a dated red badge before anyone
  is surprised by it.
- Linting, matching the rest of the family. This was the only Python package in the
  family running no linter, which is why it had quietly accumulated seven findings.
  The rule set is stated explicitly in `pyproject.toml` (`E, F, W, I, UP, B`, as in
  the scopusflow and theoryforge twins) so that it does not shift underneath the
  package: an inherited default is what turned a sibling's green CI red without a
  line of that package changing. The job lints the whole repository, where naming
  `src` and `tests` had left `app/streamlit_app.py` and `docs/_exec.py` covered by
  nothing, and the app was carrying an over-long line that no gate could see. Naming
  directories leaves a new top-level script unlinted until somebody remembers to
  extend the list.

### Changed

- `simulate_cvd()` names the three deficiencies in its refusal message, in place
  of the Python tuple it printed before, so the wording matches the R twin byte
  for byte.
- The sRGB-to-XYZ matrix behind the CIE Lab conversion is now given to seven decimal
  places instead of four. The rounded form shifted a Delta-E by a few thousandths,
  enough to round a reported distance differently from the R twin on the same
  colours. The distances `palette_safety()` reports for the default palette are
  unchanged at two decimal places.
- The accessibility claim has been narrowed to what is true. The default
  eight-colour palette clears every colour-vision check and fails the new greyscale
  check: its orange (`#e69f00`) and sky blue (`#56b4e9`) differ by 0.79 in CIE
  lightness, so a black-and-white printer renders them as the same grey. The
  Okabe-Ito guarantee is about hue confusion and was never a claim about greyscale.
  The threshold stays where it is, and the accessibility page now states the
  limitation where the claim is made, so the package's own defaults are held to the
  same standard as anybody else's.
- Two `zip()` calls now pass `strict=True`: the dendrogram builder pairs scipy's
  `icoord` with `dcoord`, and the forest plot pairs two columns of one frame. In
  both, unequal lengths are impossible by construction, so raising is the right
  response to the impossible happening, where a silent truncation would have left a
  plot missing part of its data. Import blocks were sorted and one long line
  wrapped, with no change in behaviour.

### Fixed

- `depictr_palette()` interpolated past its accessibility guarantee without saying
  so. Beyond the eight Okabe-Ito base colours the palette is a ramp, and the
  colour-vision-deficiency guarantee that is this package's reason for existing
  stops holding, so the interpolated palette fails the package's own
  `palette_safety()` check. It now warns at the point of interpolation. The R twin
  carried the same silence and was fixed with it.
- `survival_plot()` drew a phantom arm for a group that does not exist. A missing
  value in `group` became a level matching no observation. Missing groups are now
  dropped with a count, and an all-missing group is an error.
- `interaction_plot()` drew a missing grouping value as a literal `nan` line and
  legend entry, the same "a missing value stringified into a legitimate-looking
  value" defect fixed across the family this round.
- `summary_table()` counted missing-group records in `Overall` and in no group
  column, so the per-group sizes silently fell short of the headline N. They now
  get a `Missing` column. Its group columns are also ordered by level, so the table
  no longer depends on the order the rows arrived in.
- `seasonal_plot()` ordered its cycles lexicographically, sorting cycle 10 between 1
  and 2, and coloured them categorically where the sequential ramp belongs. An
  annual index also inferred a seasonal period of 1, so an explicit `period` is now
  required.
- `estimation_plot(two_panel=True)` omits the interval for a single-observation
  group. The zero-width interval it drew there before implied a precision the data
  does not have.
- `palette_safety()` rejects a palette of fewer than two colours, having previously
  called one safe at an infinite distance.
- `gain_plot()` and `lift_plot()` refuse a single-class outcome. Both previously
  substituted a denominator of 1 and drew a flat curve.
- `survival_plot()` raises a clear error for a non-finite follow-up time, where a
  bare `StopIteration` escaped before.
- `correlation_heatmap()` drops zero-variance columns with a message, and an
  undefined correlation is labelled `n/a` where the string `nan` was being printed
  in the cell.
- `acf_plot()`, `decompose_plot()` and `seasonal_plot()` silently dropped internal
  missing values, closing up the gap: the ACF correlated values across it, the
  decomposition misaligned, and every post-gap observation landed on the wrong
  within-period position. An internal missing value is now an error, in the wording
  of `survival_plot()`'s refusal of a non-finite follow-up time. Missing values at
  either end are still trimmed, since trimming only shortens the series.
- `acf_plot()` turns an all-missing or empty series away by name, as
  `seasonal_plot()` already did. Its default lag count took the base-10 logarithm of
  the length, so a series with nothing in it came back as an `OverflowError` from
  the surrounding `int()`, naming neither the argument nor the problem.
- `roc_curve_plot()`, `pr_curve_plot()` and `threshold_plot()` accepted a
  single-class outcome, annotating an AUC of nan or an average precision of 0.000 as
  if they measured something (scikit-learn computes both under a warning and does
  not refuse). They now raise the same refusal `gain_plot()` and `lift_plot()`
  already carry, with the R twin's wording per curve.
- `dendrogram_plot()` needed scikit-learn despite its documented scipy-only
  contract, because standardisation went through sklearn's `StandardScaler`. It is
  now a numpy z-score with identical semantics (population standard deviation, and a
  scale of 1 substituted for a zero-variance column), so the dendrogram runs on the
  core install and the sklearn-backed plots see the same input as before.
- The installation docs called statsmodels optional, when plotnine 0.15 requires it
  and every core install therefore already ships it. The README and the docs now
  name scikit-learn and lifelines as the optional back-ends, with `depictr[models]`
  kept to pin the tested statsmodels floor.
- The contributing guide told contributors to run `ruff check src tests`, the
  command CI stopped using when linting moved to the whole repository. It now
  names `ruff check .` and says what the wider scope buys.
- The README's list of functions not yet ported left out `format_terms` and
  `depictr_options`, and said nothing about the R-only `monthly_sales` dataset.
  All three are now named, and `legend_inside` joins the table of exported
  functions, which had omitted it.
- `summary_table()`'s documentation promised to sort the levels of an unordered
  categorical, when any categorical keeps its declared category order, which is the
  intended behaviour. The docs now say so.
- `posterior_plot()` warns when a `labels` key matches no parameter. An unmatched
  key was silently ignored, and the figure came back unrelabelled.
- Three declared dependency floors described a stack no install could produce.
  `pandas>=2.0`, `matplotlib>=3.6` and `scipy>=1.7` all sat below what
  `plotnine>=0.15` already requires (2.2, 3.8 and 1.8 respectively), and the
  `models` extra named `statsmodels>=0.14` where plotnine forces 0.14.5. A
  resolver always took the higher bound, so no install was ever affected, but a
  floor is a claim about what has been tested and this one was false. The floors
  now state what plotnine forces, and a new CI job installs them.

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
  panel, over a light background, so the figure needs no right-hand margin.
- Added a 'Getting started' guide that walks through a short analysis end to end.

### Changed

- Rebuilt the number-at-risk table beneath `survival_plot(risk_table=True)`.
  The curves now use the full panel width (no left-hand gutter), the group names
  label the rows on the y-axis, and the counts are coloured to match the curves,
  so the table reads as a strip under the plot. It used to be text floating in
  loosely spaced negative space.
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
  `geom_histogram(fill=None)` made them fully transparent, where the intention
  was to defer to the group colour mapping.
- Axis and legend titles leaked raw column names. Several plots that
  meant to leave a title blank (`labs(x=None, ...)`) instead showed the mapped
  column's literal name (`x`, `value`, `variable`, `term`, `metric`, ...) because
  this plotnine version reads `None` as unset, and only `""` as blank. Corrected 14
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
  CIE-Lab palette safety check (`palette_safety`), which validate the default
  palette.

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

[Unreleased]: https://github.com/pablobernabeu/depictr-py/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/pablobernabeu/depictr-py/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/pablobernabeu/depictr-py/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/pablobernabeu/depictr-py/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/pablobernabeu/depictr-py/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/pablobernabeu/depictr-py/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/pablobernabeu/depictr-py/releases/tag/v0.1.0
