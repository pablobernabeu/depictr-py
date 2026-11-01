# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Upgrading from 0.3.0

#### Changed results

- `vif_plot()` gives one generalised VIF per term (Fox & Monette, 1992), as R's
  `car::vif()` does, where it gave one VIF per dummy column, which depended on
  the reference level.
- When a term spans several columns, `vif_plot()` draws every bar as
  GVIF^(1/(2 df)) against the square root of `threshold`, where it drew each
  column's VIF against a line at 5.
- `vif_plot()` computes VIFs from the coefficients' covariance, as `car::vif()`
  does, where it ignored the fit's weights. Values change for a Logit or Probit
  fit, a binomial or Poisson GLM and any other GLM that weights its
  observations unequally.
- `tidy_estimates()` keeps only the fixed effects of a MixedLM, where it kept
  the variance parameters, which `coefficient_plot()` and `compare_models()`
  drew as coefficients.
- `effects_plot()` and `interaction_plot()` take the range of the swept
  variable, the means of covariates and the modes of categorical ones over the
  rows the model used, where they took every row of the frame.
- `effects_plot()` and `interaction_plot()` predict offset and exposure models
  at the mean offset, which the y-axis names, where they used zero.
- `interaction_plot()` leaves out a group level found only in rows that the fit
  dropped, which it drew.
- `group_comparison_plot()` draws t intervals at the new `conf_level`, where it
  drew mean ± 1.96 SE, which covers 81% at n = 3, so the intervals of small
  groups widen.
- `estimation_plot()` takes the first level of `group` as the reference, where
  it took the group of the first row, and warns where 0.3.0 took another.
  Passing `reference` chooses it. The default `n_boot` is 5,000, and
  `n_boot=2000` restores the old count.
- `estimation_plot()` takes the other groups in a categorical's declared order,
  or else sorted, where it took them in order of first appearance, so its
  columns, the rows of `p.differences` and, with a `seed`, their bootstrap
  intervals can change.
- `calibration_plot()` cuts the scores into bins of nearly equal count and
  draws each bin as a point sized by its count, with a Wilson interval, where
  equal-width bins could hold only a few cases. `strategy="uniform"` restores
  the old bins, and `conf_level=None` leaves out the intervals.
- `gain_plot()` and `lift_plot()` target tied scores together (Fawcett, 2006),
  where the order of the rows decided how ties entered the curve, so curves on
  tied scores change.
- `threshold_plot()` evaluates the metrics at every distinct score, where it
  used 200 quantiles of the scores past 200 distinct values, and gives an F1 of
  0, which was missing, when no positive case is predicted positive.
- `binned_residual_plot()` takes `floor(sqrt(n))` bins cut as in R, where it
  took `round(sqrt(n))`, and draws the residuals of a grouped binomial as
  proportions, with each row's variance divided by its trials, where it drew
  counts, so its points and band change.
- `binned_residual_plot()` multiplies the band's variances by the scale of a
  GLM whose scale is not 1, as from `fit(scale="X2")`, where it took a scale of
  1, and leaves out observations with zero prior weight.
- `survival_plot()` reads an `event` coded 1/2 as `survival::Surv()` does, with
  1 for censored, where it read any non-zero `event` as an event, so every
  subject was an event. A missing event, which counted as one, is dropped with
  a warning.
- The number-at-risk table of `survival_plot()` keeps exact times, where it
  rounded them to one decimal and put counts under the wrong labels, and stops
  its breaks at the last follow-up time.
- The number-at-risk table of `survival_plot()`, and its `.at_risk` frame,
  count a subject followed up to exactly a break, which a floating-point error
  in the default breaks left out.
- `cluster_plot()` runs k-means on the full standardised data, as
  `silhouette_plot()` does, where it clustered the two-dimensional PCA
  projection, so cluster assignments change. Its axes give each component's
  share of the variance.
- `acf_plot(kind="pacf")` uses the unadjusted Yule-Walker estimator, which
  gives the values of R's `pacf()`, where the adjusted estimator gave other
  values and inflated lag 1 by n / (n - 1).
- `timeseries_plot()` keeps missing values in place, so the line and the
  rolling mean break at a gap, where it dropped them and joined the values
  either side.
- `timeseries_plot()` centres the moving average of an even `rolling` window,
  as a 2×12 average for `rolling=12` (Hyndman & Athanasopoulos, 2021), where
  the average was drawn half an observation late. Odd windows are unchanged.
- `seasonal_plot()` places each observation at its month, quarter or weekday
  when the index implies the period, where it counted from the first
  observation, and labels the ticks to match. It numbers the cycles from 1,
  where it numbered them from 0.
- `seasonal_plot()` reads a `period` within 0.00001 of a whole number as that
  number, where it truncated it, as 11.999999 to 11.
- The dataset loaders return the R package's own data, whatever `seed` is
  given, so their rows, columns and values change. Factor columns are
  categoricals in R's level order, which can change a model's reference level,
  as for the `treatment` of `crop_yield()`.
- The web app offers a non-numeric column as a group, category or model
  predictor only when it has 2 to 12 distinct values, so an identifier such as
  `participant` is no longer a pane's default group or predictor.
- The web app's Kaplan-Meier pane offers only columns coded 0/1 or 1/2 as the
  event and opens on one named `event` or `status`, where it offered every
  numeric column and took the first, which could be the time.
- `summary_table()` summarises a date or date-time column as one `Range` row
  and a duration as mean (SD) in a unit that the statistic names, where it gave
  one row per value or, with `vars=None`, skipped a column with many values.
- `palette_safety()` and `check_figure()` measure colour separability with
  CIEDE2000 (Sharma, Wu & Dalal, 2005), where the 1976 distance overstated the
  differences between saturated colours, so distances and some verdicts change.
  `metric="cie76"` restores the old measure.
- `simulate_cvd()` at partial severity interpolates between the published
  matrices of Machado et al. (2009), where it drew the full-severity matrix
  towards the identity, so its colours between severities 0 and 1 change. Full
  severity is unchanged.
- `check_figure()` measures each text against its own background, holds large
  text to 3:1 and measures a translucent plot or panel background as the colour
  it shows, where it measured all text against the plot background at 4.5:1.
- With a blank panel, as under `theme_depictr()`, `check_figure()` measures
  geometry against the plot background, where it measured it against the
  invisible white axes patch.
- `check_figure()` composites translucent colours over their background with
  the alpha each geom applies, where it measured them as opaque, and compares
  colours and fill colours each among themselves, where it pooled them, so
  verdicts can change in either direction.
- `check_figure()` counts a colour or fill split across layers, one value per
  layer, as a colour encoding, where it reported none, and counts shape and
  line type split that way towards `redundant_encoding`.
- `check_figure()` adds a tenth row, `annotation_text_size`, which holds the
  text inside the panel, such as annotations, `geom_text()` labels and the cell
  values of depictr's plots, to the size floor, so a figure that passed can
  fail.

#### Changed interface

- The `app` and `all` extras require Streamlit 1.49 or later, where they
  allowed 1.30.
- `vif_plot()` takes `threshold` second, so a title passed by position must be
  named.
- `correlation_heatmap()` takes `method` third, so a title passed by position
  must be named.
- A `conf_level` outside (0, 1), such as 95, is refused by every function that
  takes one, where a level of 0 or 1 gave the model estimates zero-width or
  unbounded intervals, 95 gave missing intervals or an error and
  `survival_plot()` ignored the level.
- Every function that takes a `conf_level` refuses NaN, where most gave missing
  intervals.
- `tidy_estimates()` returns the terms of a model fitted on a frame with
  numbered columns as strings, where it returned integers, on which the plots
  failed.
- `coefficient_plot()` and `compare_models()` refuse two terms of one model
  that take the same label, which they drew under their raw names.
  `labels="raw"` keeps those names.
- `effects_plot()` and `interaction_plot()` refuse a formula term computed from
  statistics of its rows, such as `I(x - x.mean())`, which shifted the line,
  and a `var`, `x` or `group` outside the formula, which drew a flat line.
- `vif_plot()` raises `ValueError` for aliased coefficients, where it drew
  infinite VIFs, and for fewer than two predictor columns.
- `binned_residual_plot()` raises `ValueError` for an `n_bins` below 2 or a
  model with fewer than four observations.
- `survival_plot()` refuses an `event` whose codes are not all within 0/1 or
  all within 1/2, such as 0/0.5, 0/2 or a mix of 0/1 and 1/2 across groups,
  where it truncated the codes to integers and counted every non-zero one as
  an event.
- The gains, lift and threshold curves raise `ValueError` for an outcome coded
  1/2, and the ROC, precision-recall, calibration, gains, lift and threshold
  curves for one coded -1/1, where they counted the outcomes equal to 1 as
  positive.
- `gain_plot()`, `lift_plot()` and `threshold_plot()` raise `ValueError` for a
  missing outcome or score, and `calibration_plot()` for a missing score.
- `calibration_plot()` raises `ValueError` when fewer than two quantile bins
  remain, as with `n_bins=1` or when nearly all cases share one score.
  `strategy="uniform"` restores the old equal-width bins for them.
- `explore_distribution()`, `ecdf_plot()` and `ridgeline_plot()` refuse a
  numeric `group` with more than 20 distinct values, which they drew, and
  `scatter_trend()`, `raincloud_plot()`, `group_comparison_plot()` and
  `explore_categorical()` refuse one when called, where they returned a plot
  that failed when drawn.
- `scatter_trend()`, `explore_distribution()`, `ecdf_plot()`,
  `raincloud_plot()`, `group_comparison_plot()`, `explore_categorical()` and
  `ridgeline_plot()` refuse a `group` missing for every row when called, where
  they drew it or returned a plot that failed when drawn.
- `power_curve_plot()` refuses rows of one curve that share a sample size,
  where it joined them into a zig-zag line.
- `explore_categorical()` refuses a numeric column with more than 20 distinct
  values, which it drew as one bar per value.
- `timeseries_plot()`, `acf_plot()`, `decompose_plot()` and `seasonal_plot()`
  refuse a series of more than one column, where they interleaved its rows into
  one series.
- `seasonal_plot()` and `decompose_plot()` raise `ValueError` for a minute or
  millisecond index without `period`, where they took a period of 12.
- `seasonal_plot()` refuses a period below 2.
- `correlation_heatmap()` refuses a lone column, which it drew as one tile, and
  a date or duration column, which it correlated as integers.
- `depictr_palette()` refuses an `n` below 1, where a negative `n` gave a
  shortened palette and 0 gave no colours, or seven from a ramp.
  `palette_safety()` refuses a single colour string, which it read as one
  colour per character.
- `coefficient_plot()` refuses an unknown `order` or a model with no terms
  left, where it drew a panel without warning.
- `explore_distribution()` refuses an unknown `kind` and a text, boolean, date,
  date-time or duration column, where it drew a panel without warning.
- `explore_categorical()` raises `KeyError` for an unknown `x`, and
  `explore_distribution()` and `ecdf_plot()` for an unknown `group`, when
  called, where they returned a plot that raised `PlotnineError` when drawn.
- Under its default letter tags, `arrange_plots()` refuses more than 26 panels,
  where it arranged any number. `tag_levels="1"` or `tag_levels=None` accepts
  them.

#### Changed output

- `coefficient_plot(order="ascending")` puts the largest estimate at the top
  and `"descending"` at the bottom, where they ran the other way, against the
  docstring. Swap the two to keep a 0.3.0 figure.
- `coefficient_plot()`, `compare_models()` and the coefficient panel of
  `model_report()` label their terms for a reader, so a factor term reads
  `treatment: enhanced` and `soil_ph` reads `soil ph`, where they printed the
  model's names. Except in `model_report()`, `labels="raw"` restores the names.
- `coefficient_plot()` and `compare_models()` drop a `b_Intercept` term with the
  intercept, where they drew it. `intercept=True` keeps it.
- `compare_models()` draws terms of different models that take the same label
  on one row, where each name had a row of its own. `labels="raw"` keeps a row
  per name.
- `arrange_plots()`, a single plot included, and `model_report()` tag their
  panels A, B, C and so on. `arrange_plots(tag_levels=None)` restores untagged
  panels.
- `depictr_accent()` returns the vermillion `#d55e00`, where it returned an
  orange that reached only 2.25:1 against white. The accent marks the flagged
  points of `outlier_plot()`, the missing cells of `missingness_map()`, the
  cumulative line of `scree_plot()` and the rolling mean of `timeseries_plot()`.
- The residual smoothers of `residual_diagnostics_plot()` and `model_report()`
  are grey, where they were orange.
- `scale_colour_depictr()`, `scale_fill_depictr()` and the plots that use them
  interpolate the palette past eight groups and warn that the colour-vision
  guarantee is lost, where plotnine drew the ninth group in the missing-value
  grey with its own warning. `n=8` in either scale keeps the old palette.
- `estimation_plot()` gives every group a column in the difference panel, in
  both layouts, the reference's first and empty, so each difference sits in its
  own group's column, where the difference panel had one column fewer than the
  groups and spread across the full width.
- `estimation_plot()` names its interval in a caption, and a `seed` also fixes
  the jitter of the points.
- `seasonal_plot()` draws its cycle lines from a ramp that starts at
  `#4a8fc0`, at 3.51:1 against white, where the earliest cycles were close to
  invisible.
- `seasonal_plot()` warns with a `UserWarning` when it rounds a fractional
  period down, which it did without one.
- `timeseries_plot()`, `decompose_plot()` and `seasonal_plot()` keep the index
  of a one-column frame, which they dropped, and so place its values by the
  index, as for a Series, where they placed them by observation number.
- `timeseries_plot()` and `seasonal_plot()` read an object column of numbers,
  such as floats with `None`, as numbers, where `timeseries_plot()` drew no
  line and `seasonal_plot()` spaced the values evenly as categories.
- `confusion_matrix_plot()` and `correlation_heatmap()` draw each label in
  black or white, whichever contrasts more with its tile, where some labels
  fell below the 4.5:1 that WCAG asks of text.
- `pca_plot()` scales the loading arrows by one factor on both axes, where it
  stretched each axis by its own and so misdrew the arrows' directions, and
  continues each label along its arrow.
- `pca_plot()` fixes the aspect ratio, so a figure of another shape gains blank
  space, and draws its loading arrows and labels in dark grey, where they took
  the accent, the palette's second colour.
- Without `group`, `power_curve_plot()` draws one curve per level of an
  `effect`, `param` or `term` column, with a warning, where it joined them into
  one line.
- `power_curve_plot()` draws grouped curves in a line type as well as a colour.
- `power_curve_plot()` orders the curves of `group` by first appearance, where
  it sorted text levels and kept a categorical's declared order, so their
  colours and legend order can change.
- `power_curve_plot()` draws a confidence band from interval columns such as
  `<power>_lo` and `<power>_hi` or `lower` and `upper`. The y-axis takes the
  power column's name unless it is `power`, `mean` or `p`, and `y_lab="Power"`
  restores the title.
- `survival_plot()` draws confidence limits at `conf_level`, on the log-log
  scale by default, and marks censored observations, where it drew neither.
  `conf_level=None` and `censor_marks=False` leave them out.
- `save_plot()` embeds the fonts of a PDF, EPS or PostScript file as TrueType
  (Type 42), where they were Type 3, and writes a TIFF as LZW-compressed RGB on
  white, where it was uncompressed RGBA. Passing `pil_kwargs` leaves the TIFF
  to matplotlib.
- `save_plot()` warns when an EPS or PostScript file has translucent layers,
  which the format draws opaque.
- `save_plot()` saves a composition at `width` and `height`, where it took its
  last panel's figure size, 6.4 by 4.8 inches by default.
  `width=6.4, height=4.8` restores the old default.
- `scatter_trend()`, `explore_distribution()`, `ecdf_plot()`,
  `raincloud_plot()`, `group_comparison_plot()`, `explore_categorical()` and
  `ridgeline_plot()` drop rows with a missing group, with a warning, and give
  no colour to a level without rows, so the colours of the levels after it can
  shift.
- `ridgeline_plot()` colours each ridge by its level, where the colours
  followed the median order. The ridges still stack by median.
- `explore_distribution()` and `ecdf_plot()` draw a numeric group code, such as
  blocks numbered 1 and 2, as groups, where they drew one ungrouped curve.
- `scatter_trend()`, `explore_distribution()`, `ecdf_plot()`,
  `raincloud_plot()`, `group_comparison_plot()`, `explore_categorical()` and
  `ridgeline_plot()` warn once past eight groups, when the plot is made, where
  plotnine warned each time the plot was drawn.
- Given a `group`, `scatter_trend()`, `explore_distribution()`, `ecdf_plot()`
  and `raincloud_plot()` leave out rows missing the plotted value without
  plotnine's warning that it removed them.
- `explore_bivariate()` draws a date, date-time or duration as box plots by
  level against a category and as a scatter plot with a trend against another
  date, date-time or duration, where it drew a tile of counts.
- `explore_bivariate()` takes a boolean column as a category, where it drew a
  scatter plot with a regression line through 0 and 1.
- `explore_distribution()` honours an explicit `alpha` for a grouped density,
  which it drew at 0.2 whatever `alpha` said.
- `explore_categorical()` draws a category that one group never takes as a bar
  of height zero, where it drew one bar the width of the whole category.
- `dumbbell_plot()` takes its two levels in a categorical's declared order, or
  else in sorted order, where it took them in the order they first appeared.
  The first level starts each segment, orders the categories and takes the
  first colour.
- `random_effects_plot()` sorts each panel by its own term's effects, where the
  later panels took the first term's order.
- `threshold_plot()` draws its metrics as steps that change at each score,
  where it joined adjacent scores with straight lines.
- `timeseries_plot()` draws no rolling overlay for a series one observation
  longer than an even window, where it drew a two-point line, since the centred
  average spans one observation more than the window.
- `summary_table()` names in a `UserWarning` the identifier columns that
  `vars=None` skips.
- The data of `binned_residual_plot()` gain the columns `n`, `se2` and
  `outside`, those of `coefficient_plot()` and `compare_models()` a `label`
  column and those of `vif_plot()` the columns `df` and `gvif`.
- `vif_plot()` draws its threshold line solid in grey, where it was dashed in
  the accent, and warns that a model without an intercept gives VIFs from
  uncentred columns.
- `group_comparison_plot()` shows only the mean of a group of one, with a
  warning, where it dropped the interval with a generic message.
- The `text_contrast` row of `check_figure()` reports the text furthest short
  of its own threshold, with that threshold and the text's printed size and
  weight, and a translucent colour is named with its alpha.
- The colour rows of `check_figure()` say when colour is a continuous scale,
  which they do not audit, where they said that nothing was distinguished by
  colour.
- The separability rows of `check_figure()` say whether their closest pair are
  colours or fill colours, and its `text_size` and `text_contrast` rows no
  longer say that a figure draws no text when all of its text is inside the
  panel.

### Added

- `simulate_figure()` shows a finished plot or composition, legend and
  colourbar included, as a reader with a colour-vision deficiency sees it, and
  returns an image as an array of floats.
- `depictr_palette(order="contrast")`, also in `scale_colour_depictr()` and
  `scale_fill_depictr()`, reorders the palette so that up to three groups pass
  every colour check of `check_figure()` and four keep 3:1 against white.
- `exponentiate=True` in `tidy_estimates()`, `coefficient_plot()` and
  `compare_models()` gives odds, rate or risk ratios, which the plots draw on a
  log axis with the reference line at 1 and an axis title naming the ratio. The
  two plots gain `reference_line`.
- `check_figure()` audits a composition, the package's own included, panel by
  panel, with a leading `panel` column, where it refused one.
- `check_figure(max_text_pt=)` checks a ceiling on text size in a
  `text_size_max` row, and `save_plot(check=True)` audits a figure at the
  width it is saved at, gives one warning that names each failed check and
  returns the filename alone.
- `power_curve_plot(solved=)` marks a solved sample size, such as the result
  of pilotr's `target_n()`, with a dashed line and its interval as a grey band.
- `power_curve_plot()` gains `target`, `lower`, `upper`, `ribbon` and `y_lab`,
  finds its sample-size and power columns by the R twin's names when `n` and
  `power` are not given and reads pilotr's power tables as they are, a list of
  dicts included.
- `monthly_sales()`, the R package's time-series dataset.
- `format_terms()` turns raw term names, those of patsy and formulaic included,
  into readable labels, with the R twin's arguments.
- `coefficient_plot()` and `compare_models()` gain `labels`, which takes a
  dict, a list or `"raw"`, and `interaction`, which sets how an interaction is
  written.
- `cluster_plot(clusters=)` draws a clustering computed elsewhere, and every
  plot from `cluster_plot()` carries its assignments, one per complete-case
  row, as `p.clusters`.
- `correlation_heatmap()` gains `show_values`, `digits` and `method`, which
  takes `"spearman"` or `"kendall"` as well as `"pearson"`.
- `survival_plot()` gains `conf_type`, which builds the limits on the
  `"log-log"`, `"log"` or `"plain"` scale, and `logrank`, which sets whether
  the log-rank test is printed.
- `binned_residual_plot(band="empirical")` draws the band of R's
  `arm::binnedplot()`.
- `arrange_plots()` gains `tag_levels`, which takes `"A"`, `"a"`, `"1"`, `"I"`,
  `"i"` or the tags themselves, as the R twin's does, and `tag_size`, which
  sets the size of the tags.
- `theme_depictr()` gains `base_family`, which sets the font of a single plot.
  Left as `None`, it takes `plotnine.options.base_family`, as in 0.3.0.

### Changed

- `ridgeline_plot()` draws much faster.
- `calibration_plot()` no longer needs scikit-learn.
- `coefficient_plot()` accepts `order="none"`, and `calibration_plot()` accepts
  `bins`, the R twin's names for model order and `n_bins`.
- `pca_plot()` names a misspelt `group` column and the argument in its
  `KeyError`.
- `correlation_heatmap()` refuses a text column in `cols` with a `TypeError`
  that names it, where it failed with a `ValueError` that did not, and
  `estimation_plot()` refuses a fractional `n_boot` with its own `ValueError`,
  where NumPy raised `TypeError`.
- `tidy_estimates()`, `coefficient_plot()`, `compare_models()`,
  `effects_plot()`, `interaction_plot()` and `estimation_plot()` refuse a
  `conf_level` that is not a number, such as `None`, with a `ValueError` that
  names the argument, where they raised a `TypeError` from the arithmetic.
- The package metadata declares the licence as the SPDX expression `MIT`.

### Deprecated

- `metric="cie76"` in `palette_safety()` and `check_figure()`, kept to
  reproduce colour distances computed before this release. The default,
  `"ciede2000"`, replaces it, and a later release will remove it.
- The `seed` argument of the dataset loaders, which no longer generate data.
  Passing a value raises a `FutureWarning` and has no effect.
- The log-rank subtitle that `survival_plot()` prints by default for grouped
  curves. The default, `logrank=None`, keeps it with a `FutureWarning`, and a
  later release will make `logrank=False` the default. Pass `logrank=True` to
  keep it.

### Fixed

- `tidy_estimates()`, and the plots built on it, read a model fitted from
  arrays, which raised `AttributeError`, and name its terms as statsmodels
  does.
- `residual_diagnostics_plot()` and `model_report()` diagnose GLM, Logit and
  Probit fits, which raised `AttributeError`, with randomised quantile
  residuals and, for a binomial fit, binned residuals. They gain `glm_panels`
  and `seed`. `binned_residual_plot()` takes Logit and Probit fits.
- `vif_plot()` accepts WLS and GLS fits, which it refused as not fitted.
- `estimation_plot()` accepts a numeric group in the two-panel layout and a
  whole number held in a float as `n_boot`, such as `1e4`, both of which raised
  `TypeError`.
- `scatter_trend()`, `raincloud_plot()`, `group_comparison_plot()` and
  `explore_categorical()` accept a numeric group code, which raised
  `TypeError`.
- `power_curve_plot()` accepts a numeric `group`, which raised `TypeError`.
- The ROC, precision-recall, calibration, gains, lift and threshold curves read
  a two-category categorical outcome, its last category positive, which they
  refused.
- `effects_plot()` and `interaction_plot()` read only the columns the formula
  names, so an unrelated all-missing text column no longer raises
  `IndexError`.
- `save_plot()` creates a missing output directory.
- `simulate_cvd()` reads a single colour string as one colour and returns an
  empty list for an empty list, where both failed.
- `depictr_palette()` truncates a fractional `n`, which failed.
- `decompose_plot()` and `seasonal_plot()` run on pandas 3.1 without a
  `Pandas4Warning`, and work there when `future.infer_freq_returns_offset` is
  set, where they failed.
- `decompose_plot()` and `seasonal_plot()` take the period of a one-column
  frame from its index, as for a Series, where a call without `period` raised
  `ValueError`.
- `explore_bivariate()` draws a date, date-time or duration against a numeric
  column as a scatter plot with a trend, which raised `TypeError`.
- The rolling overlay of `timeseries_plot()` no longer warns that it removed
  rows from complete data.
- Install commands in the documentation and in `ImportError` messages quote
  their extras, as in `pip install "depictr[all]"`, so they paste into zsh.
- The web app reads uploaded CSV files saved in cp1252 or latin-1 and says
  what a pane needs when no column suits it. It offers as the dumbbell's group
  only two-level columns other than its category and value.
- The web app's 'exact call' reproduces the figure beside it, with the import
  and the model fit before a model's call.

### Documentation

- The README, guides and docstrings no longer call the built-in palette the
  Okabe-Ito set. The docstring of `depictr_palette()` says that it keeps six of
  that set's eight colours, with the brand blue and a grey in place of its blue
  and black.
- The README's Status section lists the differences between the twins that
  change what a figure shows or how it is computed, where it said that they
  share nearly all of an API.
- Docstrings that described behaviour the code does not have now match it.
  `scatter_trend()` draws no band with the `"lowess"` and `"rlm"` smoothers.
  `effects_plot()` and `interaction_plot()` hold a categorical predictor at its
  most common level, which need not be the reference. `explore_pairs()` always
  returns a composition.
- The install sections say that the PCA, scree, cluster and silhouette plots
  need the `classification` extra. The README and the app guide install the
  `all` extra to run the app, where they installed the `app` extra. The
  getting-started guide says that only `coefficient_plot()`, `compare_models()`
  and `frequentist_bayesian_plot()` accept a table of estimates, and the README
  and the getting-started guide say that every plotting function returns a
  plotnine object, where they said that every function did.
- The docstring of `check_figure()` gains a section on auditing a figure at the
  width it is printed at, with the text floors of four publishers, where it
  called 6 pt a common publisher floor, and says that four colours of the
  default palette fall below 3:1 against white. The docstring of `save_plot()`
  describes the formats that three publishers accept.
- The accessibility guide shows a figure drawn by `simulate_figure()` and
  covers four publishers' text floors, Nature's ceiling, the contrast-first
  palette order and how the audit treats a continuous colour scale.

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
