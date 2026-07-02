# Changelog

## 0.1.1

- **Inside legends.** Added an opt-in `legend_inside=False` parameter to
  `explore_distribution`, `ecdf_plot`, `dumbbell_plot`, `missingness_map` and
  `survival_plot`, plus the public `legend_inside()` theme helper, which places
  the legend inside the panel (over a light background) rather than in a
  right-hand margin.
- **Fix: grouped histograms were invisible.** `explore_distribution(kind="both"`
  or `"histogram")` with a `group` drew no bars at all: `geom_histogram(fill=None)`
  made them fully transparent instead of deferring to the group colour mapping.
- **Fix: axis and legend titles leaked raw column names.** Several plots that
  meant to leave a title blank (`labs(x=None, ...)`) instead showed the mapped
  column's literal name (`x`, `value`, `variable`, `term`, `metric`, ...) because
  this plotnine version treats `None` as unset rather than blank. Corrected 14
  call sites across `diagnostics`, `eda`, `estimation`, `mixed`, `models`,
  `multivariate`, `posterior`, `predictions`, `distributions_extra`,
  `timeseries` and `classification`.
- **Fix: a duplicate colour scale on `survival_plot(risk_table=True)`.** The
  grouped risk-table path added the colour scale twice, which plotnine warned
  about and silently replaced with an identical one; it is now added once.
- Corrected the Cook (1977) reference title and added DOIs to Cook (1977),
  Hedges (1981) and Allen et al. (2021).
- **Survival risk table.** Rebuilt the number-at-risk table beneath
  `survival_plot(risk_table=True)`: the curves now use the full panel width (no
  left-hand gutter), the group names label the rows on the y-axis, and the
  counts are coloured to match the curves -- a tidy strip under the curves
  rather than text floating in loosely spaced negative space.
- **Survival statistics.** The log-rank *p*-value follows APA style -- no
  leading zero, reported as *p* < .001 below that threshold -- with an
  italicised *p*. The colour legend and the risk-table rows now list the groups
  in the same order (a user-set categorical order, otherwise first appearance).
- Added a "Getting started" guide that walks through a short analysis end to end.
- The README and the PyPI project page now open with a gallery (a grouped
  density and Kaplan-Meier curves), and the documentation landing page gains the
  same hero plot and a PyPI install link.
- README image assets are kept out of the source distribution.

## 0.1.0

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

- **Exploratory:** `explore_distribution`, `explore_categorical`,
  `explore_bivariate`, `scatter_trend`, `correlation_heatmap`, `missingness_map`,
  `ecdf_plot`, `ridgeline_plot`, `dumbbell_plot`, `outlier_plot`,
  `group_comparison_plot`, `explore_pairs`, `raincloud_plot`.
- **Estimation and tables:** `estimation_plot` (single-panel Cumming or
  two-panel Gardner-Altman), `summary_table`.
- **Model estimates:** `coefficient_plot`, `tidy_estimates` (a fitted model or a
  tidy frame), `effects_plot`, `interaction_plot`, `compare_models`,
  `random_effects_plot`, `posterior_plot`, `frequentist_bayesian_plot`,
  `power_curve_plot`.
- **Diagnostics:** `qq_plot`, `influence_plot`, `vif_plot`,
  `binned_residual_plot`, `residual_diagnostics_plot`, `model_report`.
- **Classification:** `roc_curve_plot`, `pr_curve_plot`, `confusion_matrix_plot`,
  `calibration_plot`, `gain_plot`, `lift_plot`, `threshold_plot`.
- **Multivariate:** `pca_plot`, `scree_plot`, `cluster_plot`, `dendrogram_plot`,
  `silhouette_plot`.
- **Survival:** `survival_plot`, with an optional number-at-risk table.
- **Time series:** `acf_plot`, `decompose_plot`, `seasonal_plot`,
  `timeseries_plot`.
- **Composition:** `arrange_plots`, `save_plot`.

### Design

- Computation is delegated to the specialist packages (scikit-learn, statsmodels,
  lifelines, scipy) and re-skinned under the shared theme; each is an optional
  dependency installed via an extra (`depictr[classification]`, `depictr[models]`,
  `depictr[survival]`).
- Reproducibly simulated datasets (`crop_yield`, `wellbeing_survey`,
  `lexical_decision`, `clinical_trial`) and a Streamlit gallery app with a live
  colourblind-vision toggle.

### Known limitations

- plotnine compositions have no figure-level title; multi-panel grids carry
  their titles on each panel.
- `optimizer_fixef_plot` from the R package is not ported, as there is no clean
  statsmodels equivalent of `lme4::allFit`.
