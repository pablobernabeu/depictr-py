# API reference

Every public name in `depictr` is documented here, grouped along the path an analysis takes:
look at the data, reduce it or follow it over time, read the model, check the model, judge a
classifier, set the look, and compose and save the result. The groups follow the
[R package's reference index](https://pablobernabeu.github.io/depictr/reference/) in the same
order, so a function can be looked up in the same place on either site.

The two indexes are cut slightly differently in the few places where the Python package's own
families do not line up with the R one. Time series has no group of its own here, and those
functions close the 'Multivariate, survival and time series' group, which is where the Python
package has always documented them. The R index's 'Diagnostics and classification' is split in
two, because the classification curves are all computed by scikit-learn and reached through a
separate install extra. R's 'Uncertainty and power' has no counterpart either, and the posterior
and power curves are documented with the other model estimates they are drawn from. What the R
index calls 'Theming, accessibility and reporting' appears here as 'Theme, palette and
accessibility', covering the theme, the colour-vision checks and the figure audit, while plot
composition and saving move to the last group beside the datasets and the one-figure model report
stays with the diagnostics it collects.

Everything listed under a group heading is importable straight from `depictr`. The same plots are
shown rendered, with the code that produced them, in the [gallery](gallery/exploring-data.md).

## Exploratory analysis

A first look at a dataset: distributions, categories, relationships, correlation,
cumulative distributions, outliers and a missing-data map.

These plots are shown, with the code that produced them, on the
[exploring data](gallery/exploring-data.md) page of the gallery.

::: depictr.explore_distribution
::: depictr.explore_categorical
::: depictr.explore_bivariate
::: depictr.scatter_trend
::: depictr.correlation_heatmap
::: depictr.missingness_map
::: depictr.ecdf_plot
::: depictr.ridgeline_plot
::: depictr.dumbbell_plot
::: depictr.outlier_plot
::: depictr.group_comparison_plot
::: depictr.raincloud_plot
::: depictr.explore_pairs

## Multivariate, survival and time series

Principal components and clustering, Kaplan-Meier survival, and the time-series
family (series, autocorrelation, decomposition and seasonal views).

These plots are shown, with the code that produced them, on the
[multivariate and time series](gallery/multivariate-and-time-series.md) page
of the gallery, and the survival curve on the
[classification and survival](gallery/classification-and-survival.md) page.

::: depictr.pca_plot
::: depictr.scree_plot
::: depictr.cluster_plot
::: depictr.dendrogram_plot
::: depictr.silhouette_plot
::: depictr.survival_plot
::: depictr.acf_plot
::: depictr.decompose_plot
::: depictr.seasonal_plot
::: depictr.timeseries_plot

## Estimation and model estimates

Estimation plots and summary tables, and the model-estimate family: forests,
predicted effects, interactions, random effects, posteriors and power.

These plots are shown, with the code that produced them, on the
[model estimates and diagnostics](gallery/model-estimates.md) page of the
gallery.

::: depictr.estimation_plot
::: depictr.summary_table
::: depictr.coefficient_plot
::: depictr.tidy_estimates
::: depictr.effects_plot
::: depictr.interaction_plot
::: depictr.compare_models
::: depictr.random_effects_plot
::: depictr.posterior_plot
::: depictr.frequentist_bayesian_plot
::: depictr.power_curve_plot

## Diagnostics

Residual and influence diagnostics, and a one-figure model report.

These plots are shown, with the code that produced them, on the
[model estimates and diagnostics](gallery/model-estimates.md) page of the
gallery.

::: depictr.qq_plot
::: depictr.influence_plot
::: depictr.vif_plot
::: depictr.binned_residual_plot
::: depictr.residual_diagnostics_plot
::: depictr.model_report

## Classification

The standard classification curves and tables, computed by scikit-learn and
redrawn under the shared theme.

These plots are shown, with the code that produced them, on the
[classification and survival](gallery/classification-and-survival.md) page of
the gallery.

::: depictr.roc_curve_plot
::: depictr.pr_curve_plot
::: depictr.confusion_matrix_plot
::: depictr.calibration_plot
::: depictr.gain_plot
::: depictr.lift_plot
::: depictr.threshold_plot

## Theme, palette and accessibility

The shared theme and the colourblind-safe palette, the tools that verify the
palette stays distinguishable under colour-vision deficiency, and the audit that
checks a finished figure rather than the palette it was built from.

The palette, the colour-vision checks and the figure audit are shown in use on the
[accessibility](gallery/accessibility.md) page of the gallery.

::: depictr.theme_depictr
::: depictr.scale_colour_depictr
::: depictr.scale_color_depictr
::: depictr.scale_fill_depictr
::: depictr.legend_inside
::: depictr.depictr_palette
::: depictr.depictr_brand
::: depictr.depictr_accent
::: depictr.check_figure
::: depictr.palette_safety
::: depictr.simulate_cvd

## Composition and data

Compose several plots into one figure or save one to disk, and the reproducibly
simulated datasets used throughout the documentation.

The datasets are used throughout the [gallery](gallery/exploring-data.md),
where every figure is drawn from one of them.

::: depictr.arrange_plots
::: depictr.save_plot
::: depictr.crop_yield
::: depictr.wellbeing_survey
::: depictr.lexical_decision
::: depictr.clinical_trial
::: depictr.DATASETS
