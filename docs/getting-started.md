# Get started

This guide works through a short analysis with depictr, from a first look at the
data to a fitted model and its diagnostics. Every plotting function returns a
[plotnine](https://plotnine.org) object, so you can customise it further with
the usual `+` syntax.

## Install

The first command below installs the core (plotnine, pandas, numpy, matplotlib
and scipy), and the second adds the optional computation back-ends. The double
quotes keep zsh, the default shell on macOS, from reading the brackets as a
filename pattern.

```bash
pip install depictr
pip install "depictr[all]"
```

The exploratory, theme and accessibility tools work with the core install. So
do the model and time-series plots, which use statsmodels, a requirement of
plotnine itself. The classification and survival plots use scikit-learn and
lifelines, which come with the optional extras `depictr[classification]` and
`depictr[survival]`. The PCA, scree, cluster and silhouette plots need
scikit-learn too, from the `depictr[classification]` extra, while the
dendrogram needs only scipy, a core dependency. The gains, lift and calibration
curves are computed by depictr itself and need neither. The `depictr[models]`
extra only sets the tested minimum version of statsmodels.

## The idea

All plotting functions follow the same calling convention, and all plots use
the same theme and colourblind-safe palette, which is based on the Okabe-Ito
set. Specialist computations are left to the packages that provide them, and
each result is redrawn under the shared theme.

```python
import depictr as dp
```

In a Jupyter notebook, a returned plot renders on display. In a script, call
`.show()`, or save it with `dp.save_plot(p, "fig.png")`.

## A first look at the data

depictr includes the R package's synthetic datasets, among them a
lexical-decision experiment with reaction times in two priming conditions.

```python
ld = dp.lexical_decision()
dp.explore_distribution(ld, "RT", group="condition", kind="density",
                        legend_inside=True)
```

The legend sits inside the panel, in the corner the distribution leaves empty.

A correlation heatmap and a missing-data map give a quick overview of a whole
dataset:

```python
wb = dp.wellbeing_survey()
dp.correlation_heatmap(wb)
dp.missingness_map(wb, legend_inside=True)
```

## Fitting and reading a model

depictr does not fit models. It reads a model you have fitted, or a tidy table
of estimates. Fit an ordinary least-squares model with statsmodels, then plot
its estimates, effects and diagnostics.

```python
import statsmodels.formula.api as smf

cy = dp.crop_yield()
# Q() quotes "yield" because it is a Python keyword.
model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph + treatment',
                cy).fit()

dp.coefficient_plot(model, title="Drivers of crop yield")
dp.effects_plot(model, "fertiliser")
dp.residual_diagnostics_plot(model)
```

The forest plot gives each term a readable label. The treatment row reads
`treatment: enhanced`, the contrast of enhanced fields with the reference level,
standard, and `soil_ph` reads `soil ph`. `labels` renames any term,
`labels="raw"` keeps the model's own names, and `format_terms` applies the same
rules to any list of names.

`coefficient_plot` also accepts a plain data frame of estimates with columns
`term`, `estimate`, `conf_low` and `conf_high`, so estimates from any source,
such as Bayesian posteriors, bootstrap intervals or figures taken from a paper,
plot the same way. `tidy_estimates` is the shared converter that produces that
standard table from a fitted model exposing `params` and `conf_int`, such as a
statsmodels OLS, GLM or MixedLM result. `coefficient_plot`, `compare_models`
and `frequentist_bayesian_plot` accept either the model or that table. The
prediction, random-effects and diagnostic plots need the fitted model itself,
although `qq_plot` also takes an array of residuals. `posterior_plot` takes
posterior draws.

```python
dp.tidy_estimates(model)
```

## Survival and classification

Kaplan-Meier curves with confidence limits, a log-rank test and a
number-at-risk table take one call:

```python
ct = dp.clinical_trial()
dp.survival_plot(ct["time"], ct["event"], group=ct["arm"],
                 risk_table=True, legend_inside=True, logrank=True)
dp.roc_curve_plot(ct["adverse_event"], ct["biomarker"])
```

## Checking the accessibility claim

The claim that the default palette is colourblind-safe can be tested with a
simulator of colour-vision deficiency, based on the model of Machado, Oliveira
and Fernandes (2009), and a CIEDE2000 colour-difference test. Together, they
measure how far apart the palette's colours stay under each form of deficiency.

```python
dp.palette_safety()
# {'min_delta_e': ..., 'safe': True, 'by_condition': {...}, ...}
```

A figure built from a safe palette can still be inaccessible, so `check_figure`
audits the finished plot. It measures how separable the encoding colours are
under each dichromacy and in greyscale, and how small the text becomes at the
width the figure will be printed at. It also measures how well text and
geometry contrast with their backgrounds under WCAG, and whether any
distinction rests on colour alone. Each row reports the value it measured next
to the threshold it was measured against.

```python
dp.check_figure(dp.explore_distribution(ld, "RT", group="condition"),
                width_cm=8.9)
```

The gallery's [accessibility page](gallery/accessibility.md) renders both reports
in full and states the palette's limitations. The Okabe-Ito guarantee covers
hue confusion alone, and two of the palette's colours print as the same grey.
The palette's orange, the second colour, also falls below the 3:1 contrast
against white that WCAG asks of a graphical object.

## Extending and composing

plotnine's grammar-of-graphics extensions apply to every depictr plot:

```python
from plotnine import labs

dp.roc_curve_plot(
    ct["adverse_event"], ct["biomarker"]
) + labs(title="Adverse event")
```

To place several plots in one figure, use `arrange_plots`. It tags the panels
A, B and so on, as the R function does, and `tag_levels=None` leaves them
untagged:

```python
dp.arrange_plots(dp.qq_plot(model), dp.influence_plot(model), ncol=2)
```

`theme_depictr`, `depictr_palette` and `scale_colour_depictr` style your own
plots too:

```python
from plotnine import ggplot, aes, geom_point

(ggplot(cy, aes("fertiliser", "yield", colour="treatment"))
 + geom_point(alpha=0.7)
 + dp.scale_colour_depictr()
 + dp.theme_depictr())
```

For a journal that names a font, `dp.theme_depictr(base_family="Arial")` sets
it on one plot. plotnine's own option sets it on every plot made afterwards,
depictr's included, as `depictr_options(base_family = )` does in R:

```python
import plotnine

plotnine.options.base_family = "Arial"
```

A plot made before the option is set keeps its font.

## Where next

The [gallery](gallery/exploring-data.md) renders a worked example from every
family, and the [API reference](api.md) documents each
function and its options.
