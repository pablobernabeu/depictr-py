# Getting started

This guide works through a short analysis with depictr, from a first look at the
data to a fitted model and its diagnostics. Every function returns a
[plotnine](https://plotnine.org) object, so anything shown here can be refined
further with the usual `+` syntax.

## Install

```bash
pip install depictr            # core: plotnine, pandas, numpy, matplotlib, scipy
pip install depictr[all]       # plus the optional computation back-ends
```

The exploratory, theme and accessibility tools work with the core install. The
model, classification and survival plots delegate their computation to
statsmodels, scikit-learn and lifelines respectively, each an optional extra
(`depictr[models]`, `depictr[classification]`, `depictr[survival]`).

## The idea

depictr gives the whole workflow one theme, one colourblind-safe palette (the
Okabe-Ito set) and one calling convention. Where a specialist package already
computes a quantity well, depictr hands the work to it and redraws the result
under the shared theme, so a ROC curve, a coefficient plot and a survival curve
all share the same visual language.

```python
import depictr as dp
```

Each call returns a plotnine object. In a Jupyter notebook it renders on
display; in a script, call `.show()`, or save it with `dp.save_plot(p, "fig.png")`.

## A first look at the data

depictr ships a few reproducibly simulated datasets. Here is a lexical-decision
experiment with reaction times in two priming conditions.

```python
ld = dp.lexical_decision()
dp.explore_distribution(ld, "RT", group="condition", kind="density",
                        legend_inside=True)
```

The legend sits inside the panel, in the corner the distribution leaves empty.
For a wider survey, a correlation heatmap and a missing-data map give a quick
overview:

```python
wb = dp.wellbeing_survey()
dp.correlation_heatmap(wb)
dp.missingness_map(wb)
```

## Fitting and reading a model

depictr does not fit models; it reads a model you have fitted, or a tidy table
of estimates. Fit an ordinary least-squares model with statsmodels, then read it
from several angles.

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

`coefficient_plot` also accepts a plain data frame of estimates (with columns
`term`, `estimate`, `conf_low`, `conf_high`), so estimates from any source -- a
Bayesian fit, a bootstrap, a table copied from a paper -- plot the same way.

## Survival and classification

These families delegate to lifelines and scikit-learn. Kaplan-Meier curves with
a log-rank test and a number-at-risk table are one call:

```python
ct = dp.clinical_trial()
dp.survival_plot(ct["time"], ct["event"], group=ct["arm"],
                 risk_table=True, legend_inside=True)
dp.roc_curve_plot(ct["adverse_event"], ct["biomarker"])
```

## Accessibility, checked rather than asserted

The default palette is the Okabe-Ito set, and that choice is verified rather than
assumed. A Machado-2009 simulator and a CIE-Lab distance test report how far
apart the palette's colours stay under each form of colour-vision deficiency.

```python
dp.palette_safety()
# {'min_delta_e': ..., 'safe': True, 'by_condition': {...}, ...}
```

## Extending and composing

Because every function returns a plotnine object, the grammar-of-graphics
extensions apply:

```python
from plotnine import labs

dp.roc_curve_plot(ct["adverse_event"], ct["biomarker"]) + labs(title="Adverse event")
```

To place several plots in one figure, use `arrange_plots`:

```python
dp.arrange_plots(dp.qq_plot(model), dp.influence_plot(model), ncol=2)
```

## Where next

- The [gallery](gallery/exploring-data.md) renders a worked example from every family.
- The [reference](reference/theme-and-accessibility.md) documents each function and its options.
