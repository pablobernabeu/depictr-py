# depictr (Python)

A unified, colourblind-safe toolkit for publication-ready statistical
visualisation, built on [plotnine](https://plotnine.org). It is the Python
sibling of the [depictr R package](https://github.com/pablobernabeu/depictr).

## What it is for

Python already has an excellent plot for almost any statistical task, but they
live in different packages with different defaults. A single figure set for a
paper might draw on seaborn, scikit-learn, statsmodels, lifelines and ArviZ,
each with its own look, its own API and its own colour scheme, and only one of
those defaults to colourblind-safe colours. Making the set consistent and
accessible then means repeating the same theming by hand on every plot.

depictr does that work once. It gives the whole workflow one theme, one
colourblind-safe palette and one calling convention, and returns plotnine
objects you can keep extending with `+`. Where a specialist package already
computes a quantity well, depictr delegates to it and redraws the result under
the shared theme, so you keep the trusted computation and gain a coherent,
accessible figure.

## Accessibility by default

The default palette is the Okabe-Ito set, and that choice is checked rather than
asserted. The package ships a simulator of colour-vision deficiency based on the
model of Machado, Oliveira and Fernandes (2009) and a CIE-Lab distance test that
measures how far apart the palette's colours stay under each deficiency:

```python
import depictr as dp

dp.palette_safety()
# {'min_delta_e': ..., 'by_condition': {'normal': ..., 'protan': ...,
#  'deutan': ..., 'tritan': ...}, 'safe': True, 'threshold': 10.0}
```

## Installation

```bash
pip install depictr            # core (plotnine, pandas, numpy)
pip install depictr[all]       # plus the optional computation back-ends
```

The classification, model and survival plots delegate to scikit-learn,
statsmodels and lifelines respectively. Each is an optional dependency, so the
core stays light and you install only what your plots need
(`depictr[classification]`, `depictr[models]`, `depictr[survival]`).

## A short tour

```python
import depictr as dp

# Exploratory analysis
ld = dp.lexical_decision()
dp.explore_distribution(ld, "RT", group="condition", kind="both")

wb = dp.wellbeing_survey()
dp.correlation_heatmap(wb)
dp.missingness_map(wb)

# Model estimates: a fitted model OR a tidy data frame
import statsmodels.formula.api as smf
# Q() quotes "yield" because it is a Python keyword.
fit = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph + treatment',
              dp.crop_yield()).fit()
dp.coefficient_plot(fit)

# Classification: computed by scikit-learn, themed by depictr
ct = dp.clinical_trial()
dp.roc_curve_plot(ct["adverse_event"], ct["biomarker"])

# Survival: Kaplan-Meier with a log-rank test, in one call
dp.survival_plot(ct["time"], ct["event"], group=ct["arm"])
```

Every function returns a plotnine object, so the usual grammar-of-graphics
extensions apply:

```python
from plotnine import labs

dp.roc_curve_plot(ct["adverse_event"], ct["biomarker"]) + labs(title="Adverse event")
```

## The web app

A Streamlit app provides a gallery and a low-friction way to try the package: it
loads one of the bundled datasets (or a CSV you upload), draws the chosen plot,
shows the exact Python call that produced it, and offers a colourblind-vision
toggle that re-renders the figure as each deficiency would be seen. Run it with:

```bash
pip install depictr[app]
streamlit run app/streamlit_app.py
```

## Function families

| Family | Functions |
| --- | --- |
| Theme and palette | `theme_depictr`, `scale_colour_depictr`, `scale_fill_depictr`, `depictr_palette` |
| Accessibility | `palette_safety`, `simulate_cvd` |
| Exploratory analysis | `explore_distribution`, `explore_categorical`, `correlation_heatmap`, `missingness_map` |
| Model estimates | `coefficient_plot`, `tidy_estimates` |
| Classification | `roc_curve_plot`, `pr_curve_plot`, `confusion_matrix_plot`, `calibration_plot`, `gain_plot` |
| Survival | `survival_plot` |
| Composition | `save_plot` |

## Status

This is an early release (0.1.0): a working foundation rather than a complete
port of the R package. The colourblind-safe theme, the accessibility check and a
representative function from each family are in place. The roadmap is to widen
coverage towards the R package's full surface (estimation plots, regression
diagnostics, multivariate and time-series families, a number-at-risk table for
the survival plot) while keeping the delegate-and-reskin approach.

## Relationship to the R package

The two are siblings, not a shared codebase. The R package targets ggplot2 and
CRAN; this one targets plotnine and PyPI. They share the design: one accessible
theme across the whole workflow, model-or-data-frame input, and extensible plot
objects.

## Licence

MIT. See [LICENSE](LICENSE).
