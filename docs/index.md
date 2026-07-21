# depictr <span class="mrd-lang">(Python)</span>

<p class="mrd-tagline">A unified, colourblind-safe toolkit of publication-ready plots that span the whole analysis workflow, built on <a href="https://plotnine.org">plotnine</a>.</p>

This is the Python sibling of [the R package](https://pablobernabeu.github.io/depictr/) of the same name. Both give the whole workflow one theme, one colourblind-safe palette and one calling convention.

[Get started](getting-started.md){ .md-button .md-button--primary }
[Open the gallery](gallery/exploring-data.md){ .md-button }

A single figure set for a paper might otherwise draw on seaborn, scikit-learn, statsmodels, lifelines and ArviZ, each with its own defaults, its own API and its own colour scheme, and only one of those defaulting to colourblind-safe colours. depictr does that theming work once, and returns plotnine objects, so a plot can be refined further with the usual `+` syntax. Where a specialist package already computes a quantity well, depictr delegates to it and redraws the result under the shared theme, so you keep the trusted computation and gain a coherent, accessible figure.

## Accessibility by default

The default palette is the Okabe-Ito set, and that choice is checked rather than asserted. A built-in simulator and a colour-distance test confirm that the palette stays legible under each form of colour-vision deficiency. The [accessibility page](gallery/accessibility.md) of the gallery renders the full safety report.

## Install

depictr is on [PyPI](https://pypi.org/project/depictr/):

```bash
pip install depictr            # core (plotnine, pandas, numpy, matplotlib, scipy)
pip install depictr[all]       # plus the optional computation back-ends
```

The model, classification and survival plots delegate to optional back-ends, so the core install stays light. The [Get started](getting-started.md) guide lists the extras.

## A first plot

```python exec="1" html="1" session="home"
import sys; sys.path.insert(0, "docs")

from _exec import show
```

```python exec="1" html="1" source="material-block" session="home"
import depictr as dp

ld = dp.lexical_decision()
p = dp.explore_distribution(ld, "RT", group="condition", kind="both")
print(show(p))
```

That figure was drawn when this page was built, as every figure on the site is,
so the code above it is the code that produced it.

## Where to look next

```python exec="1" html="1" session="home"
wb = dp.wellbeing_survey()
ct = dp.clinical_trial()

thumbnails = [
    ("gallery/exploring-data/", "Exploring data",
     dp.correlation_heatmap(wb)),
    ("gallery/multivariate-and-time-series/", "Multivariate and time series",
     dp.pca_plot(wb, group="region")),
    ("gallery/classification-and-survival/", "Classification and survival",
     dp.survival_plot(ct["time"], ct["event"], group=ct["arm"])),
]

cards = "".join(
    f'<a class="dp-thumb" href="{href}">{show(p, width=4.6, height=3.4, alt=label)}'
    f"<span>{label}</span></a>"
    for href, label, p in thumbnails
)
print(f'<div class="dp-thumbs">{cards}</div>')
```

The [gallery](gallery/exploring-data.md) works through every family with the plots rendered, and the [reference](reference/theme-and-accessibility.md) documents each function.

Archived on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21266311.svg)](https://doi.org/10.5281/zenodo.21266311)
