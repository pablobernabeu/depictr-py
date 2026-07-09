# depictr <span class="mrd-lang">(Python)</span>

<p class="mrd-tagline">A unified, colourblind-safe toolkit of publication-ready plots that span the whole analysis workflow, built on <a href="https://plotnine.org">plotnine</a>.</p>

This is the Python sibling of [the R package](https://pablobernabeu.github.io/depictr/) of the same name, which mirrors the same design. Both give the whole workflow one theme, one colourblind-safe palette and one calling convention.

[Get started](getting-started.md){ .md-button .md-button--primary }
[Open the gallery](gallery/exploring-data.md){ .md-button }

A single figure set for a paper might otherwise draw on seaborn, scikit-learn, statsmodels, lifelines and ArviZ, each with its own defaults, its own API and its own colour scheme, and only one of those defaulting to colourblind-safe colours. depictr does that theming work once, and returns plotnine objects, so a plot can be refined further with the usual `+` syntax. Where a specialist package already computes a quantity well, depictr delegates to it and redraws the result under the shared theme, so you keep the trusted computation and gain a coherent, accessible figure.

## Accessibility by default

The default palette is the Okabe-Ito set, and that choice is checked rather than asserted. The package ships a simulator of colour-vision deficiency based on the model of Machado, Oliveira and Fernandes (2009) and a CIE-Lab distance test that measures how far apart the palette's colours stay under each deficiency:

```python
import depictr as dp

dp.palette_safety()
# {'min_delta_e': ..., 'by_condition': {'normal': ..., 'protan': ...,
#  'deutan': ..., 'tritan': ...}, 'safe': True, 'threshold': 5.0}
```

## Install

depictr is on [PyPI](https://pypi.org/project/depictr/):

```bash
pip install depictr            # core (plotnine, pandas, numpy, matplotlib, scipy)
pip install depictr[all]       # plus the optional computation back-ends
```

The classification, model and survival plots delegate to scikit-learn, statsmodels and lifelines; each is an optional dependency, so the core stays light.

## A first plot

```python
import depictr as dp

ld = dp.lexical_decision()
dp.explore_distribution(ld, "RT", group="condition", kind="both")
```

The [gallery](gallery/exploring-data.md) works through every family with the plots rendered, and the [reference](reference/theme-and-accessibility.md) documents each function.

Archived on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21266311.svg)](https://doi.org/10.5281/zenodo.21266311)
