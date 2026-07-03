# depictr

**depictr** is a unified, colourblind-safe toolkit of publication-ready plots
that span the whole analysis workflow, from a first look at the data, through
model estimates and predictions, to diagnostics, classification and survival
analysis. A single figure set for a paper might otherwise draw on seaborn,
scikit-learn, statsmodels, lifelines and ArviZ, each with its own defaults,
its own API and its own colour scheme, and only one of those defaulting to
colourblind-safe colours. depictr does that theming work once: it gives the
whole workflow *one* theme, *one* colourblind-safe palette and *one* calling
convention, and returns [plotnine](https://plotnine.org) objects, so a plot
can be refined further with the usual `+` syntax. Where a specialist package
already computes a quantity well, depictr delegates to it (scikit-learn,
statsmodels, lifelines) and redraws the result under the shared theme, so you
keep the trusted computation and gain a coherent, accessible figure.

An R sibling, [depictr](https://github.com/pablobernabeu/depictr), mirrors
the same design.

![A grouped density of response times by condition, in the colourblind-safe Okabe-Ito palette](https://raw.githubusercontent.com/pablobernabeu/depictr-py/main/images/readme-distribution.png)

```{toctree}
:maxdepth: 1
:hidden:

Getting started <getting-started>
Gallery <auto_examples/index>
API reference <api>
About <about>
```

## Accessibility by default

The default palette is the Okabe-Ito set, and that choice is checked rather
than asserted. The package ships a simulator of colour-vision deficiency
based on the model of Machado, Oliveira and Fernandes (2009) and a CIE-Lab
distance test that measures how far apart the palette's colours stay under
each deficiency:

```python
import depictr as dp

dp.palette_safety()
# {'min_delta_e': ..., 'by_condition': {'normal': ..., 'protan': ...,
#  'deutan': ..., 'tritan': ...}, 'safe': True, 'threshold': 5.0}
```

## Installation

depictr is on [PyPI](https://pypi.org/project/depictr/):

```bash
pip install depictr            # core (plotnine, pandas, numpy)
pip install depictr[all]       # plus the optional computation back-ends
```

The classification, model and survival plots delegate to scikit-learn,
statsmodels and lifelines; each is an optional dependency, so the core stays
light.

## A first plot

```python
import depictr as dp

ld = dp.lexical_decision()
dp.explore_distribution(ld, "RT", group="condition", kind="both")
```

The [gallery](auto_examples/index) works through every family with the plots
rendered, and the [API reference](api) documents each function.
