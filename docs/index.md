# depictr <span class="mrd-lang">(Python)</span>

<p class="mrd-tagline">A unified, colourblind-safe toolkit of publication-ready plots that span the whole analysis workflow, built on <a href="https://plotnine.org">plotnine</a>.</p>

This is the Python sibling of [the R package](https://pablobernabeu.github.io/depictr/) of the same name. Both give the whole workflow one theme, one colourblind-safe palette and one calling convention.

[Get started](getting-started.md){ .md-button .md-button--primary }
[Open the gallery](gallery/exploring-data.md){ .md-button }
[Try the web app](app.md){ .md-button }

A single figure set for a paper might otherwise draw on seaborn, scikit-learn, statsmodels, lifelines and ArviZ, each with its own defaults, API and colour scheme. Only one of them uses colourblind-safe colours by default. depictr does the theming once and returns plotnine objects that can be refined further with the usual `+` syntax. Where a specialist package already computes a quantity well, depictr delegates the computation to it and redraws the result under the shared theme.

## Accessibility by default

The default palette is based on the Okabe-Ito set. The package's simulator of colour-vision deficiency and its colour-distance test confirm that the palette stays legible under each form of deficiency. The colourblind-safety guarantee covers the eight built-in colours. A figure with more than eight groups needs colours interpolated between them, and depictr warns when it interpolates. The guarantee also covers only hue confusion and makes no claim about greyscale.

`check_figure` audits the figure itself. It measures the figure's colour separability under each dichromacy and in greyscale, its text size at the printed width and the contrast of its text and geometry. The [accessibility page](gallery/accessibility.md) of the gallery shows the safety report and a figure audit.

## Install

depictr is on [PyPI](https://pypi.org/project/depictr/). The first command below installs the core (plotnine, pandas, numpy, matplotlib and scipy), and the second adds the optional computation back-ends and Streamlit, which the web app uses. The double quotes keep zsh, the default shell on macOS, from reading the brackets as a filename pattern.

```bash
pip install depictr
pip install "depictr[all]"
```

The ROC, precision-recall and confusion-matrix plots, the survival plots and every multivariate plot except the dendrogram delegate to optional back-ends. The model plots use statsmodels, which is installed with plotnine. The [Get started](getting-started.md) guide lists the extras.

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

That figure, like every figure on the site, was drawn from the code above it
when the page was built.

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
     dp.survival_plot(ct["time"], ct["event"], group=ct["arm"], logrank=True)),
]

cards = "".join(
    f'<a class="dp-thumb" href="{href}">{show(p, width=4.6, height=3.4, alt=label)}'
    f"<span>{label}</span></a>"
    for href, label, p in thumbnails
)
print(f'<div class="dp-thumbs">{cards}</div>')
```

The [gallery](gallery/exploring-data.md) works through every family with the plots rendered, and the [API reference](api.md) documents each function.

If depictr contributes to published work, please [cite it](about.md#citing-depictr). Its citation gives the concept DOI <https://doi.org/10.5281/zenodo.21266311>, which resolves to the latest release archived on Zenodo.
