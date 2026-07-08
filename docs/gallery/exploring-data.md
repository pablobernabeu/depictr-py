# Exploring data

A first look at a dataset, all in one theme: distributions, categories,
correlations, cumulative distributions and a missing-data map. Every figure below
is produced by running the code shown.

```python exec="1" session="eda"
import io, base64, warnings

import matplotlib

matplotlib.use("Agg")  # render headlessly; no GUI backend during the build

from plotnine.exceptions import PlotnineWarning

warnings.filterwarnings("ignore", category=PlotnineWarning)


def show(p, width=8, height=5):
    buf = io.BytesIO()
    p.save(buf, format="png", width=width, height=height, dpi=110, verbose=False)
    print('<img alt="depictr plot" src="data:image/png;base64,'
          + base64.b64encode(buf.getvalue()).decode() + '">')
```

```python exec="1" source="material-block" session="eda"
import depictr as dp

wb = dp.wellbeing_survey()
ld = dp.lexical_decision()
```

## Cumulative distribution by group

An empirical cumulative distribution, with the legend tucked into the
bottom-right the curves leave empty once they saturate.

```python exec="1" source="material-block" session="eda"
p = dp.ecdf_plot(ld, "RT", group="condition", legend_inside=True)
show(p)
```

## A categorical comparison

```python exec="1" source="material-block" session="eda"
p = dp.explore_categorical(wb, "education", group="region")
show(p)
```

## Correlation heatmap

```python exec="1" source="material-block" session="eda"
p = dp.correlation_heatmap(wb)
show(p)
```

## Raincloud

Density, box and raw points together.

```python exec="1" source="material-block" session="eda"
p = dp.raincloud_plot(ld, "RT", group="condition")
show(p)
```

## Missing-data map

Columns are ordered most- to least-missing, so the legend sits inside the
top-right over the most-complete columns.

```python exec="1" source="material-block" session="eda"
p = dp.missingness_map(wb, legend_inside=True)
show(p)
```
