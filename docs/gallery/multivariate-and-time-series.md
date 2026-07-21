# Multivariate and time series

Principal components and clustering (via scikit-learn), and the series, its
autocorrelation and a seasonal decomposition (via statsmodels), all redrawn in
the depictr theme.

```python exec="1" html="1" session="mv"
import sys; sys.path.insert(0, "docs")

from _exec import show
```

```python exec="1" html="1" source="material-block" session="mv"
import numpy as np
import pandas as pd

import depictr as dp

wb = dp.wellbeing_survey()
```

## PCA biplot

A PCA biplot, coloured by group, with the variable loadings.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.pca_plot(wb, group="region")
print(show(p, width=8, height=6))
```

## Clustering

k-means clusters on the first two principal components.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.cluster_plot(wb, k=3)
print(show(p))
```

## Scree plot

The variance each principal component explains, with the running cumulative line.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.scree_plot(wb)
print(show(p))
```

## Dendrogram

A hierarchical-clustering tree over the regional profiles.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.dendrogram_plot(wb.groupby("region").mean(numeric_only=True))
print(show(p))
```

## Silhouette

Each observation's silhouette width, grouped and ordered by cluster. Wide
positive bars are well-placed observations, and a negative bar may belong to a
neighbouring cluster.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.silhouette_plot(wb, k=3)
print(show(p))
```

## Seasonal decomposition

A monthly series with a trend, a 12-period season and noise, and its
decomposition into observed, trend, seasonal and residual components.

```python exec="1" html="1" source="material-block" session="mv"
rng = np.random.default_rng(0)
t = np.arange(120)
series = pd.Series(
    50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 120),
    index=pd.period_range("2016-01", periods=120, freq="M"),
)

p = dp.decompose_plot(series, period=12)
print(show(p, width=8, height=8))
```

## The series with a rolling mean

The same series as a line, with a twelve-month centred rolling mean over it.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.timeseries_plot(series, rolling=12)
print(show(p))
```

## Autocorrelation

The autocorrelation and partial autocorrelation functions, with an approximate
95% band. The spikes at multiples of twelve are the seasonality.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.acf_plot(series)
print(show(p))
```

```python exec="1" html="1" source="material-block" session="mv"
p = dp.acf_plot(series, kind="pacf")
print(show(p))
```

## The seasonal pattern

One line per year across the months, so the repeating shape and any drift
between years are easy to read.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.seasonal_plot(series, period=12)
print(show(p))
```
