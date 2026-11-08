# Multivariate and time series

The principal components, the k-means clusters and the silhouette widths come
from scikit-learn, and the hierarchical clustering behind the dendrogram comes
from scipy. The autocorrelation and the seasonal decomposition of the series
come from statsmodels. depictr draws the series and redraws these results under
the shared theme.

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

k-means clusters of the standardised data, drawn on the first two principal
components. They are the clusters the silhouette plot below measures.

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
p = dp.dendrogram_plot(wb.groupby("region", observed=True).mean(numeric_only=True))
print(show(p))
```

## Silhouette

Each observation's silhouette width, grouped and ordered by cluster. Wide
positive bars are well-placed observations, and a bar below zero marks an
observation that may belong to a neighbouring cluster, because it sits closer
to that cluster than to its own.

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

The same series as a line, with a centred twelve-month moving average over it.
Twelve months is an even window, so the overlay is the 2×12 moving average. It
is a twelve-month average followed by a two-month one, which centres it on each
month.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.timeseries_plot(series, rolling=12)
print(show(p))
```

## Autocorrelation

The autocorrelation function, with an approximate 95% band. The trend keeps the
correlations high at short lags, and the seasonal cycle adds a wave to their
slow decay. The wave dips near lag six, where the series is half a cycle out of
step with itself. It peaks again at lag twelve, the only multiple of the period
within the default range of lags.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.acf_plot(series)
print(show(p))
```

The partial autocorrelation function, which leaves out what the shorter lags
already explain. Lag one carries most of the dependence, and the value at lag
twelve lies well inside the band.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.acf_plot(series, kind="pacf")
print(show(p))
```

## The seasonal pattern

One line per year across the months shows the repeating shape and any change
from year to year.

```python exec="1" html="1" source="material-block" session="mv"
p = dp.seasonal_plot(series, period=12)
print(show(p))
```
