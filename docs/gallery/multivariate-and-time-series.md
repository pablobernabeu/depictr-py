# Multivariate and time series

Principal components and clustering (via scikit-learn), and a seasonal
decomposition (via statsmodels), all redrawn in the depictr theme.

```python exec="1" session="mv"
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

```python exec="1" source="material-block" session="mv"
import numpy as np
import pandas as pd

import depictr as dp

wb = dp.wellbeing_survey()
```

## PCA biplot

A PCA biplot, coloured by group, with the variable loadings.

```python exec="1" source="material-block" session="mv"
p = dp.pca_plot(wb, group="region")
show(p, width=8, height=6)
```

## Clustering

k-means clusters on the first two principal components.

```python exec="1" source="material-block" session="mv"
p = dp.cluster_plot(wb, k=3)
show(p)
```

## Seasonal decomposition

A monthly series with a trend, a 12-period season and noise, and its
decomposition into observed, trend, seasonal and residual components.

```python exec="1" source="material-block" session="mv"
rng = np.random.default_rng(0)
t = np.arange(120)
series = pd.Series(
    50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 120),
    index=pd.period_range("2016-01", periods=120, freq="M"),
)

p = dp.decompose_plot(series, period=12)
show(p, width=8, height=8)
```
