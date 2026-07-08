# Model estimates and diagnostics

Fit a model once, then read it from several angles: a coefficient forest, the
predicted effect of a predictor, an estimation plot of a group difference, and a
panel of regression diagnostics.

```python exec="1" session="models"
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

```python exec="1" source="material-block" session="models"
import statsmodels.formula.api as smf

import depictr as dp

cy = dp.crop_yield()
# Q() quotes "yield" because it is a Python keyword.
model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph + treatment', cy).fit()
```

## Coefficient forest

A dot-and-whisker forest of the coefficients.

```python exec="1" source="material-block" session="models"
p = dp.coefficient_plot(model, title="Drivers of crop yield")
show(p)
```

## Predicted effect

The predicted response as one predictor varies, with a confidence band.

```python exec="1" source="material-block" session="models"
p = dp.effects_plot(model, "fertiliser")
show(p)
```

## Estimation plot

A two-panel Gardner-Altman estimation plot of the group difference.

```python exec="1" source="material-block" session="models"
p = dp.estimation_plot(cy, "yield", "treatment", two_panel=True, seed=1)
show(p, width=9)
```

## Residual diagnostics

The four-panel residual-diagnostic dashboard.

```python exec="1" source="material-block" session="models"
p = dp.residual_diagnostics_plot(model)
show(p, width=9, height=7)
```
