# Classification and survival

The classification metrics come from scikit-learn and the survival estimate from
lifelines; depictr redraws both under the shared theme. The survival figure adds
a number-at-risk table beneath the curves in one call.

```python exec="1" session="clf"
import io, base64, warnings

import matplotlib

matplotlib.use("Agg")  # render headlessly; no GUI backend during the build

from plotnine.exceptions import PlotnineWarning

warnings.filterwarnings("ignore", category=PlotnineWarning)


def show(p, width=7, height=5):
    buf = io.BytesIO()
    p.save(buf, format="png", width=width, height=height, dpi=110, verbose=False)
    print('<img alt="depictr plot" src="data:image/png;base64,'
          + base64.b64encode(buf.getvalue()).decode() + '">')
```

```python exec="1" source="material-block" session="clf"
import numpy as np

import depictr as dp

ct = dp.clinical_trial()
score = 1 / (1 + np.exp(-ct["biomarker"]))  # a probability-like score
```

## ROC curve

An ROC curve with the area under the curve.

```python exec="1" source="material-block" session="clf"
p = dp.roc_curve_plot(ct["adverse_event"], score)
show(p)
```

## Calibration

A calibration (reliability) curve.

```python exec="1" source="material-block" session="clf"
p = dp.calibration_plot(ct["adverse_event"], score)
show(p)
```

## Confusion matrix

A confusion-matrix heatmap, normalised by the true class.

```python exec="1" source="material-block" session="clf"
p = dp.confusion_matrix_plot(ct["adverse_event"], (score > 0.12).astype(int),
                             normalise="true")
show(p, width=6)
```

## Survival

Kaplan-Meier curves with a log-rank test and a number-at-risk table. The legend
sits in the empty top-right the descending curves leave behind.

```python exec="1" source="material-block" session="clf"
p = dp.survival_plot(ct["time"], ct["event"], group=ct["arm"],
                     risk_table=True, legend_inside=True, title="Survival by arm")
show(p, width=8, height=6)
```
