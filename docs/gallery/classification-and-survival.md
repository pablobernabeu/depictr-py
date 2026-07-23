# Classification and survival

The classification metrics come from scikit-learn and the survival estimate from
lifelines; depictr redraws both under the shared theme. The survival figure adds
a number-at-risk table beneath the curves in one call.

```python exec="1" html="1" session="clf"
import sys; sys.path.insert(0, "docs")

from functools import partial

from _exec import show as _show

# This page's figures suit a slightly narrower canvas than the gallery default.
show = partial(_show, width=7)
```

```python exec="1" html="1" source="material-block" session="clf"
import numpy as np

import depictr as dp

ct = dp.clinical_trial()
score = 1 / (1 + np.exp(-ct["biomarker"]))  # a probability-like score
```

## ROC curve

An ROC curve with the area under the curve.

```python exec="1" html="1" source="material-block" session="clf"
p = dp.roc_curve_plot(ct["adverse_event"], score)
print(show(p))
```

## Precision-recall on a rare outcome

When the positive class is rare the precision-recall curve is more informative
than the ROC curve, because it ignores the many true negatives.

```python exec="1" html="1" source="material-block" session="clf"
p = dp.pr_curve_plot(ct["adverse_event"], score)
print(show(p))
```

## Gains, lift and choosing a threshold

The cumulative gains and lift charts read off how many positives a ranked
campaign captures at each depth.

```python exec="1" html="1" source="material-block" session="clf"
p = dp.gain_plot(ct["adverse_event"], score)
print(show(p))
```

```python exec="1" html="1" source="material-block" session="clf"
p = dp.lift_plot(ct["adverse_event"], score)
print(show(p))
```

The threshold plot sweeps the decision cut-off, so the trade-off between the
metrics as the operating point moves is visible at a glance.

```python exec="1" html="1" source="material-block" session="clf"
p = dp.threshold_plot(ct["adverse_event"], score)
print(show(p))
```

## Calibration

A calibration curve asks whether a predicted probability of, say, 0.3 is borne
out by an event in about 30% of such cases, so it needs probabilities from a
fitted model rather than the ranking score the charts above are happy with. We
fit a logistic regression and pass its predicted probabilities.

```python exec="1" html="1" source="material-block" session="clf"
from sklearn.linear_model import LogisticRegression

X = ct[["biomarker", "age"]]
fit = LogisticRegression().fit(X, ct["adverse_event"])
p = dp.calibration_plot(ct["adverse_event"], fit.predict_proba(X)[:, 1],
                        n_bins=5)
print(show(p))
```

The three populated bins predict 0.10, 0.27 and 0.45 against observed rates of
0.09, 0.29 and 0.50, so the curve tracks the diagonal. Only the highest bin is
thin, holding four patients out of 300.

## Confusion matrix

A confusion-matrix heatmap, normalised by the true class.

```python exec="1" html="1" source="material-block" session="clf"
p = dp.confusion_matrix_plot(ct["adverse_event"], (score > 0.6).astype(int),
                             normalise="true")
print(show(p, width=6))
```

## Survival

Kaplan-Meier curves with a log-rank test and a number-at-risk table. The legend
sits in the empty top-right the descending curves leave behind.

```python exec="1" html="1" source="material-block" session="clf"
p = dp.survival_plot(ct["time"], ct["event"], group=ct["arm"],
                     risk_table=True, legend_inside=True, title="Survival by arm")
print(show(p, width=8, height=6))
```
