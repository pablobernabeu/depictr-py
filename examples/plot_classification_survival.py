"""
Classification and survival
===========================

The classification metrics come from scikit-learn and the survival estimate from
lifelines; depictr redraws both under the shared theme. The survival figure adds
a number-at-risk table beneath the curves in one call.
"""

import numpy as np

import depictr as dp

ct = dp.clinical_trial()
score = 1 / (1 + np.exp(-ct["biomarker"]))  # a probability-like score

# %%
# An ROC curve with the area under the curve.
p = dp.roc_curve_plot(ct["adverse_event"], score)
p

# %%
# A calibration (reliability) curve.
p = dp.calibration_plot(ct["adverse_event"], score)
p

# %%
# A confusion-matrix heatmap, normalised by the true class.
p = dp.confusion_matrix_plot(ct["adverse_event"], (score > 0.12).astype(int),
                             normalise="true")
p

# %%
# Kaplan-Meier curves with a log-rank test and a number-at-risk table.
p = dp.survival_plot(ct["time"], ct["event"], group=ct["arm"],
                     risk_table=True, title="Survival by arm")
p
