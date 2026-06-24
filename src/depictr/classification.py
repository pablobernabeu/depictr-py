"""Classification-metric plots.

The metrics are computed by ``scikit-learn`` (the de-facto standard) and the
figures are re-drawn under the depictr theme, so a ROC curve sits in the same
visual language as the rest of a report. Cumulative gains, which scikit-learn
does not provide, is included for completeness.

Install the optional dependency with ``pip install depictr[classification]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
    geom_abline,
    geom_line,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradientn,
)

from .palette import BRAND, depictr_palette
from .theme import theme_depictr


def _require_sklearn():
    try:
        import sklearn.metrics as m  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The classification plots need scikit-learn. Install it with "
            "`pip install depictr[classification]`."
        ) from exc
    return m


def roc_curve_plot(y_true, y_score, title=None):
    """ROC curve with the area under the curve (AUC) annotated.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0/1).
    y_score : array-like
        Predicted scores or probabilities for the positive class.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
    """
    m = _require_sklearn()
    fpr, tpr, _ = m.roc_curve(y_true, y_score)
    auc = m.auc(fpr, tpr)
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    return (
        ggplot(df, aes("fpr", "tpr"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="#9e9e9e")
        + geom_line(color=BRAND, size=0.9)
        + annotate("text", x=0.98, y=0.04, ha="right",
                   label=f"AUC = {auc:.3f}", color=BRAND, fontweight="bold")
        + labs(x="False positive rate", y="True positive rate", title=title)
        + theme_depictr()
    )


def pr_curve_plot(y_true, y_score, title=None):
    """Precision-recall curve with the average precision (AP) annotated."""
    m = _require_sklearn()
    precision, recall, _ = m.precision_recall_curve(y_true, y_score)
    ap = m.average_precision_score(y_true, y_score)
    baseline = float(np.mean(np.asarray(y_true)))
    df = pd.DataFrame({"recall": recall, "precision": precision})
    return (
        ggplot(df, aes("recall", "precision"))
        + geom_abline(intercept=baseline, slope=0, linetype="dashed", color="#9e9e9e")
        + geom_line(color=BRAND, size=0.9)
        + annotate("text", x=0.98, y=0.04, ha="right",
                   label=f"AP = {ap:.3f}", color=BRAND, fontweight="bold")
        + labs(x="Recall", y="Precision", title=title)
        + theme_depictr()
    )


def confusion_matrix_plot(y_true, y_pred, normalise=None, title=None):
    """Confusion-matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    normalise : {None, "true", "pred", "all"}
        Passed to scikit-learn's ``confusion_matrix(normalize=...)``.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
    """
    m = _require_sklearn()
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    cm = m.confusion_matrix(y_true, y_pred, labels=labels, normalize=normalise)
    long = (pd.DataFrame(cm, index=labels, columns=labels)
            .reset_index(names="true")
            .melt(id_vars="true", var_name="predicted", value_name="count"))
    fmt = "{:.2f}" if normalise else "{:.0f}"
    long["label"] = long["count"].map(lambda v: fmt.format(v))
    long["true"] = pd.Categorical(long["true"], categories=labels[::-1], ordered=True)
    long["predicted"] = pd.Categorical(long["predicted"], categories=labels, ordered=True)
    return (
        ggplot(long, aes("predicted", "true", fill="count"))
        + geom_tile(color="white")
        + geom_text(aes(label="label"), color="#1a1a1a", size=10)
        + scale_fill_gradientn(colors=depictr_palette(7, kind="sequential"),
                               name=("Proportion" if normalise else "Count"))
        + labs(x="Predicted", y="Actual", title=title)
        + theme_depictr(grid="none")
    )


def calibration_plot(y_true, y_score, n_bins=10, title=None):
    """Reliability (calibration) curve of predicted vs observed frequencies."""
    _require_sklearn()
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    df = pd.DataFrame({"predicted": prob_pred, "observed": prob_true})
    return (
        ggplot(df, aes("predicted", "observed"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="#9e9e9e")
        + geom_line(color=BRAND, size=0.9)
        + labs(x="Mean predicted probability", y="Observed frequency", title=title)
        + theme_depictr()
    )


def gain_plot(y_true, y_score, title=None):
    """Cumulative gains chart: positives captured as more of the ranked population is targeted."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))
    captured = np.cumsum(y_true[order]) / max(y_true.sum(), 1)
    population = np.arange(1, len(y_true) + 1) / len(y_true)
    df = pd.DataFrame({
        "population": np.concatenate([[0], population]),
        "captured": np.concatenate([[0], captured]),
    })
    return (
        ggplot(df, aes("population", "captured"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="#9e9e9e")
        + geom_line(color=BRAND, size=0.9)
        + labs(x="Population targeted", y="Positive cases captured", title=title)
        + theme_depictr()
    )
