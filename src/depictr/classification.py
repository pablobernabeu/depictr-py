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
    geom_hline,
    geom_line,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradientn,
)

from .palette import BRAND, depictr_palette
from .theme import scale_colour_depictr, theme_depictr


def _require_sklearn():
    try:
        import sklearn.metrics as m  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The classification plots need scikit-learn. Install it with "
            "`pip install depictr[classification]`."
        ) from exc
    return m


def _positive_count(y_true, message) -> int:
    """Number of positive cases, refusing a single-class outcome.

    None of the curves in this module survives a single class. The gains and
    lift curves divide by this count, and substituting 1 for a zero denominator
    draws a curve flat along the axis: a reader sees a catastrophically bad
    model where the truth is that the quantity does not exist. An all-positive
    outcome is as meaningless the other way, every depth capturing everything.
    scikit-learn does not refuse a single-class input: it computes an AUC of
    nan and an average precision of zero under a warning, numbers the
    annotations would then print as if they measured something. The R twin
    words the error per curve, so each caller passes its own ``message``.

    The count is of values equal to 1, so booleans work and anything else does
    not. A two-level outcome coded some other way, ``"yes"``/``"no"`` say, has
    no positives by this test and meets the single-class refusal, whose wording
    then describes the wrong problem. The R twin coerces a logical or a
    two-level factor to 0/1 before counting, and this one has no equivalent, so
    the documented contract on every caller is 0/1.
    """
    y_true = np.asarray(y_true)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0 or n_pos == len(y_true):
        raise ValueError(message)
    return n_pos


def roc_curve_plot(y_true, y_score, title=None):
    """ROC curve with the area under the curve (AUC) annotated.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0/1). Both classes must be present: with only one,
        the true or false positive rate has no denominator and the AUC is
        undefined.
    y_score : array-like
        Predicted scores or probabilities for the positive class.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> ct = dp.clinical_trial()
    >>> score = 1 / (1 + np.exp(-ct["biomarker"]))
    >>> p = dp.roc_curve_plot(ct["adverse_event"], score)
    """
    m = _require_sklearn()
    _positive_count(y_true, "ROC needs both positive and negative outcomes.")
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
    """Precision-recall curve with the average precision (AP) annotated.

    The dashed horizontal line is the positive rate, the precision a random
    classifier achieves, which is the baseline a precision-recall curve is read
    against (unlike a ROC curve, whose baseline is fixed at the diagonal).

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0/1). Both classes must be present: with only one,
        precision or recall has no denominator and the average precision is
        undefined.
    y_score : array-like
        Predicted scores or probabilities for the positive class.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> ct = dp.clinical_trial()
    >>> score = 1 / (1 + np.exp(-ct["biomarker"]))
    >>> p = dp.pr_curve_plot(ct["adverse_event"], score)
    """
    m = _require_sklearn()
    _positive_count(y_true,
                    "Precision-recall needs both positive and negative outcomes.")
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

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> ct = dp.clinical_trial()
    >>> score = 1 / (1 + np.exp(-ct["biomarker"]))
    >>> p = dp.confusion_matrix_plot(ct["adverse_event"], (score > 0.6).astype(int))
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
    """Reliability (calibration) curve of predicted vs observed frequencies.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0/1).
    y_score : array-like
        Predicted probabilities from a fitted model, on the probability scale.
        Unlike the ROC and gains charts, which only rank cases, a reliability
        curve compares the predicted probability with the observed frequency,
        so an arbitrary monotone score would misstate the calibration.
    n_bins : int
        Number of equal-width bins spanning 0 to 1. Empty bins are dropped, so
        a rare outcome whose scores never approach 1 leaves fewer points than
        bins requested.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> from sklearn.linear_model import LogisticRegression
    >>> ct = dp.clinical_trial()
    >>> X = ct[["biomarker", "age"]]
    >>> fit = LogisticRegression().fit(X, ct["adverse_event"])
    >>> p = dp.calibration_plot(ct["adverse_event"],
    ...                         fit.predict_proba(X)[:, 1], n_bins=5)
    """
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
    """Cumulative gains chart: positives captured as more of the ranked population is targeted.

    The population is sorted by ``y_score``, best first, and the curve traces
    the share of all positive cases captured by each depth. The dashed diagonal
    is random targeting.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0/1). Both classes must be present: with no positives
        the share captured has no denominator, and with no negatives every depth
        captures everything.
    y_score : array-like
        Predicted scores or probabilities for the positive class. Only the
        ranking matters, so any monotone score works.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> ct = dp.clinical_trial()
    >>> score = 1 / (1 + np.exp(-ct["biomarker"]))
    >>> p = dp.gain_plot(ct["adverse_event"], score)
    """
    y_true = np.asarray(y_true)
    n_pos = _positive_count(y_true,
                            "Gains/lift need both positive and negative outcomes.")
    order = np.argsort(-np.asarray(y_score))
    captured = np.cumsum(y_true[order]) / n_pos
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


def lift_plot(y_true, y_score, title=None):
    """Cumulative lift chart.

    Lift is the share of positives captured at a given depth of the
    score-ordered population, divided by that depth. A lift of 3 in the top 10%
    means that decile holds three times the baseline positive rate. The dashed
    line at 1 is random targeting.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0/1). Both classes must be present: lift is a ratio to
        the baseline positive rate, which is zero with no positives and one at
        every depth with no negatives.
    y_score : array-like
        Predicted scores or probabilities for the positive class. Only the
        ranking matters, so any monotone score works.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> ct = dp.clinical_trial()
    >>> score = 1 / (1 + np.exp(-ct["biomarker"]))
    >>> p = dp.lift_plot(ct["adverse_event"], score)
    """
    y_true = np.asarray(y_true)
    n_pos = _positive_count(y_true,
                            "Gains/lift need both positive and negative outcomes.")
    order = np.argsort(-np.asarray(y_score))
    population = np.arange(1, len(y_true) + 1) / len(y_true)
    captured = np.cumsum(y_true[order]) / n_pos
    df = pd.DataFrame({"population": population, "lift": captured / population})
    return (
        ggplot(df, aes("population", "lift"))
        + geom_hline(yintercept=1, linetype="dashed", color="#9e9e9e")
        + geom_line(color=BRAND, size=0.9)
        + labs(x="Population targeted", y="Cumulative lift", title=title)
        + theme_depictr()
    )


def threshold_plot(y_true, y_score, title=None):
    """Sensitivity, specificity, precision and F1 across the decision threshold.

    Sweeps the probability cut-off and plots each metric, so the trade-off when
    choosing an operating point can be read straight off the curves.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0/1). Both classes must be present: with only one,
        sensitivity or specificity has no denominator at any threshold.
    y_score : array-like
        Predicted scores or probabilities for the positive class.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> ct = dp.clinical_trial()
    >>> score = 1 / (1 + np.exp(-ct["biomarker"]))
    >>> p = dp.threshold_plot(ct["adverse_event"], score)
    """
    _require_sklearn()
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    _positive_count(y_true,
                    "A threshold sweep needs both positive and negative outcomes.")
    thresholds = np.unique(y_score)
    if len(thresholds) > 200:  # keep the sweep cheap on large score sets
        thresholds = np.quantile(y_score, np.linspace(0, 1, 200))
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    rows = []
    for t in thresholds:
        pred = y_score >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        sens = tp / n_pos if n_pos else np.nan
        spec = (n_neg - fp) / n_neg if n_neg else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        denom = (prec + sens) if (prec + sens) > 0 else np.nan
        f1 = 2 * prec * sens / denom
        rows.append({"threshold": float(t), "Sensitivity": sens,
                     "Specificity": spec, "Precision": prec, "F1": f1})
    long = (pd.DataFrame(rows)
            .melt(id_vars="threshold", var_name="metric", value_name="value"))
    return (
        ggplot(long, aes("threshold", "value", color="metric"))
        + geom_line(size=0.8, na_rm=True)
        + scale_colour_depictr()
        + labs(x="Decision threshold", y="Metric value", color="", title=title)
        + theme_depictr()
    )
