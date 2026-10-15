"""Classification-metric plots.

The ROC and precision-recall metrics and the confusion matrix are computed by
``scikit-learn`` (the de-facto standard), and the figures are re-drawn under
the depictr theme, so a ROC curve sits in the same visual language as the rest
of a report. The cumulative gains, lift and calibration curves are computed
here with numpy, as the R twin computes them, and need no optional dependency.

Install the optional dependency with ``pip install "depictr[classification]"``.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
    geom_abline,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    geom_step,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_identity,
    scale_fill_gradientn,
    scale_size_area,
)

from ._checks import _check_conf_level
from .audit import _legible_text_colour
from .palette import BRAND, depictr_palette
from .theme import _scale_fills, scale_colour_depictr, theme_depictr


def _require_sklearn():
    try:
        import sklearn.metrics as m  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The classification plots need scikit-learn. Install it with "
            '`pip install "depictr[classification]"`.'
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

    The count is of values equal to 1, so the outcome is first read as 0/1 by
    :func:`_binary_outcome`.
    """
    y_true = np.asarray(y_true)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0 or n_pos == len(y_true):
        raise ValueError(message)
    return n_pos


def _binary_outcome(y_true) -> np.ndarray:
    """An outcome read as 0/1, as the R twin's ``as_binary()`` reads it.

    Booleans become 0 and 1. A categorical, the counterpart of R's factor,
    must have at most two categories, and its last declared category is the
    positive class, the convention ``as_binary()`` follows. Anything else must
    already be coded 0/1, as numbers or as their text, so a 1/2 or -1/1 coding
    and text such as ``"yes"``/``"no"`` are refused, as R refuses them. Each
    curve counted the values equal to 1, so a categorical had no positives and
    met the single-class refusal, and a 1/2 coding was read with 1 as the
    positive class. A missing value stays missing, for the caller to refuse.

    Returns
    -------
    numpy.ndarray
        Integers when no value is missing, and floats with NaN otherwise.
    """
    if isinstance(y_true, pd.Series):
        values = y_true.reset_index(drop=True)
    elif isinstance(y_true, pd.Categorical):
        values = pd.Series(y_true)
    else:
        values = pd.Series(np.asarray(y_true).ravel())
    missing = values.isna()
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = list(values.cat.categories)
        if len(categories) > 2:
            raise ValueError(
                f"The outcome categorical has {len(categories)} categories; a "
                "binary outcome must have exactly two. Categories: "
                f"{', '.join(str(c) for c in categories)}."
            )
        coded = ((values == categories[-1]) if categories
                 else pd.Series(False, index=values.index))
    elif pd.api.types.is_bool_dtype(values):
        coded = values
    else:
        coded = pd.to_numeric(values, errors="coerce")
        if not (coded.isin([0, 1]) | missing).all():
            raise ValueError(
                "`y_true` must hold outcomes coded 0/1, as booleans or as a "
                "categorical with two categories."
            )
    coded = coded.astype(float).mask(missing)
    return coded.to_numpy() if missing.any() else coded.astype(int).to_numpy()


def _outcome_and_score(y_true, y_score):
    """The outcome and the score as flat arrays, refusing a length mismatch.

    The curves pair each outcome with the score in the same position, so
    vectors of different lengths have no pairing to plot.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(y_true) != len(y_score):
        raise ValueError(
            "`y_true` and `y_score` must have the same length "
            f"({len(y_true)} and {len(y_score)})."
        )
    return y_true, y_score


def _gain_table(y_true, y_score):
    """Cumulative shares of the population targeted and of positives captured.

    The twin of the R package's internal ``gain_table()``. Cases are ranked by
    score, best first, and each run of tied scores is targeted all at once.
    The cumulative counts are read at the last case of a run, once the whole
    run is in, so no order of the tied rows can favour the positives or the
    negatives. A line drawn through these points is straight across a run,
    which is the share expected when the run's cases are reached in random
    order (Fawcett, 2006). The first point is the origin.

    Returns
    -------
    tuple of numpy.ndarray
        The population and captured shares, one point per distinct score after
        the origin.
    """
    y_true, y_score = _outcome_and_score(_binary_outcome(y_true), y_score)
    # A missing outcome would count as a negative, and each missing score would
    # sort last as a run of its own, in the order of its row. The curves that
    # scikit-learn computes and calibration_plot() refuse missing values too
    # (the R twin drops such cases with a message).
    for name, values in (("y_true", y_true), ("y_score", y_score)):
        if pd.isna(values).any():
            raise ValueError(
                f"`{name}` holds a missing value. Gains/lift need an outcome and "
                "a score for every case, so drop incomplete cases first."
            )
    n_pos = _positive_count(y_true,
                            "Gains/lift need both positive and negative outcomes.")
    order = np.argsort(-y_score, kind="stable")
    ranked = y_score[order]
    last_of_run = np.append(ranked[1:] != ranked[:-1], True)
    targeted = np.arange(1, len(ranked) + 1)[last_of_run]
    positives = np.cumsum(y_true[order] == 1)[last_of_run]
    population = np.concatenate([[0.0], targeted / len(ranked)])
    captured = np.concatenate([[0.0], positives / n_pos])
    return population, captured


def _wilson_interval(successes, n, conf_level):
    """Wilson score interval for a binomial proportion, as the R twin computes it.

    The interval stays within [0, 1]. In small samples its coverage stays close
    to the nominal level where the Wald interval's is erratic, though it dips
    for rates very close to 0 or 1 (Brown et al., 2001). The formula and its
    order of operations are those of the R twin's ``wilson_interval()``.
    """
    from scipy.stats import norm

    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    z = norm.ppf(1 - (1 - conf_level) / 2)
    phat = successes / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return np.maximum(0, centre - half), np.minimum(1, centre + half)


def _calibration_bins(y, score, n_bins, strategy):
    """Per-bin mean score, observed rate, positives and count, empty bins dropped.

    ``"quantile"`` cuts at the quantiles of the scores, where numpy's default
    method is R's ``quantile()`` type 7, and merges repeated edges, as the R
    twin does. ``"uniform"`` cuts [0, 1] into equal widths, the bins
    scikit-learn's ``calibration_curve()`` gave depictr 0.3.0.
    """
    if strategy == "quantile":
        edges = np.unique(np.quantile(score, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            raise ValueError("Too few distinct scores to form bins.")
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Searching the inner edges from the left puts a score equal to an edge in
    # the bin below it. Each bin then includes its upper edge and the lowest
    # its lower edge too, as R's cut(include.lowest = TRUE) assigns them.
    bin_id = np.searchsorted(edges[1:-1], score, side="left")
    k = len(edges) - 1
    n = np.bincount(bin_id, minlength=k)
    successes = np.bincount(bin_id, weights=y, minlength=k)
    score_sum = np.bincount(bin_id, weights=score, minlength=k)
    kept = n > 0
    return pd.DataFrame({
        "predicted": score_sum[kept] / n[kept],
        "observed": successes[kept] / n[kept],
        "successes": np.rint(successes[kept]).astype(int),
        "n": n[kept],
    })


def roc_curve_plot(y_true, y_score, title=None):
    """ROC curve with the area under the curve (AUC) annotated.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes, coded 0/1, as booleans or as a categorical with two
        categories, whose last category is the positive class, as in the R
        twin. Both classes must be present: with only one, the true or false
        positive rate has no denominator and the AUC is undefined.
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
    y_true = _binary_outcome(y_true)
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
        Binary outcomes, coded 0/1, as booleans or as a categorical with two
        categories, whose last category is the positive class, as in the R
        twin. Both classes must be present: with only one, precision or recall
        has no denominator and the average precision is undefined.
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
    y_true = _binary_outcome(y_true)
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

    Each count is drawn in black or white, whichever has the higher contrast
    against its cell.

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
    fill_scale = scale_fill_gradientn(
        colors=depictr_palette(7, kind="sequential"),
        name=("Proportion" if normalise else "Count"),
    )
    # Each count is black or white, whichever contrasts more with its tile as
    # the fill scale paints it. The busiest cell is the darkest, where the
    # near-black every count used to be drawn in sat at 1.3:1.
    long["label_colour"] = _legible_text_colour(
        _scale_fills(fill_scale, long["count"]))
    return (
        ggplot(long, aes("predicted", "true", fill="count"))
        + geom_tile(color="white")
        + geom_text(aes(label="label", color="label_colour"), size=10)
        + scale_color_identity()
        + fill_scale
        + labs(x="Predicted", y="Actual", title=title)
        + theme_depictr(grid="none")
    )


def calibration_plot(y_true, y_score, n_bins=10, title=None, conf_level=0.95,
                     strategy="quantile", bins=None):
    """Reliability (calibration) curve of predicted against observed frequencies.

    The cases are grouped into bins by their predicted probability, and each
    bin is drawn at its mean predicted probability and its observed event rate,
    with the dashed diagonal marking perfect calibration. By default the bins
    hold equal numbers of cases, as in the R twin, so each point rests on about
    the same evidence. Each point is sized by its number of cases and carries a
    Wilson interval for its observed rate (Wilson, 1927), so a bin backed by
    few cases is not over-read. Brown et al. (2001) recommend the Wilson
    interval for small samples. Its coverage stays close to the nominal level
    where the Wald interval's is erratic, though it dips for rates very close
    to 0 or 1.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes, coded 0/1, as booleans or as a categorical with two
        categories, whose last category is the positive class, as in the R
        twin. Any other value, a missing one included, raises ``ValueError``.
    y_score : array-like
        Predicted probabilities from a fitted model, on the probability scale.
        Unlike the ROC and gains charts, which only rank cases, a reliability
        curve compares the predicted probability with the observed frequency,
        so an arbitrary monotone score would misstate the calibration. A value
        outside [0, 1], or a missing one, raises ``ValueError``.
    n_bins : int
        Number of bins requested. Fewer points are drawn when bins merge or
        come out empty (see ``strategy``).
    title : str, optional
    conf_level : float or None
        Confidence level of the Wilson intervals, strictly between 0 and 1.
        ``None`` leaves the intervals out.
    strategy : {"quantile", "uniform"}
        How the bins are cut. ``"quantile"`` cuts at the quantiles of the
        scores, computed as R's ``quantile()`` computes them by default, into
        bins of nearly equal count. Each bin includes its upper edge and the
        lowest its lower edge too, as R's ``cut(include.lowest = TRUE)``
        assigns them. Tied scores can make edges coincide, and repeated edges
        are merged, so fewer than ``n_bins`` bins may remain. Fewer than two
        raise ``ValueError``. ``"uniform"`` cuts [0, 1] into ``n_bins`` bins
        of equal width, the bins of depictr 0.3.0, and drops the empty ones.
        A rare outcome whose scores never approach 1 then leaves fewer points
        than bins, the highest of them often resting on a handful of cases.
    bins : int, optional
        The R twin's name for ``n_bins``, which it replaces when given.

    Returns
    -------
    plotnine.ggplot
        Its data hold one row per bin, with the columns ``predicted``,
        ``observed``, ``successes`` and ``n`` and, unless ``conf_level`` is
        ``None``, ``conf_low`` and ``conf_high``.

    References
    ----------
    Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a
    binomial proportion. Statistical Science, 16(2), 101-133.
    https://doi.org/10.1214/ss/1009213286

    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. Journal of the American Statistical Association,
    22(158), 209-212. https://doi.org/10.1080/01621459.1927.10502953

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
    if bins is not None:
        if n_bins not in (10, bins):
            raise TypeError(
                "Give the number of bins as `n_bins` or as `bins`, not both.")
        n_bins = bins
    if (isinstance(n_bins, bool) or not isinstance(n_bins, numbers.Integral)
            or n_bins < 1):
        raise ValueError("`n_bins` must be a positive whole number.")
    if strategy not in ("quantile", "uniform"):
        raise ValueError('`strategy` must be "quantile" or "uniform".')
    _check_conf_level(conf_level, allow_none=True)

    y_true, y_score = _outcome_and_score(_binary_outcome(y_true), y_score)
    try:
        y = y_true.astype(float)
    except (TypeError, ValueError):
        y = None
    if y is None or not np.all((y == 0) | (y == 1)):
        raise ValueError("`y_true` must hold outcomes coded 0/1.")
    # A missing score fails this test too, since it is no probability at all.
    if not np.all((y_score >= 0) & (y_score <= 1)):
        raise ValueError("`y_score` must hold probabilities between 0 and 1.")
    if len(y_score) == 0:
        raise ValueError("Too few distinct scores to form bins.")

    df = _calibration_bins(y, y_score, n_bins, strategy)
    if conf_level is not None:
        df["conf_low"], df["conf_high"] = _wilson_interval(
            df["successes"], df["n"], conf_level)
    p = (
        ggplot(df, aes("predicted", "observed"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="#9e9e9e")
    )
    if conf_level is not None:
        # The R twin draws its caps 0.02 wide on a unit-square axis, a fiftieth
        # of its width. This axis spans the bins alone, so the caps take a
        # fiftieth of that span, which keeps crowded bins from overlapping.
        span = float(np.ptp(df["predicted"].to_numpy()))
        p = p + geom_errorbar(aes(ymin="conf_low", ymax="conf_high"),
                              width=0.02 * span if span > 0 else 0.02,
                              size=0.5, color=BRAND, alpha=0.7)
    return (
        p
        + geom_line(color=BRAND, size=0.7)
        + geom_point(aes(size="n"), color=BRAND, alpha=0.7)
        + scale_size_area(name="n", max_size=7)
        + labs(x="Mean predicted probability", y="Observed frequency", title=title)
        + theme_depictr()
    )


def gain_plot(y_true, y_score, title=None):
    """Cumulative gains chart: positives captured as more of the ranked population is targeted.

    The population is sorted by ``y_score``, best first, and the curve traces
    the share of all positive cases captured by each depth. The dashed diagonal
    is random targeting.

    Cases with tied scores are targeted together, as in the R twin. The curve
    takes a point only once every case of a run of equal scores is in, and it
    runs straight to that point, which is the share expected when tied cases
    are reached in random order (Fawcett, 2006). The curve therefore does not
    depend on the order of the rows. With every score distinct, each case is a
    point of its own.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes, coded 0/1, as booleans or as a categorical with two
        categories, whose last category is the positive class, as in the R
        twin. Both classes must be present: with no positives the share
        captured has no denominator, and with no negatives every depth captures
        everything. A missing outcome raises ``ValueError``.
    y_score : array-like
        Predicted scores or probabilities for the positive class, one per
        outcome. Only the ranking matters, so any monotone score works. A
        missing score, which has no rank, raises ``ValueError``.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    References
    ----------
    Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition
    Letters, 27(8), 861-874. https://doi.org/10.1016/j.patrec.2005.10.010

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> ct = dp.clinical_trial()
    >>> score = 1 / (1 + np.exp(-ct["biomarker"]))
    >>> p = dp.gain_plot(ct["adverse_event"], score)
    """
    population, captured = _gain_table(y_true, y_score)
    df = pd.DataFrame({"population": population, "captured": captured})
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

    Cases with tied scores are targeted together, as in
    :func:`depictr.gain_plot` and the R twin, so the curve has a point at the
    end of each run of equal scores and does not depend on the order of the
    rows. It starts at the first such point, since lift has no value at a depth
    of zero.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes, coded 0/1, as booleans or as a categorical with two
        categories, whose last category is the positive class, as in the R
        twin. Both classes must be present: lift is a ratio to the baseline
        positive rate, which is zero with no positives and one at every depth
        with no negatives. A missing outcome raises ``ValueError``.
    y_score : array-like
        Predicted scores or probabilities for the positive class, one per
        outcome. Only the ranking matters, so any monotone score works. A
        missing score, which has no rank, raises ``ValueError``.
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
    population, captured = _gain_table(y_true, y_score)
    # Drop the origin, where the ratio would be 0 / 0.
    df = pd.DataFrame({"population": population[1:],
                       "lift": captured[1:] / population[1:]})
    return (
        ggplot(df, aes("population", "lift"))
        + geom_hline(yintercept=1, linetype="dashed", color="#9e9e9e")
        + geom_line(color=BRAND, size=0.9)
        + labs(x="Population targeted", y="Cumulative lift", title=title)
        + theme_depictr()
    )


def _threshold_counts(y_true, y_score):
    """True and false positives at every distinct score, as a threshold.

    The twin of the R package's internal ``threshold_counts()``. Cases are
    ranked by score, best first, and the cumulative counts are read at the last
    case of each run of tied scores, where every case at or above that score is
    predicted positive.

    Returns
    -------
    tuple of numpy.ndarray
        The distinct scores in decreasing order, and the true and false
        positives with each one as the threshold.
    """
    order = np.argsort(-y_score, kind="stable")
    ranked = y_score[order]
    last_of_run = np.append(ranked[1:] != ranked[:-1], True)
    tp = np.cumsum(y_true[order] == 1)[last_of_run]
    fp = np.cumsum(y_true[order] == 0)[last_of_run]
    return ranked[last_of_run], tp, fp


def threshold_plot(y_true, y_score, title=None):
    """Sensitivity, specificity, precision and F1 across the decision threshold.

    Sweeps the probability cut-off and plots each metric, so the trade-off when
    choosing an operating point can be read straight off the curves.

    A case is predicted positive when its score is at or above the threshold.
    The metrics are evaluated at every distinct score, the points at which the
    confusion matrix can change, so the curves are exact step functions. Between
    two adjacent scores, each metric keeps its value at the upper one, so each
    step changes at a score and runs flat below it. Where no positive case is
    predicted positive, precision and recall are 0 and F1 is taken as 0. The R
    twin computes and draws the same values.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes, coded 0/1, as booleans or as a categorical with two
        categories, whose last category is the positive class, as in the R
        twin. Both classes must be present: with only one, sensitivity or
        specificity has no denominator at any threshold. A missing outcome
        raises ``ValueError``.
    y_score : array-like
        Predicted scores or probabilities for the positive class, one per
        outcome. A missing score raises ``ValueError``.
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
    y_true, y_score = _outcome_and_score(_binary_outcome(y_true), y_score)
    # A missing score would rank as a run of its own and a missing outcome
    # would count as neither class, so the cumulative counts need both.
    for name, values in (("y_true", y_true), ("y_score", y_score)):
        if pd.isna(values).any():
            raise ValueError(
                f"`{name}` holds a missing value. A threshold sweep needs an "
                "outcome and a score for every case, so drop incomplete cases "
                "first."
            )
    _positive_count(y_true,
                    "A threshold sweep needs both positive and negative outcomes.")
    # Every distinct score, as in the R twin. Past 200 distinct scores the
    # sweep used to move to 200 quantiles of the scores, which fall between
    # the points where the metrics change.
    thresholds, tp, fp = _threshold_counts(y_true, y_score)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    sens = tp / n_pos
    spec = (n_neg - fp) / n_neg
    # With outcomes coded 0/1, the highest score predicts at least one
    # positive, so precision is defined at every threshold.
    predicted = tp + fp
    prec = np.divide(tp, predicted, out=np.full(len(tp), np.nan),
                     where=predicted > 0)
    total = prec + sens
    f1 = np.divide(2 * prec * sens, total, out=np.zeros_like(total),
                   where=total > 0)
    wide = pd.DataFrame({"threshold": thresholds[::-1], "Sensitivity": sens[::-1],
                         "Specificity": spec[::-1], "Precision": prec[::-1],
                         "F1": f1[::-1]})
    long = wide.melt(id_vars="threshold", var_name="metric", value_name="value")
    # A line would join adjacent thresholds with slopes through values that no
    # threshold gives. Every threshold between two adjacent scores predicts what
    # the upper score predicts. geom_step() orders the points by threshold, and
    # direction "vh" changes at each score and runs flat below it.
    return (
        ggplot(long, aes("threshold", "value", color="metric"))
        + geom_step(direction="vh", size=0.8, na_rm=True)
        + scale_colour_depictr()
        + labs(x="Decision threshold", y="Metric value", color="", title=title)
        + theme_depictr()
    )
