"""Estimation plots that put the effect size, not a p-value, at the centre.

A Cumming-style estimation plot (Cumming, 2012) compares every non-reference
group with a reference group and shows the mean difference with a bootstrap
confidence interval, so the reader sees the size of the effect and its
uncertainty directly. The standardised effect size (Hedges' g by default, or
Cohen's d) is annotated beside each difference, which is what separates an
estimation plot from a plain mean-difference plot. The bootstrap and the effect
size are computed in numpy here; there is no DABEST dependency.

Cumming, G. (2012). Understanding the new statistics: Effect sizes, confidence
intervals, and meta-analysis. Routledge.

Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size
and related estimators. Journal of Educational Statistics, 6(2), 107-128.
https://doi.org/10.3102/10769986006002107
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    expand_limits,
    geom_errorbar,
    geom_hline,
    geom_jitter,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_x_discrete,
    scale_y_continuous,
    theme,
)

from .compose import arrange_plots
from .palette import depictr_brand
from .theme import theme_depictr


def _boot_diff_ci(treat, ref, conf_level, n_boot, rng):
    """Bootstrap percentile interval for the difference in means (treat - ref).

    Each group is resampled with replacement ``n_boot`` times and the
    ``conf_level`` percentile interval is read off the resampled differences.
    Returns ``(nan, nan)`` when either group has fewer than two observations,
    since there is then no spread to resample.
    """
    nt, nr = len(treat), len(ref)
    if nt < 2 or nr < 2:
        return np.nan, np.nan
    # Vectorise the resampling: one (n_boot x n) index matrix per group.
    treat_means = treat[rng.integers(0, nt, size=(n_boot, nt))].mean(axis=1)
    ref_means = ref[rng.integers(0, nr, size=(n_boot, nr))].mean(axis=1)
    reps = treat_means - ref_means
    alpha = 1 - conf_level
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _effsize_diff(treat, ref):
    """Cohen's d and Hedges' g for the difference (treat - ref).

    Uses the pooled standard deviation (the classic two-sample Cohen's d).
    Hedges' g multiplies d by the small-sample correction
    ``J = 1 - 3 / (4 (n1 + n2) - 9)``. Returns ``nan`` for both when the pooled
    SD is undefined or zero.
    """
    n1, n2 = len(treat), len(ref)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    s1, s2 = np.var(treat, ddof=1), np.var(ref, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if not np.isfinite(sp) or sp == 0:
        return np.nan, np.nan
    d = (treat.mean() - ref.mean()) / sp
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d), float(j * d)


def estimation_plot(data, y, group, reference=None, conf_level=0.95,
                    n_boot=2000, effsize="hedges_g", two_panel=False,
                    title=None, seed=None):
    """Cumming-style estimation plot of mean differences.

    For each non-reference group, the mean difference from the reference group is
    drawn as a point with a bootstrap confidence interval. A dashed line marks a
    difference of zero (the reference), and the standardised effect size is
    annotated beside each point. With ``two_panel=True`` this difference axis sits
    beneath a panel of the raw data and group means (the Gardner-Altman layout).

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    y : str
        Name of the numeric outcome column.
    group : str
        Name of the grouping column.
    reference : str, optional
        The reference (control) group the others are compared with. Defaults to
        the first group level.
    conf_level : float
        Confidence level for the bootstrap difference intervals.
    n_boot : int
        Number of bootstrap resamples for each difference interval.
    effsize : {"hedges_g", "cohens_d", "none"}
        Standardised effect size annotated beside each difference. Hedges' g is
        the small-sample corrected default; pass ``"none"`` to omit it.
    two_panel : bool
        When ``True``, place the difference axis beneath a panel of the raw data
        and group means (the Gardner-Altman layout), returning a composition.
    title : str, optional
        Plot title.
    seed : int, optional
        Seed for the bootstrap, for reproducible intervals.

    Returns
    -------
    plotnine.ggplot or plotnine.composition.Compose
        A single panel by default, or a two-panel composition when
        ``two_panel=True``. Either carries ``.differences``, a DataFrame of the
        computed mean differences, their bootstrap intervals, and effect sizes.

    Examples
    --------
    >>> import depictr as dp
    >>> cy = dp.crop_yield()
    >>> p = dp.estimation_plot(cy, "yield", "treatment", n_boot=200, seed=1)
    """
    if y not in data.columns:
        raise KeyError(f"{y!r} is not a column of `data`.")
    if group not in data.columns:
        raise KeyError(f"{group!r} is not a column of `data`.")
    if not pd.api.types.is_numeric_dtype(data[y]):
        raise TypeError("`y` must be numeric.")
    if effsize not in {"hedges_g", "cohens_d", "none"}:
        raise ValueError("`effsize` must be 'hedges_g', 'cohens_d' or 'none'.")
    if n_boot < 1:
        raise ValueError("`n_boot` must be a positive integer.")

    d = data[[y, group]].dropna()
    levels = list(pd.unique(d[group]))
    if len(levels) < 2:
        raise ValueError("`group` must have at least two non-empty levels.")

    ref = reference if reference is not None else levels[0]
    if ref not in levels:
        raise ValueError(f"`reference` ({ref!r}) is not a level of `group`.")
    others = [g for g in levels if g != ref]

    rng = np.random.default_rng(seed)
    ref_vals = d.loc[d[group] == ref, y].to_numpy(dtype=float)

    rows = []
    for g in others:
        gv = d.loc[d[group] == g, y].to_numpy(dtype=float)
        md = gv.mean() - ref_vals.mean()
        lo, hi = _boot_diff_ci(gv, ref_vals, conf_level, n_boot, rng)
        cohens_d, hedges_g = _effsize_diff(gv, ref_vals)
        rows.append({"group": g, "reference": ref, "diff": md,
                     "lower": lo, "upper": hi,
                     "cohens_d": cohens_d, "hedges_g": hedges_g})
    diffs = pd.DataFrame(rows)
    # Keep the non-reference groups in their original order up the axis.
    diffs["group"] = pd.Categorical(diffs["group"], categories=others, ordered=True)

    brand = depictr_brand()
    p = (
        ggplot(diffs, aes(x="group", y="diff"))
        + geom_hline(yintercept=0, linetype="dashed", color="#9e9e9e", size=0.4)
        + geom_errorbar(aes(ymin="lower", ymax="upper"), width=0.12,
                        color=brand, size=0.8, na_rm=True)
        + geom_point(color=brand, size=2.8)
    )

    if effsize != "none":
        prefix = "Hedges' g = " if effsize == "hedges_g" else "Cohen's d = "
        lab = diffs[np.isfinite(diffs[effsize])].copy()
        if len(lab):
            lab["es_label"] = prefix + lab[effsize].map(lambda v: f"{v:.2f}")
            # Sit the label above the upper cap (or the point when no interval).
            lab["lab_y"] = np.where(np.isfinite(lab["upper"]),
                                    lab["upper"], lab["diff"])
            p = (p
                 + geom_text(aes(x="group", y="lab_y", label="es_label"),
                             data=lab, va="bottom", nudge_y=0.0,
                             color="#404040", size=9)
                 # Reserve headroom so the annotation is not clipped.
                 + scale_y_continuous(expand=(0.08, 0, 0.3, 0)))

    p = (p
         # Pad an invisible slot for the reference so the axis reads naturally.
         + scale_x_discrete(limits=others)
         + expand_limits(x=levels)
         + labs(x="", y=f"Mean difference\n(vs. {ref})",
                title=None if two_panel else title)
         + theme_depictr(grid="y")
         + theme(axis_text_x=element_text(weight="bold")))
    p.differences = diffs
    if not two_panel:
        return p

    # Top panel: the raw data with each group's mean and a t-based interval, on
    # the outcome scale, above the aligned difference axis.
    from scipy import stats

    summ = []
    for g in levels:
        v = d.loc[d[group] == g, y].to_numpy(dtype=float)
        n = len(v)
        m = float(v.mean())
        if n > 1:
            se = float(v.std(ddof=1)) / np.sqrt(n)
            tc = float(stats.t.ppf(1 - (1 - conf_level) / 2, n - 1))
        else:
            se = tc = 0.0
        summ.append({"group": g, "mean": m, "lo": m - tc * se, "hi": m + tc * se})
    summ_df = pd.DataFrame(summ)
    summ_df["group"] = pd.Categorical(summ_df["group"], categories=levels, ordered=True)

    top = (
        ggplot(summ_df, aes(x="group", y="mean"))
        + geom_jitter(aes(x=group, y=y), data=d, width=0.12, alpha=0.25,
                      color=brand, size=0.9, inherit_aes=False)
        + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.12, color=brand, size=0.8)
        + geom_point(color=brand, size=2.8)
        # The title rides on the top panel so it reads as the figure title
        # (plotnine compositions have no super-title).
        + labs(x="", y=y, title=title)
        + theme_depictr(grid="y")
        + theme(axis_text_x=element_text(weight="bold"))
    )
    composed = arrange_plots(top, p, ncol=1)
    composed.differences = diffs
    return composed
