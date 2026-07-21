"""Kaplan-Meier survival plots.

The estimate and the log-rank test come from ``lifelines``; the figure is drawn
under the depictr theme as a single call -- the survminer/ggsurvfit pattern that
Python otherwise leaves to manual assembly. ``risk_table=True`` draws a
number-at-risk table beneath the curves, and the counts are also returned on
the plot (``plot.at_risk``) for composing a table of your own.

Install the optional dependency with ``pip install depictr[survival]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
    geom_hline,
    geom_step,
    geom_text,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
)

from .theme import legend_inside as _legend_inside
from .theme import scale_colour_depictr, theme_depictr


def _nice_breaks(tmax, n=5):
    """Round, evenly spaced axis breaks from 0 to about ``tmax``.

    Picks a 1/2/2.5/5/10 step so the time axis reads in round numbers rather
    than the arbitrary values an even split would give.
    """
    if tmax <= 0:
        return np.array([0.0])
    raw = tmax / n
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    return np.arange(0, tmax + step * 0.5, step)


def _require_lifelines():
    try:
        import lifelines  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "survival_plot needs lifelines. Install it with "
            "`pip install depictr[survival]`."
        ) from exc
    return lifelines


def _apa_p(p):
    """Format a p-value in APA style: no leading zero, floored at ``.001``."""
    if p < 0.001:
        return "< .001"
    s = f"{p:.3f}"
    return "= " + (s[1:] if s.startswith("0") else s)


def survival_plot(time, event, group=None, conf_level=0.95, risk_table=False,
                  legend_inside=False, title=None, x_lab="Time",
                  y_lab="Survival probability"):
    """Kaplan-Meier survival curves, optionally by group, with a log-rank test.

    Parameters
    ----------
    time : array-like
        Follow-up times.
    event : array-like
        Event indicator (1 = event, 0 = censored).
    group : array-like, optional
        Group label per observation; one curve per group, plus a log-rank test
        of the difference.
    conf_level : float
        Accepted for future use; the Python twin does not yet draw the
        confidence band or censor marks the R package draws.
    risk_table : bool
        Add a number-at-risk table as a thin strip beneath the curves.
    legend_inside : bool
        When ``True`` (and there are groups), place the group legend inside the
        panel rather than in a right-hand margin: the bottom-left for a plain
        curve, or the top-right when a risk table occupies the bottom strip.
        Either corner is one a descending survival curve leaves empty.
    title : str, optional
    x_lab, y_lab : str
        Axis labels.

    Returns
    -------
    plotnine.ggplot
        The plot carries ``.at_risk`` (a DataFrame of number-at-risk counts) and,
        when grouped, ``.logrank_p`` and ``.logrank_stat``.

    Examples
    --------
    >>> import depictr as dp
    >>> ct = dp.clinical_trial()
    >>> p = dp.survival_plot(ct["time"], ct["event"], group=ct["arm"])
    """
    _require_lifelines()
    from lifelines import KaplanMeierFitter

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    groups = (np.asarray(group) if group is not None
              else np.repeat("all", len(time)))
    # One factor order for the whole figure -- a user-set categorical order if
    # there is one, otherwise first appearance -- so the colour legend and the
    # risk-table rows list the groups identically.
    if isinstance(getattr(group, "dtype", None), pd.CategoricalDtype):
        seen = set(groups.tolist())
        levels = [c for c in group.cat.categories if c in seen]
    else:
        levels = list(pd.unique(groups))

    curves, at_risk_rows = [], []
    breaks = _nice_breaks(float(np.max(time)))
    for lvl in levels:
        mask = groups == lvl
        kmf = KaplanMeierFitter()
        kmf.fit(time[mask], event[mask], label=str(lvl))
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["time", "surv"]
        sf["group"] = str(lvl)
        curves.append(sf)
        for b in breaks:
            at_risk_rows.append({"group": str(lvl), "time": round(float(b), 1),
                                 "n_at_risk": int(np.sum(time[mask] >= b))})
    curve = pd.concat(curves, ignore_index=True)
    curve["group"] = pd.Categorical(curve["group"],
                                    categories=[str(lvl) for lvl in levels],
                                    ordered=True)

    multi = len(levels) > 1
    if multi:
        mapping = aes("time", "surv", color="group")
        p = ggplot(curve, mapping) + geom_step(size=0.9) + scale_colour_depictr()
    else:
        from .palette import BRAND
        p = ggplot(curve, aes("time", "surv")) + geom_step(size=0.9, color=BRAND)

    subtitle = None
    logrank_p = logrank_stat = None
    if multi:
        from lifelines.statistics import multivariate_logrank_test
        res = multivariate_logrank_test(time, groups, event)
        logrank_p, logrank_stat = res.p_value, res.test_statistic
        subtitle = (f"Log-rank χ²({len(levels) - 1}) = {logrank_stat:.1f}, "
                    f"$p$ {_apa_p(logrank_p)}")

    at_risk_df = pd.DataFrame(at_risk_rows)
    tmax = float(np.max(time))

    if not risk_table:
        p = (p
             + scale_y_continuous(limits=(0, 1))
             + scale_x_continuous(limits=(0, tmax))
             + labs(x=x_lab, y=y_lab, title=title, subtitle=subtitle)
             + theme_depictr())
        # A monotone-decreasing curve leaves the bottom-left empty (early times
        # are still near survival 1), so the legend can sit there.
        if legend_inside and multi:
            p = p + _legend_inside("bottom left")
        p.at_risk = at_risk_df
        p.logrank_p, p.logrank_stat = logrank_p, logrank_stat
        return p

    # Number-at-risk table as a thin strip below the y = 0 axis, in the same
    # panel as the curves so it shares the time axis. The group names label the
    # rows on the y-axis -- so the curves keep the full width with no left-hand
    # gutter -- and the counts are coloured to match the curves.
    order = [str(lvl) for lvl in levels]
    row_h, header = 0.05, 0.05
    y_of = {g: -(header + row_h * (i + 0.5)) for i, g in enumerate(order)}
    tbl = at_risk_df.copy()
    tbl["group"] = tbl["group"].astype(str)
    tbl["y"] = tbl["group"].map(y_of)
    ymin = -(header + row_h * len(order) + 0.02)
    breaks = [b for b in np.unique(at_risk_df["time"]) if b >= 0]
    # Survival-axis ticks plus one tick per table row, labelled with the group.
    surv_breaks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_breaks = surv_breaks + [y_of[g] for g in order]
    y_labels = [f"{b:.2f}" for b in surv_breaks] + list(order)

    p = (
        p
        + geom_hline(yintercept=0, color="#cccccc", size=0.4)
        + geom_text(aes(x="time", y="y", label="n_at_risk", color="group"),
                    data=tbl, size=8, show_legend=False, inherit_aes=False)
        + annotate("text", x=0, y=-header * 0.5, label="Number at risk",
                   ha="left", fontweight="bold", color="#1a1a1a", size=9)
    )
    if not multi:
        # The single curve above has no colour aesthetic (a plain BRAND line),
        # but the risk-table text always maps color="group", so it still needs
        # a scale. When multi, the curve already added one at the top.
        p = p + scale_colour_depictr()
    p = (
        p
        + scale_y_continuous(breaks=y_breaks, labels=y_labels, limits=(ymin, 1.0))
        + scale_x_continuous(limits=(0, tmax), breaks=breaks)
        + labs(x=x_lab, y=y_lab, title=title, subtitle=subtitle)
        + theme_depictr(grid="y")
    )
    # Descending KM curves leave the panel's top-right empty and the risk table
    # owns the bottom strip, so an inside legend there uses the full width
    # rather than stranding it in a right-hand margin.
    if legend_inside and multi:
        p = p + _legend_inside("top right")
    p.at_risk = at_risk_df
    p.logrank_p, p.logrank_stat = logrank_p, logrank_stat
    return p
