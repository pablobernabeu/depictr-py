"""Kaplan-Meier survival plots.

The estimate and the log-rank test come from ``lifelines``; the figure is drawn
under the depictr theme as a single call -- the survminer/ggsurvfit pattern that
Python otherwise leaves to manual assembly. The number-at-risk counts are
computed and returned on the plot (``plot.at_risk``) for composing a table; a
built-in table panel is planned.

Install the optional dependency with ``pip install depictr[survival]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    geom_step,
    geom_text,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)

from .compose import arrange_plots
from .theme import scale_colour_depictr, theme_depictr


def _require_lifelines():
    try:
        import lifelines  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "survival_plot needs lifelines. Install it with "
            "`pip install depictr[survival]`."
        ) from exc
    return lifelines


def survival_plot(time, event, group=None, conf_level=0.95, risk_table=False,
                  title=None, x_lab="Time", y_lab="Survival probability"):
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
        Confidence level (reserved for the confidence band; the step curve is
        drawn now, the band is planned).
    title : str, optional
    x_lab, y_lab : str
        Axis labels.

    Returns
    -------
    plotnine.ggplot
        The plot carries ``.at_risk`` (a DataFrame of number-at-risk counts) and,
        when grouped, ``.logrank_p`` and ``.logrank_stat``.
    """
    lifelines = _require_lifelines()
    from lifelines import KaplanMeierFitter

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    groups = (np.asarray(group) if group is not None
              else np.repeat("all", len(time)))
    levels = list(pd.unique(groups))

    curves, at_risk_rows = [], []
    breaks = np.linspace(0, float(np.max(time)), 6)
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
        p_txt = "< 0.0001" if logrank_p < 1e-4 else f"= {logrank_p:.4f}"
        subtitle = f"Log-rank χ²({len(levels) - 1}) = {logrank_stat:.1f}, p {p_txt}"

    at_risk_df = pd.DataFrame(at_risk_rows)
    tmax = float(np.max(time))
    p = (p
         + scale_y_continuous(limits=(0, 1))
         + scale_x_continuous(limits=(0, tmax))
         + labs(x=x_lab, y=y_lab, title=title, subtitle=subtitle)
         + theme_depictr())
    p.at_risk = at_risk_df
    p.logrank_p, p.logrank_stat = logrank_p, logrank_stat
    if not risk_table:
        return p

    # Compose a number-at-risk table sharing the time axis beneath the curves.
    tbl_df = at_risk_df.copy()
    tbl_df["group"] = pd.Categorical(
        tbl_df["group"], categories=[str(lvl) for lvl in levels][::-1], ordered=True)
    tbl = (
        ggplot(tbl_df, aes(x="time", y="group"))
        + geom_text(aes(label="n_at_risk"), size=8, color="#1a1a1a")
        + scale_x_continuous(limits=(0, tmax))
        + labs(x=x_lab, y=None, title="Number at risk")
        + theme_depictr(grid="none")
        + theme(panel_grid_major=element_blank())
    )
    composed = arrange_plots(p, tbl, ncol=1)
    composed.at_risk = at_risk_df
    composed.logrank_p, composed.logrank_stat = logrank_p, logrank_stat
    return composed
