"""Kaplan-Meier survival plots.

The estimate and the log-rank test come from ``lifelines``; the figure is drawn
under the depictr theme as a single call -- the survminer/ggsurvfit pattern that
Python otherwise leaves to manual assembly. The confidence limits and the
censor marks are drawn as the R twin draws them, and the limits are computed
with its formulas. ``risk_table=True`` draws a number-at-risk table beneath the
curves, and the counts are also returned on the plot (``plot.at_risk``) for
composing a table of your own.

Install the optional dependency with ``pip install depictr[survival]``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
    geom_hline,
    geom_point,
    geom_step,
    geom_text,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
)

from .theme import legend_inside as _legend_inside
from .theme import scale_colour_depictr, theme_depictr

_CODING_MESSAGE = (
    "`event` must use the 0/1 (0 = censored) or 1/2 (1 = censored) coding."
)
_CONF_TYPES = ("log-log", "log", "plain")


def _km_limits(surv, se_log, conf_level, conf_type):
    """Pointwise confidence limits for a survival curve, as the R twin draws them.

    ``se_log`` is the standard error of log S, the square root of Greenwood's
    sum. The formulas are those of R's ``survival::survfit()``. On the log-log
    scale, the limits are ``S ** exp(-z se / log S)`` and
    ``S ** exp(z se / log S)``, the interval lifelines computes. On the log
    scale, they are ``S exp(-z se)`` and ``S exp(z se)``, with the upper limit
    capped at 1, and untransformed they are ``S - z S se`` and ``S + z S se``,
    kept between 0 and 1. Where S is 1, the log-log formula is 0/0 and the
    limits are 1, as survfit gives. Where S is 0, every formula breaks down,
    and the limits keep the previous row's values, as in the R twin. The
    lifelines interval is 0 at both ends there, and survfit gives missing
    values, which would end a band drawn as steps at the previous event.
    """
    # Imported here, as elsewhere in the package, so that importing depictr
    # does not load scipy.stats.
    from scipy.stats import norm

    surv = np.asarray(surv, dtype=float)
    se_log = np.asarray(se_log, dtype=float)
    z = norm.ppf(1 - (1 - conf_level) / 2)
    lower = np.full(surv.shape, np.nan)
    upper = np.full(surv.shape, np.nan)
    inside = (surv > 0) & (surv < 1) & ~np.isnan(se_log)
    s = surv[inside]
    w = z * se_log[inside]
    if conf_type == "log-log":
        lower[inside] = s ** np.exp(-w / np.log(s))
        upper[inside] = s ** np.exp(w / np.log(s))
    elif conf_type == "log":
        lower[inside] = s * np.exp(-w)
        upper[inside] = np.minimum(1.0, s * np.exp(w))
    else:
        lower[inside] = np.maximum(0.0, s - w * s)
        upper[inside] = np.minimum(1.0, s + w * s)
    top = surv >= 1
    lower[top] = 1.0
    upper[top] = 1.0
    return (pd.Series(lower).ffill().fillna(1.0).to_numpy(),
            pd.Series(upper).ffill().fillna(1.0).to_numpy())


def _greenwood_se(table):
    """The standard error of log S at each row of a lifelines event table.

    The square root of Greenwood's sum of d / (n (n - d)) over the event times
    so far. It is infinite from a step to zero on, where every observation
    still at risk has the event.
    """
    d = table["observed"].to_numpy(dtype=float)
    n = table["at_risk"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.cumsum(d / (n * (n - d))))


def _nice_breaks(tmax, n=5):
    """Round, evenly spaced axis breaks from 0 to at most ``tmax``.

    Picks a 1/2/2.5/5/10 step so the time axis reads in round numbers rather
    than the arbitrary values an even split would give.
    """
    # Not finite: the step search below compares `raw <= m * mag`, and every
    # comparison against a NaN is False, so the generator would run dry and
    # raise a bare StopIteration naming nothing.
    if not np.isfinite(tmax) or tmax <= 0:
        return np.array([0.0])
    raw = tmax / n
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    breaks = np.arange(0, tmax + step * 0.5, step)
    # arange accumulates floating-point error, so a 0.1 step reaches
    # 0.30000000000000004. A break a hair above its round value would leave out
    # a subject followed up to exactly that time, so the breaks are rounded to
    # the step's own decimals.
    breaks = np.round(breaks, max(0, 1 - int(np.floor(np.log10(step)))))
    # A break past the last follow-up time is a time nobody reached. It would
    # add a column of zeros to the at-risk table, outside the x scale, and
    # plotnine would drop it with a warning about missing values.
    return breaks[breaks <= tmax]


def _event_values(event):
    """The event indicator as floats, with NaN wherever a value is missing.

    Going through pandas also reads the missing value of its nullable integer
    and boolean columns, which numpy cannot convert on its own.
    """
    try:
        return pd.Series(event).to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "`event` must be numeric (0/1 or 1/2) or boolean; recode a "
            "categorical or string indicator first."
        ) from exc


def _normalise_event(event):
    """Read a complete event indicator as 0/1 integers, with 1 = event.

    The two codings are the R twin's: 0/1 (0 = censored, 1 = event) and the
    ``survival::Surv()`` 1/2 coding (1 = censored, 2 = event). Booleans arrive
    as 0/1. The codes are judged over every observation at once, so a group
    censored throughout keeps its meaning under the 1/2 coding. Any other set
    of codes raises. Given 1/2 codes, lifelines would count each censoring as
    a death, since it takes every non-zero value for an event, and a cast to
    int would read 0.5 as a censoring.
    """
    codes = set(np.unique(event).tolist())
    if codes <= {0, 1}:
        return event.astype(int)
    if codes <= {1, 2}:
        return event.astype(int) - 1
    raise ValueError(_CODING_MESSAGE)


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
                  y_lab="Survival probability", censor_marks=True, logrank=None,
                  conf_type="log-log"):
    """Kaplan-Meier survival curves, optionally by group, with confidence limits.

    Parameters
    ----------
    time : array-like
        Follow-up times. Must be finite: a missing or infinite time raises,
        since dropping or imputing it is the analyst's decision.
    event : array-like
        Event indicator, coded 0/1 (0 = censored, 1 = event) or with the
        ``survival::Surv()`` 1/2 coding (1 = censored, 2 = event), the two
        codings the R twin accepts. Booleans read as 0/1. The coding is judged
        over all observations together, and any other coding raises, a code
        that is not a whole number among them. An observation whose event is
        missing records neither an event nor a censoring, so it is dropped
        with a warning.
    group : array-like, optional
        Group label per observation; one curve per group, plus a log-rank test
        of the difference. Observations with a missing group are dropped with a
        warning, since they belong to no arm.
    conf_level : float or None
        Confidence level of the pointwise limits drawn as dotted steps around
        each curve, as the R twin draws them, strictly between 0 and 1.
        ``None`` leaves them out.
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
    censor_marks : bool
        Mark each censored observation with a "+" at the survival level
        current at its time.
    logrank : bool or None
        Whether to print the log-rank test of the group difference as a
        subtitle when there are groups. ``None``, the default, prints it and
        raises a ``FutureWarning``, since a future release will make ``False``
        the default, as in the R twin. The statistic and p-value are attached
        to the plot whenever there are groups, whatever this argument says.
    conf_type : {"log-log", "log", "plain"}
        The scale on which the limits are built from se, the Greenwood
        standard error of log S. ``"log-log"``, the default, gives
        ``S ** exp(±z * se / log S)``, the interval lifelines computes. These
        limits stay between 0 and 1, and in small samples they cover better
        than the untransformed limits ``S ± z * S * se`` (Borgan & Liestøl,
        1990). ``"plain"`` draws the untransformed limits, clipped to that
        range, and ``"log"`` draws ``S * exp(±z * se)``, capped at 1, as R's
        ``survival::survfit()`` does by default. The limits are 1 while the
        estimate is 1, and where a curve falls to 0, they keep the values of
        the step before. The formulas and these two rules are the R twin's, so
        the twins draw the same limits.

    Returns
    -------
    plotnine.ggplot
        The plot carries ``.at_risk`` (a DataFrame of number-at-risk counts at
        round times from 0 to at most the longest follow-up, which are the
        time-axis breaks when ``risk_table=True``) and, when grouped,
        ``.logrank_p`` and ``.logrank_stat``. Its data hold one row per time
        in each group's event table, with the columns ``time``, ``surv`` and
        ``group`` and, unless ``conf_level`` is ``None``, ``lower`` and
        ``upper``.

    References
    ----------
    Borgan, Ø., & Liestøl, K. (1990). A note on confidence intervals and bands
    for the survival function based on transformations. Scandinavian Journal
    of Statistics, 17(1), 35-41.

    Examples
    --------
    >>> import depictr as dp
    >>> ct = dp.clinical_trial()
    >>> p = dp.survival_plot(ct["time"], ct["event"], group=ct["arm"], logrank=True)
    """
    _require_lifelines()
    from lifelines import KaplanMeierFitter

    if conf_type not in _CONF_TYPES:
        raise ValueError(
            "`conf_type` must be one of 'log-log', 'log' or 'plain'."
        )
    # A level outside (0, 1), such as 95 meant as a percentage, has no normal
    # quantile, and _km_limits() would carry the origin's limits of 1 through
    # the whole curve.
    if conf_level is not None and not 0 < conf_level < 1:
        raise ValueError(
            "`conf_level` must lie strictly between 0 and 1, or be None to "
            "leave the limits out."
        )

    time = np.asarray(time, dtype=float)
    # A missing or infinite follow-up time has no place on a time axis, and it
    # used to surface far downstream as an unattributed StopIteration from the
    # axis-break search. Deciding whether to drop or impute is the analyst's
    # call, not something to make silently on their behalf.
    if not np.isfinite(time).all():
        raise ValueError(
            "`time` must be finite; drop or impute missing follow-up times "
            "before plotting."
        )
    event = _event_values(event)
    groups = (np.asarray(group) if group is not None
              else np.repeat("all", len(time)))
    # A missing group belongs to no arm, so drop those rows before the levels
    # are taken: pd.unique() keeps the missing value as a level of its own,
    # `groups == lvl` then matches nothing, and the arm is drawn as a phantom
    # all-censored curve labelled "None".
    missing = pd.isna(groups)
    if missing.any():
        warnings.warn(
            f"{int(missing.sum())} observation(s) with a missing group were "
            f"dropped.",
            UserWarning,
            stacklevel=2,
        )
        keep = ~missing
        time, event, groups = time[keep], event[keep], groups[keep]
        if not len(groups):
            raise ValueError("`group` is missing for every observation.")
    # A missing event is neither an event nor a censoring, so the observation
    # cannot enter the estimate. A cast to int used to turn it into the integer
    # sentinel numpy fills a NaN with, which lifelines counted as a death. The
    # warning is the R twin's message, word for word.
    missing_event = np.isnan(event)
    if missing_event.any():
        warnings.warn(
            f"{int(missing_event.sum())} observation(s) with a missing status "
            f"were dropped.",
            UserWarning,
            stacklevel=2,
        )
        keep = ~missing_event
        time, event, groups = time[keep], event[keep], groups[keep]
        if not len(event):
            raise ValueError("`event` is missing for every observation.")
    event = _normalise_event(event)
    # One factor order for the whole figure -- a user-set categorical order if
    # there is one, otherwise first appearance -- so the colour legend and the
    # risk-table rows list the groups identically.
    if isinstance(getattr(group, "dtype", None), pd.CategoricalDtype):
        seen = set(groups.tolist())
        levels = [c for c in group.cat.categories if c in seen]
    else:
        levels = list(pd.unique(groups))

    band = conf_level is not None
    curves, censors, at_risk_rows = [], [], []
    breaks = _nice_breaks(float(np.max(time)))
    for lvl in levels:
        mask = groups == lvl
        kmf = KaplanMeierFitter()
        kmf.fit(time[mask], event[mask], label=str(lvl))
        # The survival function is reported at every time of the event table,
        # censoring times included, so the two line up row by row.
        table = kmf.event_table
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["time", "surv"]
        if band:
            sf["lower"], sf["upper"] = _km_limits(
                sf["surv"], _greenwood_se(table), conf_level, conf_type)
        sf["group"] = str(lvl)
        curves.append(sf)
        # One mark per censored observation, at the survival level the curve
        # holds at its time, as the R twin draws them.
        n_censored = table["censored"].to_numpy(dtype=int)
        censors.append(pd.DataFrame({
            "time": np.repeat(sf["time"].to_numpy(), n_censored),
            "surv": np.repeat(sf["surv"].to_numpy(), n_censored),
            "group": str(lvl),
        }))
        for b in breaks:
            at_risk_rows.append({"group": str(lvl), "time": float(b),
                                 "n_at_risk": int(np.sum(time[mask] >= b))})
    order = [str(lvl) for lvl in levels]
    curve = pd.concat(curves, ignore_index=True)
    curve["group"] = pd.Categorical(curve["group"], categories=order, ordered=True)
    censor = pd.concat(censors, ignore_index=True)
    censor["group"] = pd.Categorical(censor["group"], categories=order,
                                     ordered=True)

    multi = len(levels) > 1
    if multi:
        p = ggplot(curve, aes("time", "surv", color="group"))
        colour = {}
    else:
        from .palette import BRAND
        p = ggplot(curve, aes("time", "surv"))
        colour = {"color": BRAND}
    if band:
        p = (p
             + geom_step(aes(y="lower"), linetype="dotted", size=0.4, **colour)
             + geom_step(aes(y="upper"), linetype="dotted", size=0.4, **colour))
    p = p + geom_step(size=0.9, **colour)
    if censor_marks and len(censor):
        p = p + geom_point(data=censor, shape="+", size=2, **colour)
    if multi:
        p = p + scale_colour_depictr()

    subtitle = None
    logrank_p = logrank_stat = None
    if multi:
        from lifelines.statistics import multivariate_logrank_test
        res = multivariate_logrank_test(time, groups, event)
        logrank_p, logrank_stat = res.p_value, res.test_statistic
        if logrank is None:
            warnings.warn(
                "survival_plot() adds the log-rank test as a subtitle when "
                "`logrank` is not given. A future release will make "
                "logrank=False the default, as in the R package, so pass "
                "logrank=True to keep the subtitle or logrank=False to leave "
                "it out.",
                FutureWarning, stacklevel=2,
            )
        if logrank is None or logrank:
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
    row_h, header = 0.05, 0.05
    y_of = {g: -(header + row_h * (i + 0.5)) for i, g in enumerate(order)}
    tbl = at_risk_df.copy()
    tbl["group"] = tbl["group"].astype(str)
    tbl["y"] = tbl["group"].map(y_of)
    ymin = -(header + row_h * len(order) + 0.02)
    # The ticks sit at the exact break times the counts were taken at, and only
    # their labels are formatted. Rounding the times themselves collapsed short
    # follow-up onto one tick and put the count at 0.25 under a "0.2".
    break_labels = [f"{b:g}" for b in breaks]
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
        + scale_x_continuous(limits=(0, tmax), breaks=breaks.tolist(),
                             labels=break_labels)
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
