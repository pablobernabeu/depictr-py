"""Statistical-power plots, themed consistently.

Power analysis answers how large a sample needs to be to detect an effect of a
given size. The curve of power against sample size is the usual way to read off
that number. This plots a power curve from a tidy table you have already
computed (with statsmodels' power solvers, G*Power, simulation, pilotr or a
paper); it does no power calculation of its own.
"""

from __future__ import annotations

import warnings

import pandas as pd
from plotnine import (
    aes,
    geom_hline,
    geom_line,
    geom_linerange,
    geom_point,
    geom_ribbon,
    ggplot,
    labs,
    scale_y_continuous,
)

from .palette import BRAND
from .theme import scale_colour_depictr, scale_fill_depictr, theme_depictr

# The conventional target power: an 80% chance of detecting a true effect
# (Cohen, 1988). Drawn as a reference line so the crossing sample size is easy
# to read off.
_TARGET_POWER = 0.8

# Column names read without being named, as in the R twin. Sample size: simr's
# summary gives `nlevels`, and pilotr's sweeps over participants give
# `n_subject`. Power: simr gives `mean`. A power column found here is labelled
# "Power" on the axis.
_N_CANDIDATES = ("n", "nlevels", "x", "sample_size", "n_subject")
_POWER_CANDIDATES = ("power", "mean", "p")
# pilotr names the focal effect `effect` in power curves and `param` in
# precision curves. A tidy coefficient table names it `term`.
_GROUP_CANDIDATES = ("effect", "param", "term")
_LOWER_CANDIDATES = ("lower", "ci_low", "conf.low")
_UPPER_CANDIDATES = ("upper", "ci_high", "conf.high")

# The R twin's grey88 band and grey50 error bars.
_BAND_FILL = "#e0e0e0"
_BAR_COLOUR = "#7f7f7f"


def _pick(data, candidates):
    return next((c for c in candidates if c in data.columns), None)


def _check_column(data, col, arg):
    if col is not None and col not in data.columns:
        raise KeyError(f"Column {col!r} (argument `{arg}`) not found in the data.")
    return col


def power_curve_plot(data, n=None, power=None, group=None, title=None,
                     target=_TARGET_POWER, lower=None, upper=None,
                     ribbon=True, y_lab=None):
    """Line plot of statistical power against sample size.

    Reads a precomputed tidy table and draws power as a function of sample size,
    with a dashed reference line at the target power (0.8 by default). Where the
    curve crosses the line is the sample size that reaches adequate power.

    A table may hold several curves, one per effect or design, stacked by row.
    Each is drawn as its own line, told apart by colour and line type. The
    tables pilotr returns are read as they are: ``power_curve_mixed()`` gives
    ``n_subject``, ``effect``, ``power``, ``power_lo`` and ``power_hi``, and
    ``precision_curve()`` gives ``n_subject``, ``param`` and the decision
    probabilities ``p_meaningful`` and ``p_equivalent`` with their own ``_lo``
    and ``_hi`` bounds, which ``power`` can name.

    Parameters
    ----------
    data : pandas.DataFrame or list of dict
        A tidy table with one row per sample size (per curve, if there are
        several). A list of dicts, one per row, is read with
        :class:`pandas.DataFrame`.
    n : str, optional
        Name of the sample-size column. By default, the first of ``n``,
        ``nlevels``, ``x``, ``sample_size`` and ``n_subject`` present.
    power : str, optional
        Name of the power column, on the 0-1 scale. By default, the first of
        ``power``, ``mean`` and ``p`` present. Any proportion works, such as
        pilotr's ``p_meaningful``.
    group : str, optional
        A column that separates several curves. Each level is drawn as its own
        line, in its own colour and line type. By default, an ``effect``,
        ``param`` or ``term`` column with more than one level is used, with a
        warning. Rows that share a sample size with no such column raise a
        ``ValueError``, because joining them would draw a zig-zag.
    title : str, optional
        Plot title.
    target : float, optional
        Target power, drawn as a horizontal reference line. ``None`` omits it.
    lower, upper : str, optional
        Columns holding the lower and upper confidence limits. Name both or
        neither. By default, ``<power>_lo`` and ``<power>_hi`` are used, then
        ``lower``/``upper``, ``ci_low``/``ci_high`` or ``conf.low``/``conf.high``.
    ribbon : bool
        Whether to draw the confidence band as a shaded ribbon (``True``) or as
        error bars (``False``).
    y_lab : str, optional
        Y-axis label. By default, "Power" when the power column is one of the
        default names, and the column's name otherwise.

    Returns
    -------
    plotnine.ggplot

    References
    ----------
    Cohen, J. (1988). Statistical power analysis for the behavioral sciences
    (2nd ed.). Lawrence Erlbaum Associates.

    Examples
    --------
    >>> import depictr as dp
    >>> import pandas as pd
    >>> from statsmodels.stats.power import TTestIndPower
    >>> analysis = TTestIndPower()
    >>> power = pd.DataFrame({"n": range(25, 425, 25)})
    >>> power["power"] = [analysis.power(effect_size=0.5, nobs1=n, alpha=0.05)
    ...                   for n in power["n"]]
    >>> p = dp.power_curve_plot(power)

    Two curves in one table, laid out as pilotr's ``power_curve_mixed()``
    returns them, one row per sample size and effect:

    >>> rows = [
    ...     {"n_subject": n, "effect": e, "power": pw,
    ...      "power_lo": pw - 0.08, "power_hi": pw + 0.08}
    ...     for n, e, pw in [(20, "condition", 0.31), (20, "frequency", 0.12),
    ...                      (40, "condition", 0.58), (40, "frequency", 0.21),
    ...                      (60, "condition", 0.78), (60, "frequency", 0.33)]
    ... ]
    >>> p = dp.power_curve_plot(rows, group="effect")
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    n = _check_column(data, n, "n") or _pick(data, _N_CANDIDATES)
    power = _check_column(data, power, "power") or _pick(data, _POWER_CANDIDATES)
    if n is None or power is None:
        raise KeyError("Could not find sample-size and power columns in the "
                       "data. Name them with `n` and `power`.")

    _check_column(data, lower, "lower")
    _check_column(data, upper, "upper")
    if (lower is None) != (upper is None):
        raise ValueError("Name both `lower` and `upper`, or neither.")
    if lower is None:
        # The bounds of the chosen power column come first, so that a pilotr
        # precision table banded by p_meaningful_lo is not given another
        # column's bounds.
        lower = _pick(data, (f"{power}_lo", *_LOWER_CANDIDATES))
        upper = _pick(data, (f"{power}_hi", *_UPPER_CANDIDATES))
    has_band = lower is not None and upper is not None

    group = _check_column(data, group, "group")
    if group is None:
        # A column with a single level names the one curve the table holds and
        # separates nothing.
        multi = [c for c in _GROUP_CANDIDATES
                 if c in data.columns and data[c].dropna().nunique() > 1]
        if multi:
            group = multi[0]
            warnings.warn(
                f"power_curve_plot(): drawing one curve per level of `{group}`, "
                f"with sample size from `{n}` and power from `{power}`. Name "
                f"the column with `group` to silence this warning.",
                UserWarning,
                stacklevel=2,
            )

    rows = data[data[n].notna()]
    if group is None:
        if rows[n].duplicated().any():
            raise ValueError("Several rows share a sample size; name the column "
                             "that separates the curves with `group`.")
    else:
        if rows.duplicated(subset=[n, group]).any():
            raise ValueError("Several rows of one curve share a sample size. "
                             "Summarise them, or name a column that separates "
                             "them with `group`.")
        # Levels in their order of appearance, as in the R twin, and drawn as
        # discrete even when the column holds numbers.
        data = data.assign(**{group: pd.Categorical(
            data[group], categories=pd.unique(data[group].dropna()))})

    if y_lab is None:
        y_lab = "Power" if power in _POWER_CANDIDATES else power

    mapping = (aes(x=n, y=power, color=group, linetype=group, group=group)
               if group else aes(x=n, y=power))
    p = ggplot(data, mapping)

    if has_band:
        if ribbon and group:
            # Translucent, so that overlapping bands stay visible, and without
            # the line type, which belongs to the curve.
            p = (p + geom_ribbon(aes(ymin=lower, ymax=upper, fill=group),
                                 alpha=0.15, color="none", linetype="solid",
                                 show_legend=False)
                 + scale_fill_depictr(guide=None))
        elif ribbon:
            p = p + geom_ribbon(aes(ymin=lower, ymax=upper), fill=_BAND_FILL)
        elif group:
            p = p + geom_linerange(aes(ymin=lower, ymax=upper),
                                   linetype="solid")
        else:
            p = p + geom_linerange(aes(ymin=lower, ymax=upper),
                                   color=_BAR_COLOUR)

    if target is not None and not pd.isna(target):
        p = p + geom_hline(yintercept=target, linetype="dashed",
                           color="#9e9e9e")
    if group:
        p = (p + geom_line(size=0.9) + geom_point(size=2)
             + scale_colour_depictr())
    else:
        p = (p + geom_line(color=BRAND, size=0.9)
             + geom_point(color=BRAND, size=2))
    return (
        p
        + scale_y_continuous(limits=(0, 1))
        + labs(x="Sample size", y=y_lab, color=group, linetype=group,
               title=title)
        + theme_depictr()
    )
