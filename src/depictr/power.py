"""Statistical-power plots, themed consistently.

Power analysis answers how large a sample needs to be to detect an effect of a
given size. The curve of power against sample size is the usual way to read off
that number. This plots a power curve from a tidy table you have already
computed (with statsmodels' power solvers, G*Power, simulation, or a paper); it
does no power calculation of its own.
"""

from __future__ import annotations

from plotnine import (
    aes,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_y_continuous,
)

from .palette import BRAND
from .theme import scale_colour_depictr, theme_depictr

# The conventional target power: an 80% chance of detecting a true effect
# (Cohen, 1988). Drawn as a reference line so the crossing sample size is easy
# to read off.
_TARGET_POWER = 0.8


def power_curve_plot(data, n="n", power="power", group=None, title=None):
    """Line plot of statistical power against sample size.

    Reads a precomputed tidy table and draws power as a function of sample size,
    with a dashed reference line at 0.8 (the conventional target). Where the
    curve crosses the line is the sample size that reaches adequate power. Pass a
    ``group`` to overlay one coloured curve per condition (for example several
    effect sizes or designs).

    Parameters
    ----------
    data : pandas.DataFrame
        A tidy table with one row per sample size (per group, if grouped).
    n : str
        Name of the sample-size column.
    power : str
        Name of the power column, on the 0-1 scale.
    group : str, optional
        A column mapped to colour, one curve per level.
    title : str, optional
        Plot title.

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
    """
    for col in (n, power, *( (group,) if group else () )):
        if col not in data.columns:
            raise KeyError(f"{col!r} is not a column of `data`.")

    mapping = (aes(x=n, y=power, color=group, group=group) if group
               else aes(x=n, y=power))
    p = ggplot(data, mapping)
    p = p + geom_hline(yintercept=_TARGET_POWER, linetype="dashed",
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
        + labs(x="Sample size", y="Power",
               color=group if group else None, title=title)
        + theme_depictr()
    )
