"""Grouped-distribution EDA plots, themed consistently.

These extend the basic distribution view in :mod:`depictr.eda` with comparisons
across groups: empirical cumulative curves, overlapping ridgelines, paired
dumbbells, outlier flagging and group means with confidence intervals. Each is a
one-call function returning a single plotnine object you can extend with ``+``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_text,
    facet_grid,
    geom_area,
    geom_boxplot,
    geom_errorbar,
    geom_jitter,
    geom_point,
    geom_segment,
    ggplot,
    labs,
    stat_ecdf,
    theme,
)

from .palette import ACCENT, BRAND
from .theme import (
    legend_inside as _legend_inside,
    scale_colour_depictr,
    scale_fill_depictr,
    theme_depictr,
)


def ecdf_plot(data, x, group=None, legend_inside=False, title=None):
    """Empirical cumulative distribution function, one step curve per group.

    The ECDF reads off the proportion of observations at or below each value, so
    it shows the whole distribution without the smoothing choices a density makes.
    Curves that sit to the right are shifted towards larger values.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    x : str
        Name of the numeric column.
    group : str, optional
        A grouping column mapped to colour, drawn as one curve per level.
    legend_inside : bool
        When ``True`` and a ``group`` is given, place the legend in the
        bottom-right, which an ECDF leaves empty once it saturates.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    if x not in data.columns:
        raise KeyError(f"{x!r} is not a column of `data`.")
    if group:
        p = (ggplot(data, aes(x=x, color=group))
             + stat_ecdf(size=0.9)
             + scale_colour_depictr())
    else:
        p = ggplot(data, aes(x=x)) + stat_ecdf(size=0.9, color=BRAND)
    p = p + labs(x=x, y="Cumulative proportion", title=title) + theme_depictr()
    if legend_inside and group:
        p = p + _legend_inside("bottom right")
    return p


def ridgeline_plot(data, x, group, title=None):
    """Overlapping densities, one ridge per group level.

    Each level gets its own filled density on a shared x-axis, stacked so the
    ridges overlap a little. Reading top to bottom gives a quick sense of how the
    distribution shifts across levels. Levels are ordered by their median so the
    progression is easy to follow.

    The overlap is faked with :func:`plotnine.facet_grid`: one narrow row per
    level, panel spacing pulled negative so neighbouring ridges touch. It stays a
    single ggplot, since plotnine has no dedicated ridgeline geom.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    x : str
        Name of the numeric column.
    group : str
        The grouping column; one ridge is drawn per level.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    for col in (x, group):
        if col not in data.columns:
            raise KeyError(f"{col!r} is not a column of `data`.")

    df = data[[x, group]].dropna().copy()
    # Order levels by median so the ridges read as a smooth progression. The
    # facet rows go top to bottom, so reverse the order to put the largest on top.
    medians = df.groupby(group, observed=True)[x].median().sort_values()
    order = list(medians.index)[::-1]
    df[group] = pd.Categorical(df[group], categories=order, ordered=True)

    return (
        ggplot(df, aes(x=x, fill=group))
        + geom_area(stat="density", alpha=0.8, color="white", size=0.3)
        + facet_grid(f"{group} ~ .", scales="free_y")
        + scale_fill_depictr()
        # Negative panel spacing makes neighbouring ridges overlap; the y-axis
        # within each strip carries no quantitative meaning, so drop it.
        + theme_depictr(grid="x")
        + theme(
            panel_spacing_y=-0.02,
            axis_text_y=element_blank(),
            axis_ticks_major_y=element_blank(),
            legend_position="none",
            # The default row-strip label is rotated; reading group names
            # horizontally beside each ridge is easier.
            strip_text_y=element_text(angle=0, ha="left"),
        )
        + labs(x=x, y=group, title=title)
    )


def dumbbell_plot(data, category, value, group, legend_inside=False, title=None):
    """Dumbbell plot pairing two group values per category.

    For a group with exactly two levels, each category becomes one row with a
    point for each level joined by a segment. The length of the segment is the gap
    between the two levels, which is the comparison the plot is built to show.

    Parameters
    ----------
    data : pandas.DataFrame
        Either one row per category-and-group combination, or raw rows that are
        averaged to one value per combination.
    category : str
        The categorical column, one row per level on the y-axis.
    value : str
        The numeric column plotted along the x-axis.
    group : str
        A two-level grouping column; its levels are the two points.
    legend_inside : bool
        When ``True``, place the two-group legend in the top-right of the panel
        rather than in a right-hand margin.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    for col in (category, value, group):
        if col not in data.columns:
            raise KeyError(f"{col!r} is not a column of `data`.")
    if category == group:
        raise ValueError("`category` and `group` must be different columns.")

    levels = list(pd.unique(data[group].dropna()))
    if len(levels) != 2:
        raise ValueError(
            f"`dumbbell_plot` needs exactly two levels in {group!r}; "
            f"found {len(levels)}."
        )

    # Collapse to one value per category-and-group cell (a no-op if the caller
    # already passed a summary), then widen so the two points share a row.
    cell = (data.groupby([category, group], observed=True)[value]
            .mean().reset_index())
    wide = cell.pivot(index=category, columns=group, values=value)
    wide = wide.dropna(subset=levels).reset_index()

    # Order categories by the first level's value so the plot reads top to bottom.
    order = list(wide.sort_values(levels[0])[category])
    wide[category] = pd.Categorical(wide[category], categories=order, ordered=True)

    long = wide.melt(id_vars=category, value_vars=levels,
                     var_name=group, value_name=value)

    p = (
        ggplot(wide, aes(y=category))
        + geom_segment(aes(x=levels[0], xend=levels[1],
                           yend=category), color="#9e9e9e", size=1.2)
        + geom_point(long, aes(x=value, y=category, color=group), size=3.5)
        + scale_colour_depictr()
        + labs(x=value, y=category, title=title)
        + theme_depictr(grid="x")
    )
    if legend_inside:
        p = p + _legend_inside("top right")
    return p


def outlier_plot(data, x, title=None):
    """Box plot of one variable with outliers flagged in the accent colour.

    Points beyond 1.5 times the interquartile range from the nearest quartile,
    the usual Tukey rule the box plot's whiskers already use, are drawn on top in
    the accent colour. The box itself suppresses its own outlier markers so they
    are not drawn twice.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    x : str
        Name of the numeric column to inspect.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    if x not in data.columns:
        raise KeyError(f"{x!r} is not a column of `data`.")

    values = data[x].dropna()
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = values[(values < lo) | (values > hi)]
    flagged = pd.DataFrame({x: outliers, "_y": 0.0})

    # A single box, drawn horizontally; the constant y just gives geom_boxplot a
    # group to sit on. outlier_alpha=0 hides the box's own markers.
    base = data[[x]].dropna().assign(_y=0.0)
    p = (ggplot(base, aes(x="_y", y=x))
         + geom_boxplot(width=0.4, fill=BRAND, alpha=0.25, color=BRAND,
                        outlier_alpha=0))
    if len(flagged):
        p = p + geom_point(flagged, aes(x="_y", y=x), color=ACCENT, size=2.5,
                           position="jitter")
    n = len(outliers)
    sub = f"{n} outlier{'s' if n != 1 else ''} beyond 1.5 x IQR"
    return (
        p
        + coord_flip()
        + labs(x="", y=x, title=title, subtitle=sub)
        + theme_depictr(grid="x")
        + theme(axis_text_y=element_blank(), axis_ticks_major_y=element_blank())
    )


def group_comparison_plot(data, x, group, title=None):
    """Group means with confidence intervals over the raw points.

    The estimate-with-uncertainty alternative to a bar chart: each group shows its
    raw observations (jittered), the mean, and a confidence interval for the mean.
    Showing the spread of the data alongside the estimate guards against reading
    too much into a difference in means.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    x : str
        Name of the numeric column being compared.
    group : str
        The grouping column on the x-axis.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    Notes
    -----
    The interval is a normal-approximation 95% confidence interval for the mean
    (mean plus or minus 1.96 standard errors). It describes the precision of the
    mean, not the spread of the data.
    """
    for col in (x, group):
        if col not in data.columns:
            raise KeyError(f"{col!r} is not a column of `data`.")

    df = data[[x, group]].dropna()
    stats = (df.groupby(group, observed=True)[x]
             .agg(["mean", "std", "count"]).reset_index())
    se = stats["std"] / np.sqrt(stats["count"])
    stats["lower"] = stats["mean"] - 1.96 * se
    stats["upper"] = stats["mean"] + 1.96 * se

    return (
        ggplot(df, aes(x=group, color=group))
        + geom_jitter(aes(y=x), width=0.12, height=0, alpha=0.25, size=1.5)
        + geom_errorbar(stats, aes(ymin="lower", ymax="upper"),
                        width=0.15, size=0.9)
        + geom_point(stats, aes(y="mean"), size=3.5)
        + scale_colour_depictr()
        + labs(x=group, y=x, title=title)
        + theme_depictr()
        + theme(legend_position="none")
    )
