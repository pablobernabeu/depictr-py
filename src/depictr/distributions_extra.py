"""Grouped-distribution EDA plots, themed consistently.

These extend the basic distribution view in :mod:`depictr.eda` with comparisons
across groups: empirical cumulative curves, overlapping ridgelines, paired
dumbbells, outlier flagging and group means with confidence intervals. Each is a
one-call function returning a single plotnine object you can extend with ``+``.
"""

from __future__ import annotations

import warnings

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

from ._checks import _check_conf_level
from ._groups import _group_scales, _prepare_group
from .palette import ACCENT, BRAND
from .theme import (
    legend_inside as _legend_inside,
)
from .theme import (
    scale_colour_depictr,
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
        A grouping column mapped to colour, drawn as one curve per level. A
        numeric code counts as a group, but a numeric column with more than 20
        distinct values raises ``ValueError``. Rows with a missing group are
        dropped with a warning, and categories without rows are left out of
        the plot and its palette.
    legend_inside : bool
        When ``True`` and a ``group`` is given, place the legend in the
        bottom-right, which an ECDF leaves empty once it saturates.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> ld = dp.lexical_decision()
    >>> p = dp.ecdf_plot(ld, "RT", group="condition", legend_inside=True)
    """
    if x not in data.columns:
        raise KeyError(f"{x!r} is not a column of `data`.")
    if group:
        # Rows missing x are not drawn, so they are left out before the groups
        # are formed, and the levels are those of the rows drawn.
        data, _, palette = _prepare_group(data.dropna(subset=[x]), group)
        p = (ggplot(data, aes(x=x, color=group))
             + stat_ecdf(size=0.9)
             + _group_scales(palette, ("color",)))
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
    progression is easy to follow, and each keeps the colour of its place in
    the group's own order, as in the R twin.

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
        The grouping column; one ridge is drawn per level. A numeric code
        counts as a group, but a numeric column with more than 20 distinct
        values raises ``ValueError``. Rows with a missing group are dropped
        with a warning, and categories without rows are left out of the plot
        and its palette.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.ridgeline_plot(wb, "life_satisfaction", "region")
    """
    for col in (x, group):
        if col not in data.columns:
            raise KeyError(f"{col!r} is not a column of `data`.")

    # Rows missing x are not drawn, so they are left out before the groups are
    # formed. The palette names each level's colour before the levels are
    # reordered, so a ridge's colour follows its level, not its median.
    df, _, palette = _prepare_group(data[[x, group]].dropna(subset=[x]), group)
    # Order levels by median so the ridges read as a smooth progression. The
    # facet rows go top to bottom, so reverse the order to put the largest on top.
    medians = df.groupby(group, observed=True)[x].median().sort_values()
    order = list(medians.index)[::-1]
    df[group] = pd.Categorical(df[group], categories=order, ordered=True)

    return (
        ggplot(df, aes(x=x, fill=group))
        # Each facet row holds one level, so there is nothing to stack, and
        # geom_area's default stacking made the plot many times slower to draw.
        + geom_area(stat="density", position="identity", alpha=0.8,
                    color="white", size=0.3)
        + facet_grid(f"{group} ~ .", scales="free_y")
        + _group_scales(palette, ("fill",))
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
        A two-level grouping column; its levels are the two points. They are
        taken in a categorical's declared order, and otherwise sorted, as in
        the R twin. The first level is the start of each segment, orders the
        categories and takes the first colour of the palette.
    legend_inside : bool
        When ``True``, place the two-group legend in the top-right of the panel
        rather than in a right-hand margin.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> wb = dp.wellbeing_survey()
    >>> wb = wb.assign(age_group=np.where(wb["age"] < 50, "under 50", "50 or over"))
    >>> p = dp.dumbbell_plot(wb, "region", "life_satisfaction", "age_group")
    """
    for col in (category, value, group):
        if col not in data.columns:
            raise KeyError(f"{col!r} is not a column of `data`.")
    if category == group:
        raise ValueError("`category` and `group` must be different columns.")

    # The two levels are the two ends of every segment. The first of them
    # orders the categories, and each takes its colour in turn. They are taken
    # as the R twin takes them, over the rows that are drawn: a categorical's
    # declared order, and otherwise sorted. Taking them in order of first
    # appearance let the row order of the input rearrange the figure.
    drawn = data.loc[data[[category, value, group]].notna().all(axis=1), group]
    codes = (drawn.cat.remove_unused_categories()
             if isinstance(drawn.dtype, pd.CategoricalDtype)
             else drawn.astype("category"))
    levels = list(codes.cat.categories)
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
    long[group] = pd.Categorical(long[group], categories=levels)

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

    Examples
    --------
    >>> import depictr as dp
    >>> cy = dp.crop_yield()
    >>> p = dp.outlier_plot(cy, "yield")
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


def group_comparison_plot(data, x, group, title=None, conf_level=0.95):
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
        The grouping column on the x-axis. A numeric code counts as a group,
        but a numeric column with more than 20 distinct values raises
        ``ValueError``. Rows with a missing group are dropped with a warning,
        and categories without rows are left out of the plot and its palette.
    title : str, optional
        Plot title.
    conf_level : float
        Confidence level for the t intervals, a single number strictly between
        0 and 1.

    Returns
    -------
    plotnine.ggplot

    Notes
    -----
    Each interval is the t interval for the group's mean, as in the R twin and
    in the upper panel of :func:`depictr.estimation_plot`. Its half-width is the
    standard error times the ``1 - (1 - conf_level) / 2`` quantile of the t
    distribution with n - 1 degrees of freedom. It describes the precision
    of the mean, not the spread of the data. depictr 0.3.0 used 1.96 standard
    errors, a normal approximation that covers only 81% of the time at n = 3
    and 88% at n = 5 for a nominal 95%.

    A group with a single observation has no degrees of freedom for the
    interval, so only its mean is drawn and a warning names the group.

    Examples
    --------
    >>> import depictr as dp
    >>> ld = dp.lexical_decision()
    >>> p = dp.group_comparison_plot(ld, "RT", "condition")
    """
    for col in (x, group):
        if col not in data.columns:
            raise KeyError(f"{col!r} is not a column of `data`.")
    _check_conf_level(conf_level)

    from scipy.stats import t as t_dist

    # Rows missing x are not drawn, so they are left out before the groups are
    # formed, and a group without observations is no group at all.
    df, _, palette = _prepare_group(data[[x, group]].dropna(subset=[x]), group)
    stats = (df.groupby(group, observed=True)[x]
             .agg(["mean", "std", "count"]).reset_index())
    se = stats["std"] / np.sqrt(stats["count"])
    # A single observation has no degrees of freedom: its standard deviation
    # and t quantile are NaN, so its bounds are NaN and only the mean is drawn.
    df_t = (stats["count"] - 1).where(stats["count"] > 1)
    tc = t_dist.ppf(1 - (1 - conf_level) / 2, df_t)
    stats["lower"] = stats["mean"] - tc * se
    stats["upper"] = stats["mean"] + tc * se
    sparse = stats.loc[stats["count"] < 2, group]
    if len(sparse):
        warnings.warn(
            "No confidence interval drawn for group(s) with n < 2: "
            f"{', '.join(str(g) for g in sparse)}; showing the mean only.",
            UserWarning,
            stacklevel=2,
        )

    return (
        ggplot(df, aes(x=group, color=group))
        + geom_jitter(aes(y=x), width=0.12, height=0, alpha=0.25, size=1.5)
        # na_rm: the NaN bounds of a single-observation group are dropped
        # quietly, since the warning above has already named the group.
        + geom_errorbar(stats, aes(ymin="lower", ymax="upper"),
                        width=0.15, size=0.9, na_rm=True)
        + geom_point(stats, aes(y="mean"), size=3.5)
        + _group_scales(palette, ("color",))
        + labs(x=group, y=x, title=title)
        + theme_depictr()
        + theme(legend_position="none")
    )
