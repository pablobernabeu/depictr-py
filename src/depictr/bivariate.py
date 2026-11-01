"""Two-variable exploratory plots, themed consistently.

These cover the second look at a dataset, once you have moved past single-variable
distributions and want to see how two columns relate. :func:`scatter_trend` is
the explicit numeric-by-numeric case; :func:`explore_bivariate` picks a sensible
plot from the column types so you do not have to.
"""

from __future__ import annotations

from pandas.api.types import is_bool_dtype, is_timedelta64_dtype
from plotnine import (
    aes,
    geom_boxplot,
    geom_jitter,
    geom_point,
    geom_smooth,
    geom_tile,
    ggplot,
    labs,
    position_jitter,
    scale_fill_gradientn,
)

from ._groups import _group_scales, _prepare_group
from .palette import BRAND, depictr_palette
from .tables import _duration_unit, _is_continuous
from .theme import scale_fill_depictr, theme_depictr


def scatter_trend(data, x, y, group=None, method="lm", title=None):
    """Scatter plot with a fitted trend line, and a confidence band where the smoother gives one.

    The band appears only when the smoother computes one, as ``"lm"``,
    ``"glm"``, ``"gls"`` and ``"mavg"`` do, and ``"loess"`` once scikit-misc is
    installed (plotnine raises an error for it otherwise). ``"lowess"`` and
    ``"rlm"`` draw the line alone, and plotnine warns that it has no interval
    for them.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    x, y : str
        Names of the numeric columns for the horizontal and vertical axes.
    group : str, optional
        A grouping column. When given, points and a separate trend line are
        coloured by group in the depictr palette. A numeric code counts as a
        group, but a numeric column with more than 20 distinct values raises
        ``ValueError``. Rows with a missing group are dropped with a warning,
        and categories without rows are left out of the plot and its palette.
    method : str
        Smoothing method passed to ``plotnine.geom_smooth``, for example
        ``"lm"`` for a straight line with a band, or ``"lowess"`` for a local
        fit that needs no extra dependency and draws no band.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> cy = dp.crop_yield()
    >>> p = dp.scatter_trend(cy, "fertiliser", "yield", group="treatment")
    """
    for col in (x, y):
        if col not in data.columns:
            raise KeyError(f"{col!r} is not a column of `data`.")

    if group:
        # Rows missing x or y are not drawn, so they are left out before the
        # groups are formed, and the levels are those of the points drawn.
        data, _, palette = _prepare_group(data.dropna(subset=[x, y]), group)
        p = (
            ggplot(data, aes(x=x, y=y, color=group, fill=group))
            + geom_point(alpha=0.6)
            + geom_smooth(method=method, alpha=0.2)
            + _group_scales(palette)
        )
    else:
        # One series: keep it on the brand colour so a lone scatter still reads
        # as depictr without a redundant legend.
        p = (
            ggplot(data, aes(x=x, y=y))
            + geom_point(color=BRAND, alpha=0.6)
            + geom_smooth(method=method, color=BRAND, fill=BRAND, alpha=0.2)
        )
    return p + labs(x=x, y=y, title=title) + theme_depictr()


def explore_bivariate(data, x, y, title=None):
    """Plot the relationship between two columns, choosing the plot from their types.

    The plot is selected from the column dtypes:

    - two numeric columns give a scatter plot with a trend line
      ([`scatter_trend`][depictr.scatter_trend]);
    - one numeric and one categorical column give boxplots of the numeric
      variable across the categories, with the raw points jittered over them;
    - two categorical columns give a tile of joint counts, shaded by frequency.

    Datetime and timedelta columns count as numeric, so against a numeric
    column they give a scatter plot with a trend, and against a categorical one
    boxplots of their values across the categories. A boolean column counts as
    categorical, as a logical vector does in the R twin. A duration is drawn as a
    number in the unit [`summary_table`][depictr.summary_table] reports it in,
    named in the axis title, as in ``stay (days)``. The R twin draws a
    ``difftime`` the same way, in its own units.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    x, y : str
        Names of the two columns to relate.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> ld = dp.lexical_decision()
    >>> p = dp.explore_bivariate(ld, "condition", "RT")
    """
    for col in (x, y):
        if col not in data.columns:
            raise KeyError(f"{col!r} is not a column of `data`.")

    # Dates and durations are continuous, although pandas does not call them
    # numeric, and a boolean column is a category, although pandas calls it
    # numeric, as the R twin's is.numeric() takes a logical column for a
    # category.
    x_num = _is_continuous(data[x]) and not is_bool_dtype(data[x])
    y_num = _is_continuous(data[y]) and not is_bool_dtype(data[y])
    data, axis_title = _durations_as_numbers(data, (x, y))

    if x_num and y_num:
        return (scatter_trend(data, x, y, title=title)
                + labs(x=axis_title[x], y=axis_title[y]))

    if x_num != y_num:
        # One of each: draw the numeric variable across the categories. Putting
        # the categorical on the x axis keeps the boxes upright and readable.
        cat, num = (y, x) if x_num else (x, y)
        p = (
            ggplot(data, aes(x=cat, y=num, fill=cat))
            + geom_boxplot(alpha=0.7, outlier_alpha=0)
            + geom_jitter(position=position_jitter(width=0.2), alpha=0.3,
                          color="#4d4d4d", size=1)
            + scale_fill_depictr()
            + labs(x=cat, y=axis_title[num], title=title)
            + theme_depictr()
        )
        return p

    # Two categoricals: a frequency tile of the joint distribution.
    counts = (data.groupby([x, y], observed=True).size()
              .rename("n").reset_index())
    return (
        ggplot(counts, aes(x=x, y=y, fill="n"))
        + geom_tile(color="white")
        + scale_fill_gradientn(colors=depictr_palette(7, kind="sequential"),
                               name="Count")
        + labs(x=x, y=y, title=title)
        + theme_depictr(grid="none")
    )


def _durations_as_numbers(data, cols):
    """Return ``data`` with each timedelta column among ``cols`` as a number in
    the unit :func:`summary_table` reports it in, and each column's axis title,
    which names that unit.

    A number in a stated unit is how the R twin draws a ``difftime``, for which
    ggplot2 has no scale. It also keeps the axis off plotnine's timedelta
    scale, whose breaks mizani converts in a way that recent NumPy releases
    deprecate.
    """
    titles = {}
    for col in dict.fromkeys(cols):  # x and y may be one column
        titles[col] = col
        if is_timedelta64_dtype(data[col]):
            unit, size = _duration_unit(data[col])
            data = data.copy()
            data[col] = data[col].dt.total_seconds() / size
            titles[col] = f"{col} ({unit})"
    return data, titles
