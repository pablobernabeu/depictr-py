"""Two-variable exploratory plots, themed consistently.

These cover the second look at a dataset, once you have moved past single-variable
distributions and want to see how two columns relate. :func:`scatter_trend` is
the explicit numeric-by-numeric case; :func:`explore_bivariate` picks a sensible
plot from the column types so you do not have to.
"""

from __future__ import annotations

from pandas.api.types import is_numeric_dtype
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

from .palette import BRAND, depictr_palette
from .theme import scale_colour_depictr, scale_fill_depictr, theme_depictr


def scatter_trend(data, x, y, group=None, method="lm", title=None):
    """Scatter plot with a fitted trend line and confidence band.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    x, y : str
        Names of the numeric columns for the horizontal and vertical axes.
    group : str, optional
        A grouping column. When given, points and a separate trend line are
        coloured by group via :func:`depictr.theme.scale_colour_depictr`.
    method : str
        Smoothing method passed to :func:`plotnine.geom_smooth`, for example
        ``"lm"`` for a straight line or ``"lowess"`` for a local fit.
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
        if group not in data.columns:
            raise KeyError(f"{group!r} is not a column of `data`.")
        p = (
            ggplot(data, aes(x=x, y=y, color=group, fill=group))
            + geom_point(alpha=0.6)
            + geom_smooth(method=method, alpha=0.2)
            + scale_colour_depictr()
            + scale_fill_depictr()
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
      (:func:`scatter_trend`);
    - one numeric and one categorical column give boxplots of the numeric
      variable across the categories, with the raw points jittered over them;
    - two categorical columns give a tile of joint counts, shaded by frequency.

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

    x_num = is_numeric_dtype(data[x])
    y_num = is_numeric_dtype(data[y])

    if x_num and y_num:
        return scatter_trend(data, x, y, title=title)

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
            + labs(x=cat, y=num, title=title)
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
