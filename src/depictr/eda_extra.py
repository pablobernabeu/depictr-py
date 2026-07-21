"""Further exploratory-data-analysis plots, themed consistently.

These pick up where :mod:`depictr.eda` leaves off: a raincloud for looking at a
distribution and its raw points at once, and a scatter-plot matrix for the
pairwise relationships among a handful of numeric variables. Both return a
plotnine object you can extend with ``+`` (the matrix is a composition, so it
carries ``.draw`` and ``.save`` like any single plot).
"""

from __future__ import annotations

from plotnine import (
    aes,
    element_blank,
    geom_boxplot,
    geom_density,
    geom_point,
    geom_violin,
    ggplot,
    labs,
    position_jitter,
    position_nudge,
    theme,
)

from .compose import arrange_plots
from .palette import BRAND
from .theme import scale_colour_depictr, scale_fill_depictr, theme_depictr

# Past this many panels the cells get too small to read, so the matrix is capped.
_MAX_PAIRS_COLS = 5


def raincloud_plot(data, x, group=None, title=None):
    """Raincloud plot of a numeric variable, optionally split by a group.

    A raincloud sets three views of the same distribution side by side: a
    half-violin density (the "cloud"), a narrow boxplot for the median and
    quartiles, and the jittered raw points (the "rain"). Seeing the shape, the
    summary and every observation together guards against the boxplot hiding
    bimodality or a thin tail (Allen et al., 2021).

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    x : str
        Name of the numeric column whose distribution is shown.
    group : str, optional
        A categorical column. Each level gets its own raincloud along the
        horizontal axis, coloured by the depictr palette. With no group a single
        raincloud is drawn.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    References
    ----------
    Allen, M., Poggiali, D., Whitaker, K., Marshall, T. R., van Langen, J., &
    Kievit, R. A. (2021). Raincloud plots: A multi-platform tool for robust data
    visualization. Wellcome Open Research, 4, 63.
    https://doi.org/10.12688/wellcomeopenres.15191.2

    Examples
    --------
    >>> import depictr as dp
    >>> ld = dp.lexical_decision()
    >>> p = dp.raincloud_plot(ld, "RT", group="condition")
    """
    if x not in data.columns:
        raise KeyError(f"{x!r} is not a column of `data`.")
    if group is not None and group not in data.columns:
        raise KeyError(f"{group!r} is not a column of `data`.")

    # With no group, a constant axis category gives every layer one slot to
    # share, so the half-violin, box and points line up.
    if group is None:
        data = data.assign(_all="")
        cat = "_all"
        mapping = aes(x=cat, y=x)
        violin = geom_violin(style="left", trim=True, width=1.0,
                             fill=BRAND, color=None, alpha=0.5)
        box = geom_boxplot(width=0.10, position=position_nudge(x=0.12),
                           outlier_alpha=0, alpha=0.7, color="#4d4d4d",
                           fill="white")
        points = geom_point(position=position_jitter(width=0.06, height=0),
                            color=BRAND, size=0.9, alpha=0.35)
    else:
        cat = group
        mapping = aes(x=cat, y=x, fill=group, color=group)
        # The half-violin sits on the left of each slot; the box and points are
        # nudged right so they sit beside it rather than on top.
        violin = geom_violin(style="left", trim=True, width=1.0,
                             color=None, alpha=0.5)
        box = geom_boxplot(width=0.10, position=position_nudge(x=0.12),
                           outlier_alpha=0, alpha=0.7, color="#4d4d4d")
        points = geom_point(position=position_jitter(width=0.06, height=0),
                            size=0.9, alpha=0.35)

    p = ggplot(data, mapping) + violin + box + points
    if group is not None:
        p = p + scale_fill_depictr() + scale_colour_depictr()
    x_lab = None if group is None else group
    p = p + labs(x=x_lab, y=x, title=title) + theme_depictr(grid="y")
    if group is not None:
        # The group is already named on the x-axis, so the colour legend only
        # repeats it; drop it.
        p = p + theme(legend_position="none")
    return p


def explore_pairs(data, cols=None, title=None):
    """Scatter-plot matrix over a few numeric columns.

    Builds the familiar pairs grid: a scatter of every variable against every
    other off the diagonal, and that variable's own density on the diagonal. Each
    cell is a small themed plot, composed into an N x N grid. The number of
    columns is capped at five, beyond which the cells are too small to read.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    cols : list of str, optional
        Numeric columns to include, in order. Defaults to the first few numeric
        columns of ``data``. More than five are trimmed to the first five.
    title : str, optional
        Accepted for API symmetry, but dropped with a warning: plotnine
        compositions cannot carry a figure-level title, so the matrix draws
        without one.

    Returns
    -------
    plotnine.ggplot or plotnine.composition.Compose
        The composed grid from :func:`depictr.compose.arrange_plots`.

    Examples
    --------
    >>> import depictr as dp
    >>> cy = dp.crop_yield()
    >>> p = dp.explore_pairs(cy, cols=["rainfall", "fertiliser", "yield"])
    """
    if cols is None:
        cols = list(data.select_dtypes("number").columns)
    else:
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise KeyError(f"not column(s) of `data`: {missing}")
    if len(cols) < 2:
        raise ValueError("explore_pairs needs at least two numeric columns.")
    cols = cols[:_MAX_PAIRS_COLS]
    n = len(cols)

    # Axis titles and tick labels belong only on the outer edge, so the inner
    # cells stay clean and the variable names sit on the left and bottom.
    panels = []
    for r, y_var in enumerate(cols):
        for c, x_var in enumerate(cols):
            on_left = c == 0
            on_bottom = r == n - 1
            if r == c:
                cell = (ggplot(data, aes(x=x_var))
                        + geom_density(fill=BRAND, color=BRAND, alpha=0.4))
            else:
                cell = (ggplot(data, aes(x=x_var, y=y_var))
                        + geom_point(color=BRAND, size=0.8, alpha=0.4))
            cell = (cell
                    + labs(x=x_var if on_bottom else "",
                           y=y_var if on_left else "")
                    + theme_depictr(base_size=9))
            edge = {}
            if not on_bottom:
                edge["axis_text_x"] = element_blank()
            if not on_left:
                edge["axis_text_y"] = element_blank()
            if edge:
                cell = cell + theme(**edge)
            panels.append(cell)

    return arrange_plots(*panels, ncol=n, title=title)
