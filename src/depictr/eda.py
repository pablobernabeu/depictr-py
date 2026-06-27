"""Exploratory-data-analysis plots, themed consistently.

These cover the high-traffic first look at a dataset -- distributions, category
counts, correlations and missingness -- each as a one-call function returning a
plotnine object you can extend with ``+``.
"""

from __future__ import annotations

import pandas as pd
from plotnine import (
    aes,
    after_stat,
    element_text,
    geom_bar,
    geom_density,
    geom_histogram,
    geom_path,
    geom_text,
    geom_tile,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_fill_gradientn,
    scale_fill_manual,
    scale_y_continuous,
    theme,
)

from .palette import ACCENT, BRAND, depictr_palette
from .theme import (
    legend_inside as _legend_inside,
    scale_colour_depictr,
    scale_fill_depictr,
    theme_depictr,
)


class _geom_density_linekey(geom_density):
    """A density geom whose legend key is a coloured line, not a filled box.

    ``geom_density`` inherits ``geom_polygon``'s legend drawer, which always
    draws a filled rectangle. Borrowing ``geom_path``'s drawer makes the key a
    short coloured line segment that matches the curve it labels.
    """

    draw_legend = staticmethod(geom_path.draw_legend)


def explore_distribution(data, x, group=None, kind="density", bins=30,
                         alpha=0.6, legend_inside=False, title=None):
    """Plot the distribution of a numeric variable, optionally split by a group.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    x : str
        Name of the numeric column to display.
    group : str, optional
        A grouping column mapped to colour/fill.
    kind : {"density", "histogram", "both"}
        What to draw.
    bins : int
        Number of histogram bins.
    alpha : float
        Fill transparency, useful when groups overlap.
    legend_inside : bool
        When ``True`` and a ``group`` is given, place the legend inside the
        top-right of the panel (a unimodal distribution leaves it empty) rather
        than in a right-hand margin.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    if x not in data.columns:
        raise KeyError(f"{x!r} is not a column of `data`.")
    mapping = aes(x=x, fill=group, color=group) if group else aes(x=x)
    p = ggplot(data, mapping)
    if kind in {"histogram", "both"}:
        y = aes(y=after_stat("density")) if kind == "both" else None
        if group:
            # Bars are filled by the group aesthetic; "none" hides the edge so
            # overlapping (identity-positioned) bars stay readable. A bare
            # fill=None would instead make the bars transparent, not group-filled.
            p = p + geom_histogram(y, bins=bins, alpha=alpha,
                                   position="identity", color="none")
        else:
            p = p + geom_histogram(y, bins=bins, alpha=alpha,
                                   position="identity", color="white", fill=BRAND)
    if kind in {"density", "both"}:
        if group:
            # A light fill (0.2) keeps overlapping curves distinguishable
            # without the muddy overlap a heavier fill makes; the coloured
            # outline carries the comparison. For "both" the density fill is
            # dropped so it does not muddy the histogram beneath it.
            p = p + _geom_density_linekey(alpha=0 if kind == "both" else 0.2)
        else:
            p = p + geom_density(alpha=alpha, color=BRAND, fill=BRAND)
        # The density baseline sits at 0; drop the default lower pad so the
        # curves are flush with the x-axis, keeping a little headroom on top.
        p = p + scale_y_continuous(expand=(0, 0, 0.05, 0))
    if group:
        p = p + scale_fill_depictr() + scale_colour_depictr()
        if kind in {"density", "both"}:
            # The density line is the keyed geom, so show a single legend keyed
            # by its colour: drop the redundant fill legend and make the key
            # fully opaque so it matches the curve.
            p = p + guides(
                color=guide_legend(override_aes={"alpha": 1, "size": 1.2}),
                fill=False,
            )
    y_lab = "Count" if kind == "histogram" else "Density"
    p = p + labs(x=x, y=y_lab, title=title) + theme_depictr()
    if legend_inside and group:
        p = p + _legend_inside("top right")
    return p


def explore_categorical(data, x, group=None, proportion=True, title=None):
    """Bar chart of a categorical variable, optionally grouped (dodged).

    Parameters
    ----------
    data : pandas.DataFrame
    x : str
        The categorical column.
    group : str, optional
        A second categorical column mapped to fill, drawn side by side.
    proportion : bool
        Show proportions within each group rather than raw counts.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
    """
    if group:
        counts = (data.groupby([group, x], observed=True).size()
                  .rename("n").reset_index())
        if proportion:
            counts["value"] = counts["n"] / counts.groupby(group, observed=True)["n"].transform("sum")
        else:
            counts["value"] = counts["n"]
        p = (ggplot(counts, aes(x=x, y="value", fill=group))
             + geom_bar(stat="identity", position="dodge")
             + scale_fill_depictr())
        y_lab = "Proportion" if proportion else "Count"
    else:
        p = ggplot(data, aes(x=x)) + geom_bar(fill=BRAND)
        y_lab = "Count"
    return p + labs(x=x, y=y_lab, title=title) + theme_depictr()


def correlation_heatmap(data, cols=None, title=None):
    """Heatmap of pairwise Pearson correlations among numeric columns.

    Parameters
    ----------
    data : pandas.DataFrame
    cols : list of str, optional
        Columns to include; defaults to all numeric columns.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
    """
    num = data[cols] if cols else data.select_dtypes("number")
    corr = num.corr()
    order = list(corr.columns)
    long = corr.reset_index().melt(id_vars="index", var_name="var2",
                                   value_name="r").rename(columns={"index": "var1"})
    long["var1"] = pd.Categorical(long["var1"], categories=order, ordered=True)
    long["var2"] = pd.Categorical(long["var2"], categories=order[::-1], ordered=True)
    long["label"] = long["r"].map(lambda v: f"{v:.2f}")
    return (
        ggplot(long, aes(x="var1", y="var2", fill="r"))
        + geom_tile(color="white")
        + geom_text(aes(label="label"), size=8, color="#1a1a1a")
        + scale_fill_gradientn(
            colors=depictr_palette(7, kind="diverging"), limits=(-1, 1),
            name="Correlation",
        )
        + labs(x="", y="", title=title)
        + theme_depictr(grid="none")
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
    )


def missingness_map(data, sort=True, legend_inside=False, title=None):
    """Tile map of missing values, one column per variable and one row per record.

    Variables are ordered most- to least-missing, so the worst offenders sit on
    the left. The percentage missing is shown in each axis label.

    Parameters
    ----------
    data : pandas.DataFrame
    sort : bool
        Order columns by their proportion of missing values.
    legend_inside : bool
        When ``True`` (and ``sort`` is on), place the legend in the top-right.
        Because the columns are sorted, the right-hand ones are the most complete,
        so a legend there sits over a solid "Present" block and hides no missing
        marks.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
    """
    miss = data.isna()
    frac = miss.mean()
    order = list(frac.sort_values(ascending=False).index) if sort else list(data.columns)
    labels = {c: f"{c} ({frac[c] * 100:.0f}%)" for c in order}
    long = (miss[order].reset_index(names="row")
            .melt(id_vars="row", var_name="variable", value_name="missing"))
    long["status"] = long["missing"].map({True: "Missing", False: "Present"})
    long["variable"] = pd.Categorical(long["variable"].map(labels),
                                      categories=[labels[c] for c in order],
                                      ordered=True)
    overall = miss.to_numpy().mean()
    p = (
        ggplot(long, aes(x="variable", y="row", fill="status"))
        + geom_tile()
        + scale_fill_manual(values={"Present": "#d9d9d9", "Missing": ACCENT},
                            name=None)
        + labs(x="", y="Record",
               title=title or f"{overall * 100:.1f}% of all values are missing")
        + theme_depictr(grid="none")
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
    )
    if legend_inside and sort:
        p = p + _legend_inside("top right")
    return p
