"""Exploratory-data-analysis plots, themed consistently.

These cover the high-traffic first look at a dataset -- distributions, category
counts, correlations and missingness -- each as a one-call function returning a
plotnine object you can extend with ``+``.
"""

from __future__ import annotations

import warnings

import pandas as pd
from pandas.api.types import is_numeric_dtype
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
    scale_color_identity,
    scale_fill_gradientn,
    scale_fill_manual,
    scale_y_continuous,
    theme,
)

from ._groups import _group_scales, _prepare_group
from .audit import _legible_text_colour
from .palette import ACCENT, BRAND, depictr_palette
from .theme import (
    _scale_fills,
    theme_depictr,
)
from .theme import (
    legend_inside as _legend_inside,
)


class _geom_density_linekey(geom_density):
    """A density geom whose legend key is a coloured line, not a filled box.

    ``geom_density`` inherits ``geom_polygon``'s legend drawer, which always
    draws a filled rectangle. Borrowing ``geom_path``'s drawer makes the key a
    short coloured line segment that matches the curve it labels.
    """

    draw_legend = staticmethod(geom_path.draw_legend)


def explore_distribution(data, x, group=None, kind="density", bins=30,
                         alpha=None, legend_inside=False, title=None):
    """Plot the distribution of a numeric variable, optionally split by a group.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    x : str
        Name of the numeric column to display.
    group : str, optional
        A grouping column mapped to colour/fill. A numeric code counts as a
        group, but a numeric column with more than 20 distinct values raises
        ``ValueError``. Rows with a missing group are dropped with a warning,
        and categories without rows are left out of the plot and its palette.
    kind : {"density", "histogram", "both"}
        What to draw.
    bins : int
        Number of histogram bins.
    alpha : float, optional
        Fill transparency, useful when groups overlap. ``None``, the default,
        fills a grouped density at 0.2, light enough for overlapping curves to
        stay distinct, and everything else at 0.6. The R twin's default is 0.6
        throughout. Over a histogram (``kind="both"``), a grouped density is
        drawn as an outline whatever ``alpha`` is, and the bars take the
        transparency.
    legend_inside : bool
        When ``True`` and a ``group`` is given, place the legend inside the
        top-right of the panel (a unimodal distribution leaves it empty) rather
        than in a right-hand margin.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> ld = dp.lexical_decision()
    >>> p = dp.explore_distribution(ld, "RT", group="condition", kind="both")
    """
    if x not in data.columns:
        raise KeyError(f"{x!r} is not a column of `data`.")
    # The grouped density's light default is kept apart from the 0.6 of every
    # other fill, so that an explicit alpha reaches it too. It was fixed at 0.2.
    density_alpha = 0.2 if alpha is None else alpha
    alpha = 0.6 if alpha is None else alpha
    if group:
        # Rows missing x are not drawn, so they are left out before the groups
        # are formed, and the levels are those of the rows drawn.
        data, _, palette = _prepare_group(data.dropna(subset=[x]), group)
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
            # A light fill (0.2 by default) keeps overlapping curves
            # distinguishable without the muddy overlap a heavier fill makes;
            # the coloured outline carries the comparison. For "both" the
            # density fill is dropped so it does not muddy the histogram
            # beneath it.
            p = p + _geom_density_linekey(
                alpha=0 if kind == "both" else density_alpha)
        else:
            p = p + geom_density(alpha=alpha, color=BRAND, fill=BRAND)
        # The density baseline sits at 0; drop the default lower pad so the
        # curves are flush with the x-axis, keeping a little headroom on top.
        p = p + scale_y_continuous(expand=(0, 0, 0.05, 0))
    if group:
        p = p + _group_scales(palette)
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
        A second categorical column mapped to fill, drawn side by side. A
        numeric code counts as a group, but a numeric column with more than 20
        distinct values raises ``ValueError``. Rows with a missing group are
        dropped with a warning, and categories without rows are left out of
        the plot and its palette.
    proportion : bool
        Show proportions within each group rather than raw counts.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.explore_categorical(wb, "education", group="region")
    """
    if group:
        # Rows missing x are not counted, so they are left out before the
        # groups are formed, and the levels are those of the rows counted.
        data, _, palette = _prepare_group(data.dropna(subset=[x]), group)
        counts = (data.groupby([group, x], observed=True).size()
                  .rename("n").reset_index())
        if proportion:
            group_total = counts.groupby(group, observed=True)["n"].transform("sum")
            counts["value"] = counts["n"] / group_total
        else:
            counts["value"] = counts["n"]
        p = (ggplot(counts, aes(x=x, y="value", fill=group))
             + geom_bar(stat="identity", position="dodge")
             + _group_scales(palette, ("fill",)))
        y_lab = "Proportion" if proportion else "Count"
    else:
        p = ggplot(data, aes(x=x)) + geom_bar(fill=BRAND)
        y_lab = "Count"
    return p + labs(x=x, y=y_lab, title=title) + theme_depictr()


def correlation_heatmap(data, cols=None, title=None):
    """Heatmap of pairwise Pearson correlations among numeric columns.

    Each cell is labelled with its correlation, drawn in black or white,
    whichever has the higher contrast against the cell.

    Parameters
    ----------
    data : pandas.DataFrame
    cols : list of str, optional
        Columns to include; defaults to all numeric columns. A named column
        that is not numeric or boolean, such as text or a datetime, raises
        ``TypeError``.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
        Zero-variance columns are dropped with a warning, and any correlation
        that remains undefined is labelled ``n/a`` rather than left blank.

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.correlation_heatmap(wb)
    """
    num = data[cols] if cols else data.select_dtypes("number")
    # Text failed inside pandas with "could not convert string to float", and a
    # datetime was correlated as the integers that store it. The wording is the
    # R twin's, as in its explore_pairs().
    not_numeric = [c for c in num.columns if not is_numeric_dtype(num[c])]
    if not_numeric:
        raise TypeError(
            "correlation_heatmap() needs numeric columns; these are not: "
            f"{', '.join(str(c) for c in not_numeric)}."
        )
    # A correlation is undefined for a constant column, so pandas fills the
    # whole row and column with NaN, which the label formatter then rendered as
    # the literal string "nan". Drop those columns and say so.
    const = [c for c in num.columns if num[c].dropna().nunique() < 2]
    if const:
        warnings.warn(
            "correlation_heatmap(): dropping zero-variance column(s): "
            f"{', '.join(str(c) for c in const)}.",
            UserWarning,
            stacklevel=2,
        )
        num = num.drop(columns=const)
    corr = num.corr()
    order = list(corr.columns)
    long = corr.reset_index().melt(id_vars="index", var_name="var2",
                                   value_name="r").rename(columns={"index": "var1"})
    long["var1"] = pd.Categorical(long["var1"], categories=order, ordered=True)
    long["var2"] = pd.Categorical(long["var2"], categories=order[::-1], ordered=True)
    # Pairwise-complete overlap too small to correlate still leaves an NaN cell
    # after the constant columns have gone, so label it rather than print "nan".
    long["label"] = long["r"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    fill_scale = scale_fill_gradientn(
        colors=depictr_palette(7, kind="diverging"), limits=(-1, 1),
        name="Correlation",
    )
    # Each label is black or white, whichever contrasts more with its tile as
    # the fill scale paints it, the scale's grey for an undefined correlation
    # included. The r = 1 diagonal is the darkest tile on the ramp, where the
    # near-black every label used to be drawn in sat at 2.4:1.
    long["label_colour"] = _legible_text_colour(
        _scale_fills(fill_scale, long["r"]))
    return (
        ggplot(long, aes(x="var1", y="var2", fill="r"))
        + geom_tile(color="white")
        + geom_text(aes(label="label", color="label_colour"), size=8)
        + scale_color_identity()
        + fill_scale
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

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.missingness_map(wb, legend_inside=True)
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
