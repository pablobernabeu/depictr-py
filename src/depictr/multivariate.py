"""Multivariate plots: PCA, clustering and dendrograms, themed consistently.

The decompositions and clusterings come from the de-facto standard packages --
``scikit-learn`` for PCA, k-means and silhouettes, ``scipy`` for hierarchical
linkage -- and the figures are re-drawn under the depictr theme, so a biplot or a
scree plot sits in the same visual language as the rest of a report. Numeric
columns are standardised (zero mean, unit variance) before any decomposition, so
no single variable dominates through its scale alone.

Install the optional dependency with ``pip install "depictr[classification]"``
(scipy is a core dependency, so the dendrogram needs no extra).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
    coord_fixed,
    coord_flip,
    geom_bar,
    geom_hline,
    geom_line,
    geom_point,
    geom_segment,
    geom_text,
    ggplot,
    labs,
    scale_y_continuous,
)

from .palette import ACCENT, BRAND
from .theme import scale_colour_depictr, scale_fill_depictr, theme_depictr


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The multivariate plots need scikit-learn. Install it with "
            '`pip install "depictr[classification]"`.'
        ) from exc


def _require_scipy():
    try:
        import scipy.cluster.hierarchy  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "dendrogram_plot needs scipy, a core dependency of depictr. "
            "Install it with `pip install scipy`."
        ) from exc


def _numeric_frame(data, cols):
    """Select numeric columns and drop rows with any missing value.

    PCA and clustering have no natural handling of missing data, so rows with an
    ``NA`` in any of the selected columns are dropped (complete-case analysis).
    """
    num = data[cols] if cols else data.select_dtypes("number")
    if num.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for a multivariate plot.")
    return num.dropna()


def _standardise(num):
    """Return the columns centred and scaled to unit variance, as an array.

    A numpy z-score with scikit-learn's ``StandardScaler`` semantics: the
    population (ddof=0) standard deviation, and a scale of 1 substituted where
    the deviation is zero, so a constant column is centred to zeros rather than
    divided by zero. Kept sklearn-free because ``dendrogram_plot`` shares this
    helper, and the dendrogram's contract is that scipy, a core dependency, is
    all it needs.
    """
    X = num.to_numpy(dtype=float)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    return (X - X.mean(axis=0)) / scale


def _nice_label(name):
    """Show underscores in a variable name as spaces, matching the R package."""
    return str(name).replace("_", " ")


#: Loading arrows and their labels: R's grey20, a neutral dark grey outside the
#: qualitative palette, so that no arrow reads as one of the groups.
_LOADING_INK = "#333333"


def pca_plot(data, cols=None, group=None, title=None):
    """PCA biplot: observations on the first two components, with loading arrows.

    Observations are scored on PC1 and PC2 and shown as points, optionally
    coloured by ``group``. Each original variable is drawn as an arrow from the
    origin along its loading, labelled with the variable name just beyond the
    arrow head. The axis labels report the percentage of variance each component
    explains. The columns used are standardised to zero mean and unit variance
    first, so the components are those of their correlation matrix, and rows
    with a missing value in any of them are dropped.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    cols : list of str, optional
        Numeric columns to decompose; defaults to all numeric columns.
    group : str, optional
        A column mapped to the colour of the observation points. Kept aside from
        the decomposition, which uses only the numeric ``cols``.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    Raises
    ------
    KeyError
        If ``group`` is not a column of ``data``.

    Notes
    -----
    Each arrow is a variable's loadings on the two components, multiplied by a
    factor that all the arrows share, as in the R twin. The factor makes the
    largest coordinate of any arrow three quarters of the largest absolute
    score. Both axes use that one factor, as base R's ``biplot()`` uses one
    ratio for both, and the aspect ratio is fixed at 1. An arrow's angle on the
    page is therefore its angle in loading space, whatever the size of the
    figure. The panel's height and width follow the ranges of its two axes, so
    a figure of another shape leaves blank space around the panel. Because the
    scores are drawn as they are and the arrows as loadings, the angle between
    two arrows does not estimate the correlation between their variables.
    Arrows and labels are drawn in a neutral dark grey (``#333333``), which is
    not in the palette, so no arrow reads as part of a group.

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.pca_plot(wb, group="region")
    """
    _require_sklearn()
    from sklearn.decomposition import PCA

    if group is not None and group not in data.columns:
        # The wording of the R twin, which names the argument as well.
        raise KeyError(f"Column {group!r} (argument `group`) not found in the data.")

    num = _numeric_frame(data, cols)
    X = _standardise(num)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    pts = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]})
    if group:
        # Align the kept-aside group to the complete-case rows used above.
        pts[group] = data.loc[num.index, group].to_numpy()

    # One factor for both axes, so that each arrow points along its loading. A
    # factor per axis stretches the arrows unevenly and bends every direction.
    loadings = pca.components_.T  # variables x components
    span = np.abs(scores).max()
    mult = span / np.abs(loadings).max() * 0.75
    end = loadings * mult
    # Each label continues its arrow outward by 8% of the span, clear of the
    # arrow head. An arrow of zero length has no direction and keeps its label
    # at the origin.
    length = np.hypot(end[:, 0], end[:, 1])[:, None]
    unit = np.divide(end, length, out=np.zeros_like(end), where=length > 0)
    label_at = end + unit * span * 0.08
    load = pd.DataFrame({
        "x": 0.0, "y": 0.0,
        "xend": end[:, 0], "yend": end[:, 1],
        "labx": label_at[:, 0], "laby": label_at[:, 1],
        "label": [_nice_label(c) for c in num.columns],
    })

    if group:
        p = (ggplot(pts, aes("PC1", "PC2", color=group))
             + geom_point(alpha=0.7, size=2)
             + scale_colour_depictr())
    else:
        p = ggplot(pts, aes("PC1", "PC2")) + geom_point(alpha=0.6, size=2, color=BRAND)

    return (
        p
        + geom_segment(aes(x="x", y="y", xend="xend", yend="yend"),
                       data=load, color=_LOADING_INK, size=0.7,
                       arrow=_arrow())
        + geom_text(aes(x="labx", y="laby", label="label"), data=load,
                    color=_LOADING_INK, size=9, fontweight="bold")
        # One unit of PC1 spans the same length on the page as one unit of PC2,
        # so the arrows keep their angles on a figure of any shape.
        + coord_fixed(ratio=1)
        + labs(x=f"PC1 ({var[0]:.1f}%)", y=f"PC2 ({var[1]:.1f}%)", title=title)
        + theme_depictr()
    )


def _arrow():
    """A small filled arrowhead for the loading segments."""
    from plotnine import arrow

    return arrow(length=0.12, type="closed")


def scree_plot(data, cols=None, title=None):
    """Scree plot: variance explained per component, with the cumulative line.

    Bars give the proportion of variance each principal component explains; an
    overlaid line and points give the running cumulative proportion. Both share
    the (0, 1) y-axis. The columns used are standardised and rows with a
    missing value in any of them are dropped, as in
    [`pca_plot`][depictr.pca_plot].

    Parameters
    ----------
    data : pandas.DataFrame
    cols : list of str, optional
        Numeric columns to decompose; defaults to all numeric columns.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.scree_plot(wb)
    """
    _require_sklearn()
    from sklearn.decomposition import PCA

    num = _numeric_frame(data, cols)
    X = _standardise(num)
    pca = PCA().fit(X)
    var = pca.explained_variance_ratio_
    df = pd.DataFrame({
        "component": np.arange(1, len(var) + 1),
        "variance": var,
        "cumulative": np.cumsum(var),
    })
    df["component"] = pd.Categorical(df["component"], ordered=True)

    return (
        ggplot(df, aes(x="component"))
        + geom_bar(aes(y="variance"), stat="identity", fill=BRAND, width=0.7)
        + geom_line(aes(y="cumulative", group=1), color=ACCENT, size=0.9)
        + geom_point(aes(y="cumulative"), color=ACCENT, size=2.4)
        + annotate("text", x=len(var), y=1.0, ha="right", va="bottom",
                   label="Cumulative", color=ACCENT, fontweight="bold")
        + scale_y_continuous(limits=(0, 1))
        + labs(x="Principal component", y="Proportion of variance", title=title)
        + theme_depictr()
    )


def cluster_plot(data, cols=None, k=3, title=None, clusters=None):
    """k-means clusters drawn on the first two principal components.

    k-means is run on the standardised data in all of its dimensions, as the
    R twin's ``cluster_plot()`` runs it by default, and with the settings
    [`silhouette_plot`][depictr.silhouette_plot] uses, so the two functions
    show the same partition for the same ``k``. The principal components
    serve only to display it. Points are drawn at their PC1 and PC2 scores,
    coloured by cluster, and each centroid is marked by a larger outlined
    point at the projection of its k-means centre. The axis labels report the
    percentage of variance each component explains.

    Parameters
    ----------
    data : pandas.DataFrame
    cols : list of str, optional
        Numeric columns to use; defaults to all numeric columns.
    k : int
        Number of clusters. Ignored when ``clusters`` is supplied.
    title : str, optional
    clusters : array-like, optional
        A clustering computed elsewhere, such as the labels of a hierarchical
        or PAM clustering. When it is given, it is drawn and k-means is not
        run. It needs one entry per row of ``data``. Rows with a missing value
        in ``cols`` are dropped from both, and the labels are kept as they are.

    Returns
    -------
    plotnine.ggplot
        The cluster of each complete-case row is attached as ``p.clusters``, a
        ``pandas.Series`` on the index of ``data``. k-means clusters are
        numbered from 1.

    Raises
    ------
    ValueError
        If ``clusters`` does not have one entry per row of ``data``.

    Notes
    -----
    The two displayed components hold only part of the variance, so k-means
    run on them would partition a different space from the one the silhouette
    widths describe. A k-means centroid is the cluster's mean in the full
    space, projected, which equals the mean of its members' scores. Supplied
    clusters are marked at the mean of their members' scores.

    k-means settles in a local optimum. scikit-learn's algorithm and random
    starts differ from those of R's ``kmeans()``, so on data without clear
    clusters the two twins can still assign some points differently for the
    same ``k``.

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.cluster_plot(wb, k=3)
    >>> p.clusters.value_counts().sort_index().index.tolist()
    [1, 2, 3]
    """
    _require_sklearn()
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    if clusters is not None and len(clusters) != len(data):
        # Checked against every row before the incomplete ones are dropped, so
        # labels sized to the complete rows cannot be misaligned in silence.
        # The R twin's wording.
        raise ValueError("`clusters` must have one entry per row of `data`.")

    num = _numeric_frame(data, cols)
    X = _standardise(num)
    pca = PCA(n_components=2).fit(X)
    scores = pca.transform(X)
    var = pca.explained_variance_ratio_ * 100

    if clusters is None:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        labels = pd.Series(km.labels_ + 1, index=num.index, name="cluster")
        centres = pca.transform(km.cluster_centers_)
        cent = pd.DataFrame(centres, columns=["PC1", "PC2"])
        cent["cluster"] = np.arange(1, k + 1)
    else:
        # The complete-case rows by position, so a repeated index cannot
        # misalign the labels.
        kept = data[num.columns].notna().all(axis=1).to_numpy()
        labels = pd.Series(np.asarray(clusters)[kept], index=num.index, name="cluster")
        cent = (pd.DataFrame(scores, columns=["PC1", "PC2"])
                .groupby(labels.to_numpy(), sort=True).mean()
                .rename_axis("cluster").reset_index())
    levels = pd.Categorical(labels.to_numpy()).categories
    cent["cluster"] = pd.Categorical(cent["cluster"], categories=levels)

    pts = pd.DataFrame({
        "PC1": scores[:, 0], "PC2": scores[:, 1],
        "cluster": pd.Categorical(labels.to_numpy(), categories=levels),
    })

    p = (
        ggplot(pts, aes("PC1", "PC2", color="cluster"))
        + geom_point(alpha=0.7, size=2)
        + geom_point(data=cent, size=6, shape="X", color="#1a1a1a")
        + geom_point(data=cent, size=4, shape="X")
        + scale_colour_depictr(n=len(levels), name="Cluster")
        + labs(x=f"PC1 ({var[0]:.1f}%)", y=f"PC2 ({var[1]:.1f}%)", title=title)
        + theme_depictr()
    )
    p.clusters = labels
    return p


def dendrogram_plot(data, cols=None, method="ward", title=None):
    """Hierarchical-clustering dendrogram drawn from a scipy linkage.

    The linkage is computed by scipy and the dendrogram coordinates it returns
    are redrawn as line segments under the depictr theme, so the tree matches the
    rest of the figure set. Leaves are not labelled, since the intent is to read
    the overall merge structure rather than individual observations. The
    columns used are standardised, rows with a missing value in any of them
    are dropped and the linkage is computed on Euclidean distances.

    Parameters
    ----------
    data : pandas.DataFrame
    cols : list of str, optional
        Numeric columns to cluster on; defaults to all numeric columns.
    method : str
        Linkage method passed to ``scipy.cluster.hierarchy.linkage``
        (for example ``"ward"``, ``"average"``, ``"complete"``).
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.dendrogram_plot(wb.groupby("region", observed=True).mean(numeric_only=True))
    """
    _require_scipy()
    from scipy.cluster.hierarchy import dendrogram, linkage

    num = _numeric_frame(data, cols)
    X = _standardise(num)
    Z = linkage(X, method=method)
    # no_plot returns the drawing coordinates without touching matplotlib.
    dnd = dendrogram(Z, no_plot=True)

    # Each entry of icoord/dcoord is the four-point bracket joining one merge;
    # turn each bracket into three segments (up, across, down).
    rows = []
    # strict=True: scipy returns one icoord and one dcoord entry per merge, so a
    # length mismatch is impossible by construction and should raise rather than
    # silently drop the tail of the dendrogram.
    for xs, ys in zip(dnd["icoord"], dnd["dcoord"], strict=True):
        for i in range(3):
            rows.append({"x": xs[i], "y": ys[i],
                         "xend": xs[i + 1], "yend": ys[i + 1]})
    seg = pd.DataFrame(rows)

    return (
        ggplot(seg, aes(x="x", y="y", xend="xend", yend="yend"))
        + geom_segment(color=BRAND, size=0.6)
        + labs(x="", y="Distance", title=title)
        + theme_depictr(grid="y")
        + scale_y_continuous(expand=(0, 0, 0.05, 0))
    )


def silhouette_plot(data, cols=None, k=3, title=None):
    """Silhouette widths per observation, grouped and ordered by cluster.

    k-means is run on the standardised data and the silhouette width of each
    observation is drawn as a horizontal bar, sorted within its cluster and
    coloured by cluster. A dashed reference line marks the mean silhouette width,
    a quick read on how well-separated the clustering is overall. Rows with a
    missing value in any of the columns used are dropped.

    Parameters
    ----------
    data : pandas.DataFrame
    cols : list of str, optional
        Numeric columns to use; defaults to all numeric columns.
    k : int
        Number of clusters.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.silhouette_plot(wb, k=3)
    """
    _require_sklearn()
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples

    num = _numeric_frame(data, cols)
    X = _standardise(num)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    widths = silhouette_samples(X, labels)

    df = pd.DataFrame({"cluster": labels + 1, "width": widths})
    # Order bars by cluster then by ascending width within cluster, and give
    # each its own row position up the axis.
    df = df.sort_values(["cluster", "width"]).reset_index(drop=True)
    df["position"] = np.arange(len(df))
    df["cluster"] = pd.Categorical(df["cluster"])
    mean_width = float(widths.mean())

    return (
        ggplot(df, aes(x="position", y="width", fill="cluster"))
        + geom_bar(stat="identity", width=1.0)
        + geom_hline(yintercept=mean_width, linetype="dashed", color="#1a1a1a")
        + annotate("text", x=0, y=mean_width, ha="left", va="bottom",
                   label=f"Mean = {mean_width:.2f}", color="#1a1a1a",
                   fontweight="bold")
        + scale_fill_depictr(n=k, name="Cluster")
        + coord_flip()
        + labs(x="Observation", y="Silhouette width", title=title)
        + theme_depictr(grid="x")
    )
