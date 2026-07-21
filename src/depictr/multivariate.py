"""Multivariate plots: PCA, clustering and dendrograms, themed consistently.

The decompositions and clusterings come from the de-facto standard packages --
``scikit-learn`` for PCA, k-means and silhouettes, ``scipy`` for hierarchical
linkage -- and the figures are re-drawn under the depictr theme, so a biplot or a
scree plot sits in the same visual language as the rest of a report. Numeric
columns are standardised (zero mean, unit variance) before any decomposition, so
no single variable dominates through its scale alone.

Install the optional dependency with ``pip install depictr[classification]``
(scipy is a core dependency, so the dendrogram needs no extra).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
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
            "`pip install depictr[classification]`."
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
    """Return the columns centred and scaled to unit variance, as an array."""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(num.to_numpy(dtype=float))


def _nice_label(name):
    """Show underscores in a variable name as spaces, matching the R package."""
    return str(name).replace("_", " ")


def pca_plot(data, cols=None, group=None, title=None):
    """PCA biplot: observations on the first two components, with loading arrows.

    Observations are scored on PC1 and PC2 and shown as points, optionally
    coloured by ``group``. Each original variable is drawn as an arrow from the
    origin in the direction of its loading, labelled with the variable name. The
    axis labels report the percentage of variance each component explains.

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

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.pca_plot(wb, group="region")
    """
    _require_sklearn()
    from sklearn.decomposition import PCA

    num = _numeric_frame(data, cols)
    X = _standardise(num)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    pts = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]})
    if group:
        # Align the kept-aside group to the complete-case rows used above.
        pts[group] = data.loc[num.index, group].to_numpy()

    # Loadings, scaled to the span of the scores so the arrows sit among the
    # points rather than near the origin (a standard biplot rescaling).
    loadings = pca.components_.T  # variables x components
    span = np.abs(scores).max(axis=0)
    arrow_len = np.abs(loadings).max(axis=0)
    scale = span / arrow_len * 0.9
    load = pd.DataFrame({
        "x": 0.0, "y": 0.0,
        "xend": loadings[:, 0] * scale[0],
        "yend": loadings[:, 1] * scale[1],
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
                       data=load, color=ACCENT, size=0.7,
                       arrow=_arrow())
        + geom_text(aes(x="xend", y="yend", label="label"), data=load,
                    color=ACCENT, size=9, fontweight="bold",
                    nudge_y=span[1] * 0.04)
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
    the (0, 1) y-axis.

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


def cluster_plot(data, cols=None, k=3, title=None):
    """k-means clusters drawn on the first two principal components.

    The data are reduced to two principal components for display, k-means is run
    in that two-dimensional space, and points are coloured by cluster with each
    cluster centroid marked by a larger outlined point.

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
    >>> p = dp.cluster_plot(wb, k=3)
    """
    _require_sklearn()
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    num = _numeric_frame(data, cols)
    X = _standardise(num)
    scores = PCA(n_components=2).fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(scores)

    pts = pd.DataFrame({
        "PC1": scores[:, 0], "PC2": scores[:, 1],
        "cluster": pd.Categorical(km.labels_ + 1),
    })
    cent = pd.DataFrame(km.cluster_centers_, columns=["PC1", "PC2"])
    cent["cluster"] = pd.Categorical(np.arange(1, k + 1))

    return (
        ggplot(pts, aes("PC1", "PC2", color="cluster"))
        + geom_point(alpha=0.7, size=2)
        + geom_point(data=cent, size=6, shape="X", color="#1a1a1a")
        + geom_point(data=cent, size=4, shape="X")
        + scale_colour_depictr(n=k, name="Cluster")
        + labs(x="PC1", y="PC2", title=title)
        + theme_depictr()
    )


def dendrogram_plot(data, cols=None, method="ward", title=None):
    """Hierarchical-clustering dendrogram drawn from a scipy linkage.

    The linkage is computed by scipy and the dendrogram coordinates it returns
    are redrawn as line segments under the depictr theme, so the tree matches the
    rest of the figure set. Leaves are not labelled, since the intent is to read
    the overall merge structure rather than individual observations.

    Parameters
    ----------
    data : pandas.DataFrame
    cols : list of str, optional
        Numeric columns to cluster on; defaults to all numeric columns.
    method : str
        Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`
        (for example ``"ward"``, ``"average"``, ``"complete"``).
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> p = dp.dendrogram_plot(wb.groupby("region").mean(numeric_only=True))
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
    for xs, ys in zip(dnd["icoord"], dnd["dcoord"]):
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
    a quick read on how well-separated the clustering is overall.

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
