"""The multivariate plots build a figure, and the PCA biplot draws its loadings faithfully."""

import io
import warnings

import numpy as np
import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_hex
from plotnine import coord_fixed, ggplot, theme
from plotnine.exceptions import PlotnineWarning

from depictr.data import wellbeing_survey
from depictr.multivariate import (
    _standardise,
    cluster_plot,
    dendrogram_plot,
    pca_plot,
    scree_plot,
    silhouette_plot,
)
from depictr.palette import depictr_palette

# scikit-learn is the optional [classification] extra rather than a core
# dependency, so the tests of the sklearn-backed plots (PCA, k-means and the
# silhouettes) each skip when it is absent, as the rest of the suite does for
# its optional back-ends. The dendrogram is deliberately not among them: its
# documented contract is that scipy, a core dependency, is all it needs, so its
# tests run everywhere, including the minimum-versions CI job that installs the
# core floors alone.

WB = wellbeing_survey()
NUM = ["age", "income", "stress", "sleep_hours", "life_satisfaction"]


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


def test_pca_plot_with_and_without_group():
    pytest.importorskip("sklearn", reason="PCA delegates to scikit-learn")
    assert _builds(pca_plot(WB, cols=NUM))
    assert _builds(pca_plot(WB, cols=NUM, group="region", title="PCA"))


def test_pca_plot_default_numeric_columns():
    pytest.importorskip("sklearn", reason="PCA delegates to scikit-learn")
    assert _builds(pca_plot(WB))


def _pca(frame):
    """Scores and unit loadings (one row per variable) on PC1 and PC2."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    scores = pca.fit_transform(_standardise(frame))
    return scores, pca.components_.T


def _draw_laid_out(p, size=(7, 4.5)):
    """Draw `p` at `size` inches and return its panel axes in final position.

    plotnine lays the panel out when the figure is rendered, so the figure is
    saved to a buffer first. Only then does the data-to-display transform carry
    the panel's final shape.
    """
    fig = (p + theme(figure_size=size)).draw(show=False)
    fig.savefig(io.BytesIO(), format="png")
    return fig.axes[0]


def _drawn_arrows(ax):
    """The loading arrows as drawn: an array of (variable, start/end, x/y)."""
    (arrows,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    return arrows, np.array(arrows.get_segments())


def _angles(seg):
    return np.arctan2(seg[:, 1, 1] - seg[:, 0, 1], seg[:, 1, 0] - seg[:, 0, 0])


def test_pca_plot_arrows_point_along_the_loadings():
    # The arrows were once scaled by a separate factor on each axis (8.43 for
    # PC1 and 3.81 for PC2 on the wellbeing_survey() of 0.3.0), which bent them
    # by up to 22 degrees. One factor for both axes keeps each arrow on its
    # loading's direction.
    pytest.importorskip("sklearn", reason="PCA delegates to scikit-learn")
    scores, load = _pca(WB[NUM].dropna())
    _, seg = _drawn_arrows(_draw_laid_out(pca_plot(WB, cols=NUM, group="region")))
    np.testing.assert_allclose(
        _angles(seg), np.arctan2(load[:, 1], load[:, 0]), rtol=0, atol=1e-9
    )
    # The factor is the R twin's: the largest coordinate of any arrow is three
    # quarters of the largest absolute score.
    mult = 0.75 * np.abs(scores).max() / np.abs(load).max()
    np.testing.assert_allclose(seg[:, 1, :], load * mult, rtol=0, atol=1e-9)


def test_pca_plot_fixes_the_aspect_ratio():
    # With a free aspect ratio, the device stretches one axis against the other
    # and the arrows' angles on the page drift from their angles in the data.
    pytest.importorskip("sklearn", reason="PCA delegates to scikit-learn")
    p = pca_plot(WB, cols=NUM, group="region")
    assert isinstance(p.coordinates, coord_fixed)
    assert p.coordinates.ratio == 1
    for size in [(8, 4), (4, 8)]:
        ax = _draw_laid_out(p, size)
        _, seg = _drawn_arrows(ax)
        on_page = np.array([ax.transData.transform(s) for s in seg])
        np.testing.assert_allclose(_angles(on_page), _angles(seg), rtol=0, atol=1e-9)


def test_pca_plot_labels_sit_beyond_their_arrows():
    # Each label continues its own arrow outward by 8% of the largest absolute
    # score, as in the R twin, so it names the arrow it sits on.
    pytest.importorskip("sklearn", reason="PCA delegates to scikit-learn")
    scores, load = _pca(WB[NUM].dropna())
    ax = _draw_laid_out(pca_plot(WB, cols=NUM))
    _, seg = _drawn_arrows(ax)
    tip = seg[:, 1, :]
    at = {t.get_text(): np.asarray(t.get_position(), dtype=float) for t in ax.texts}
    labels = np.array([at[name.replace("_", " ")] for name in NUM])
    np.testing.assert_allclose(
        np.arctan2(labels[:, 1], labels[:, 0]), np.arctan2(load[:, 1], load[:, 0]),
        rtol=0, atol=1e-9,
    )
    np.testing.assert_allclose(
        np.hypot(*labels.T) - np.hypot(*tip.T), 0.08 * np.abs(scores).max(), rtol=1e-9
    )


def test_pca_plot_zero_length_arrow_keeps_its_label_at_the_origin():
    # A constant column loads on neither component, so its arrow has no
    # direction to continue. Dividing by the arrow's length would put the label
    # at NaN, and plotnine would drop it with a warning.
    pytest.importorskip("sklearn", reason="PCA delegates to scikit-learn")
    frame = WB[NUM].dropna().assign(constant=7.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", PlotnineWarning)
        ax = _draw_laid_out(pca_plot(frame))
    _, seg = _drawn_arrows(ax)
    at = {t.get_text(): np.asarray(t.get_position(), dtype=float) for t in ax.texts}
    np.testing.assert_array_equal(seg[-1, 1, :], [0.0, 0.0])
    np.testing.assert_array_equal(at["constant"], [0.0, 0.0])


def test_pca_plot_loadings_are_neutral_grey():
    # The arrows used to take the accent, which is also the second group colour,
    # so they read as belonging to that group.
    pytest.importorskip("sklearn", reason="PCA delegates to scikit-learn")
    ax = _draw_laid_out(pca_plot(WB, cols=NUM, group="region"))
    arrows, _ = _drawn_arrows(ax)
    arrow_colours = {to_hex(c) for c in arrows.get_edgecolor()}
    label_colours = {to_hex(t.get_color()) for t in ax.texts}
    assert arrow_colours == label_colours == {"#333333"}
    (points,) = [c for c in ax.collections
                 if isinstance(c, PathCollection) and len(c.get_offsets()) > len(NUM)]
    group_colours = {to_hex(c) for c in points.get_facecolor()}
    assert len(group_colours) == WB["region"].nunique()
    assert "#333333" not in group_colours | set(depictr_palette())


def test_pca_plot_mistyped_group_names_the_column():
    pytest.importorskip("sklearn", reason="PCA delegates to scikit-learn")
    with pytest.raises(
        KeyError, match=r"Column 'regoin' \(argument `group`\) not found in the data\."
    ):
        pca_plot(WB, cols=NUM, group="regoin")


def test_scree_plot():
    pytest.importorskip("sklearn", reason="the scree plot delegates to scikit-learn")
    assert _builds(scree_plot(WB, cols=NUM, title="Scree"))


def test_cluster_plot():
    pytest.importorskip("sklearn", reason="k-means delegates to scikit-learn")
    assert _builds(cluster_plot(WB, cols=NUM, k=3, title="Clusters"))
    assert _builds(cluster_plot(WB, cols=NUM, k=4))


def test_dendrogram_plot_methods():
    assert _builds(dendrogram_plot(WB, cols=NUM, method="ward"))
    assert _builds(dendrogram_plot(WB, cols=NUM, method="average", title="Tree"))


def test_dendrogram_plot_needs_no_sklearn(monkeypatch):
    # The documented contract: scipy is a core dependency, so the dendrogram
    # needs no extra. Standardisation must therefore stay sklearn-free, however
    # the sklearn-backed plots standardise.
    import builtins

    real_import = builtins.__import__

    def no_sklearn(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sklearn)
    assert _builds(dendrogram_plot(WB, cols=NUM))


def test_standardise_matches_standardscaler():
    # The z-score must reproduce StandardScaler, including its population
    # (ddof=0) deviation and the scale of 1 it substitutes for a zero-variance
    # column, so the sklearn-backed plots see the same input as before.
    pytest.importorskip("sklearn", reason="the comparison target is scikit-learn")
    from sklearn.preprocessing import StandardScaler

    frame = WB[NUM].dropna().assign(constant=7.0)
    expected = StandardScaler().fit_transform(frame.to_numpy(dtype=float))
    np.testing.assert_allclose(_standardise(frame), expected, atol=1e-12)


def test_silhouette_plot():
    pytest.importorskip("sklearn", reason="silhouettes delegate to scikit-learn")
    assert _builds(silhouette_plot(WB, cols=NUM, k=3, title="Silhouette"))


def test_needs_two_numeric_columns():
    with pytest.raises(ValueError):
        dendrogram_plot(WB, cols=["age"])
