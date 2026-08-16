"""Smoke tests: the multivariate plots build a figure without error."""

import numpy as np
import pytest
from plotnine import ggplot

from depictr.data import wellbeing_survey
from depictr.multivariate import (
    _standardise,
    cluster_plot,
    dendrogram_plot,
    pca_plot,
    scree_plot,
    silhouette_plot,
)

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
