"""Smoke tests: the multivariate plots build a figure without error."""

import pytest
from plotnine import ggplot

from depictr.data import wellbeing_survey
from depictr.multivariate import (
    cluster_plot,
    dendrogram_plot,
    pca_plot,
    scree_plot,
    silhouette_plot,
)

WB = wellbeing_survey()
NUM = ["age", "income", "stress", "sleep_hours", "life_satisfaction"]


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


def test_pca_plot_with_and_without_group():
    assert _builds(pca_plot(WB, cols=NUM))
    assert _builds(pca_plot(WB, cols=NUM, group="region", title="PCA"))


def test_pca_plot_default_numeric_columns():
    assert _builds(pca_plot(WB))


def test_scree_plot():
    assert _builds(scree_plot(WB, cols=NUM, title="Scree"))


def test_cluster_plot():
    assert _builds(cluster_plot(WB, cols=NUM, k=3, title="Clusters"))
    assert _builds(cluster_plot(WB, cols=NUM, k=4))


def test_dendrogram_plot_methods():
    assert _builds(dendrogram_plot(WB, cols=NUM, method="ward"))
    assert _builds(dendrogram_plot(WB, cols=NUM, method="average", title="Tree"))


def test_silhouette_plot():
    assert _builds(silhouette_plot(WB, cols=NUM, k=3, title="Silhouette"))


def test_needs_two_numeric_columns():
    with pytest.raises(ValueError):
        pca_plot(WB, cols=["age"])
