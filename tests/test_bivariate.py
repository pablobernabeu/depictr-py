"""Smoke tests: the bivariate plots build a figure without error."""

import pytest
from plotnine import ggplot

from depictr.bivariate import explore_bivariate, scatter_trend
from depictr.data import crop_yield, lexical_decision

CY = crop_yield()
LD = lexical_decision()


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


def test_scatter_trend_with_and_without_group():
    assert _builds(scatter_trend(CY, "fertiliser", "yield"))
    assert _builds(scatter_trend(CY, "fertiliser", "yield", group="treatment"))


def test_scatter_trend_method():
    assert _builds(scatter_trend(LD, "word_frequency", "RT", method="lowess"))


def test_scatter_trend_bad_column():
    with pytest.raises(KeyError):
        scatter_trend(CY, "fertiliser", "not_a_column")


def test_explore_bivariate_numeric_numeric():
    assert _builds(explore_bivariate(CY, "fertiliser", "yield"))


def test_explore_bivariate_categorical_numeric():
    # Both orderings of category/numeric land in the boxplot branch.
    assert _builds(explore_bivariate(CY, "treatment", "yield"))
    assert _builds(explore_bivariate(LD, "RT", "condition"))


def test_explore_bivariate_categorical_categorical():
    assert _builds(explore_bivariate(LD, "condition", "modality"))


def test_explore_bivariate_bad_column():
    with pytest.raises(KeyError):
        explore_bivariate(CY, "fertiliser", "not_a_column")
