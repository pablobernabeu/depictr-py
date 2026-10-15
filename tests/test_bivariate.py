"""Smoke tests: the bivariate plots build a figure without error."""

import copy
import warnings

import matplotlib.pyplot as plt
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


def test_scatter_trend_draws_a_band_only_where_the_smoother_computes_one():
    # The docstring promised a confidence band and suggested "lowess", for
    # which plotnine draws the line alone. It now says which smoothers give one.
    expected = {"lm": True, "glm": True, "gls": True, "mavg": True,
                "lowess": False, "rlm": False}
    for method, band in expected.items():
        p = copy.deepcopy(scatter_trend(CY, "fertiliser", "yield", method=method))
        with warnings.catch_warnings():
            # plotnine says it has no interval for lowess and rlm, and that it
            # chose a window for mavg.
            warnings.simplefilter("ignore")
            figure = p.draw(show=False)
        plt.close(figure)
        smooth = p.layers[1].data
        drawn = "ymin" in smooth and bool(smooth["ymin"].notna().any())
        assert drawn == band, method
    doc = " ".join(scatter_trend.__doc__.replace("``", "").split())
    assert '"lowess" and "rlm" draw the line alone' in doc


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
