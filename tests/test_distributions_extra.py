"""Smoke tests: every grouped-distribution plot builds a figure without error."""

import pandas as pd
import pytest
from plotnine import ggplot

import depictr as dp
from depictr.distributions_extra import (
    dumbbell_plot,
    ecdf_plot,
    group_comparison_plot,
    outlier_plot,
    ridgeline_plot,
)

WB = dp.wellbeing_survey()
LD = dp.lexical_decision()


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


def test_ecdf_plot():
    assert _builds(ecdf_plot(LD, "RT", group="condition"))
    assert _builds(ecdf_plot(WB, "income"))


def test_ecdf_plot_bad_column():
    with pytest.raises(KeyError):
        ecdf_plot(LD, "not_a_column")


def test_ridgeline_plot():
    assert _builds(ridgeline_plot(WB, "life_satisfaction", "region"))
    assert _builds(ridgeline_plot(LD, "RT", "condition"))


def test_dumbbell_plot():
    # Two-level groups in each dataset: modality in LD, a derived split in WB.
    assert _builds(dumbbell_plot(LD, "modality", "RT", "condition"))
    wb = WB.assign(young=pd.cut(WB["age"], bins=[0, 40, 200],
                               labels=["under 40", "40 and over"]))
    assert _builds(dumbbell_plot(wb, "region", "life_satisfaction", "young"))


def test_dumbbell_plot_needs_two_levels():
    with pytest.raises(ValueError):
        dumbbell_plot(WB, "education", "income", "region")


def test_dumbbell_plot_rejects_same_category_and_group():
    with pytest.raises(ValueError, match="different columns"):
        dumbbell_plot(LD, "condition", "RT", "condition")


def test_outlier_plot():
    assert _builds(outlier_plot(WB, "income"))
    assert _builds(outlier_plot(LD, "RT"))


def test_group_comparison_plot():
    assert _builds(group_comparison_plot(LD, "RT", "condition"))
    assert _builds(group_comparison_plot(WB, "life_satisfaction", "region"))
