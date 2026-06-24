"""Smoke tests: the extra EDA plots and the power curve build without error."""

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

import depictr as dp
from depictr.eda_extra import explore_pairs, raincloud_plot
from depictr.power import power_curve_plot

WB = dp.wellbeing_survey()
LD = dp.lexical_decision()
NUM = ["age", "income", "stress", "sleep_hours", "life_satisfaction"]


def _builds(p):
    # Single plots are ggplots; explore_pairs returns an arrange_plots
    # composition, which is not a ggplot but still draws.
    p.draw(show=False)
    return True


def _power_frame():
    ns = np.arange(10, 210, 20)
    return pd.DataFrame({"n": ns, "power": 1 / (1 + np.exp(-(ns - 90) / 25))})


def test_raincloud_grouped():
    p = raincloud_plot(LD, "RT", group="condition", title="RT")
    assert isinstance(p, ggplot)
    assert _builds(p)
    # Four-level group on the wellbeing survey.
    assert _builds(raincloud_plot(WB, "life_satisfaction", group="region"))


def test_raincloud_ungrouped():
    p = raincloud_plot(WB, "income")
    assert isinstance(p, ggplot)
    assert _builds(p)


def test_raincloud_bad_columns():
    with pytest.raises(KeyError):
        raincloud_plot(LD, "not_a_column")
    with pytest.raises(KeyError):
        raincloud_plot(LD, "RT", group="not_a_column")


def test_explore_pairs_default_and_explicit():
    assert _builds(explore_pairs(WB, cols=NUM))
    assert _builds(explore_pairs(LD))


def test_explore_pairs_caps_columns():
    # Six columns requested, but the matrix is capped at five.
    six = ["age", "income", "stress", "sleep_hours", "life_satisfaction", "age"]
    assert _builds(explore_pairs(WB, cols=six))


def test_explore_pairs_needs_two_columns():
    with pytest.raises(ValueError):
        explore_pairs(WB, cols=["age"])


def test_explore_pairs_bad_column():
    with pytest.raises(KeyError):
        explore_pairs(WB, cols=["age", "not_a_column"])


def test_power_curve_ungrouped():
    p = power_curve_plot(_power_frame())
    assert isinstance(p, ggplot)
    assert _builds(p)


def test_power_curve_grouped():
    base = _power_frame()
    ns = base["n"]
    grouped = pd.concat([
        base.assign(effect="small", power=1 / (1 + np.exp(-(ns - 130) / 25))),
        base.assign(effect="large", power=1 / (1 + np.exp(-(ns - 60) / 20))),
    ])
    p = power_curve_plot(grouped, group="effect", title="Power by effect size")
    assert isinstance(p, ggplot)
    assert _builds(p)


def test_power_curve_custom_column_names():
    df = _power_frame().rename(columns={"n": "sample", "power": "pwr"})
    assert _builds(power_curve_plot(df, n="sample", power="pwr"))


def test_power_curve_bad_column():
    with pytest.raises(KeyError):
        power_curve_plot(_power_frame(), power="not_a_column")
