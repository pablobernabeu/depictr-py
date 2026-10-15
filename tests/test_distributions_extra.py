"""Every grouped-distribution plot builds a figure without error, and the group
comparison draws t intervals for the mean."""

import copy
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import geom_errorbar, ggplot
from plotnine.exceptions import PlotnineWarning
from scipy import stats

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


def test_ridgeline_plot_draws_each_ridge_unstacked():
    # Each facet row holds a single level, so the stacking that geom_area does
    # by default had nothing to stack and took most of the drawing time. Every
    # ridge still rises from zero to its own density.
    from plotnine.positions import position_identity

    p = ridgeline_plot(WB, "life_satisfaction", "region")
    (layer,) = p.layers
    assert isinstance(layer.position, position_identity)
    built = copy.deepcopy(p)
    built._build()
    drawn = built.layers[0].data
    assert (drawn["ymin"] == 0).all()
    assert np.allclose(drawn["ymax"], drawn["density"])


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


def _dumbbell_order(p):
    """The two segment ends, the category order and each level's point colour.

    A drawn point is matched to its level through the value it sits at, read
    from the one-row-per-category table the segments are drawn from.
    """
    category = p.mapping["y"]
    mapping = p.layers[0].mapping
    ends = (mapping["x"], mapping["xend"])
    categories = list(p.data[category].cat.categories)
    wide = p.data.set_index(category)
    built = copy.deepcopy(p)
    built._build()
    colours = {}
    for _, point in built.layers[1].data.iterrows():
        row = categories[int(round(point["y"])) - 1]
        for level in ends:
            if np.isclose(wide.loc[row, level], point["x"]):
                colours.setdefault(level, set()).add(point["color"])
    assert all(len(found) == 1 for found in colours.values())
    return ends, categories, {k: found.pop() for k, found in colours.items()}


def test_dumbbell_plot_orders_its_levels_as_r_does_not_by_first_row():
    # The two levels are the two ends of every segment, and the first of them
    # orders the categories. They were taken in order of first appearance, so
    # moving one row to the top reordered the whole figure. As in the R twin,
    # a categorical keeps its declared order and anything else is sorted.
    wb = WB.assign(age_group=np.where(WB["age"] < 50, "under 50", "50 or over"))
    first_other = wb.index[wb["age_group"] != wb["age_group"].iloc[0]][0]
    moved = pd.concat([wb.loc[[first_other]], wb.drop(index=first_other)])
    a = _dumbbell_order(dumbbell_plot(wb, "region", "life_satisfaction",
                                      "age_group"))
    b = _dumbbell_order(dumbbell_plot(moved, "region", "life_satisfaction",
                                      "age_group"))
    assert a[0] == ("50 or over", "under 50")
    assert a == b
    # The first level takes the first colour, the brand blue, as in R.
    assert a[2]["50 or over"] == dp.depictr_palette(2)[0]

    declared = wb.assign(age_group=pd.Categorical(
        wb["age_group"], categories=["under 50", "50 or over", "unused"]))
    c = _dumbbell_order(dumbbell_plot(declared, "region", "life_satisfaction",
                                      "age_group"))
    assert c[0] == ("under 50", "50 or over")
    assert c[2] == {"under 50": dp.depictr_palette(2)[0],
                    "50 or over": dp.depictr_palette(2)[1]}


def test_outlier_plot():
    assert _builds(outlier_plot(WB, "income"))
    assert _builds(outlier_plot(LD, "RT"))


def test_group_comparison_plot():
    assert _builds(group_comparison_plot(LD, "RT", "condition"))
    assert _builds(group_comparison_plot(WB, "life_satisfaction", "region"))


def _interval_rows(p):
    """The summary table drawn by the error-bar layer, indexed by group."""
    geom = next(lyr.geom for lyr in p.layers if isinstance(lyr.geom, geom_errorbar))
    return geom.data.set_index(geom.data.columns[0])


@pytest.mark.parametrize("conf_level", [0.95, 0.9])
def test_group_comparison_plot_uses_t_intervals(conf_level):
    # A five-observation group is where the old mean +/- 1.96 SE fell short:
    # it covers about 88% of the time for a nominal 95%. The interval must be
    # scipy's t interval with n - 1 degrees of freedom, as in the R twin.
    d = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                      "g": ["a"] * 5 + ["b"] * 5})
    # The default is checked as the default, without naming it.
    kw = {} if conf_level == 0.95 else {"conf_level": conf_level}
    p = group_comparison_plot(d, "y", "g", **kw)
    rows = _interval_rows(p)
    for g, v in d.groupby("g")["y"]:
        lo, hi = stats.t.interval(conf_level, len(v) - 1, loc=v.mean(),
                                  scale=stats.sem(v))
        assert rows.loc[g, "lower"] == pytest.approx(lo)
        assert rows.loc[g, "upper"] == pytest.approx(hi)
    if conf_level == 0.95:
        # The verifier's worked example: [1.037, 4.963], not [1.614, 4.386].
        assert rows.loc["a", "lower"] == pytest.approx(1.0368, abs=1e-4)
        assert rows.loc["a", "upper"] == pytest.approx(4.9632, abs=1e-4)
    assert _builds(p)


def test_group_comparison_plot_single_observation_shows_mean_only():
    d = pd.DataFrame({"y": [1.0, 2.0, 3.0, 7.0], "g": ["a", "a", "a", "b"]})
    msg = ("No confidence interval drawn for group(s) with n < 2: b; "
           "showing the mean only.")
    with pytest.warns(UserWarning, match=re.escape(msg)):
        p = group_comparison_plot(d, "y", "g")
    rows = _interval_rows(p)
    assert np.isnan(rows.loc["b", "lower"]) and np.isnan(rows.loc["b", "upper"])
    assert np.isfinite(rows.loc["a", "lower"])
    # The missing bounds are dropped quietly: depictr's warning is the only one.
    with warnings.catch_warnings():
        warnings.simplefilter("error", PlotnineWarning)
        p.draw(show=False)


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.2])
def test_group_comparison_plot_rejects_bad_conf_level(bad):
    with pytest.raises(ValueError, match="conf_level"):
        group_comparison_plot(LD, "RT", "condition", conf_level=bad)
