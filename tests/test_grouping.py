"""One rule for the grouping variable of every grouped plot.

A numeric code becomes a group, and a numeric column with more than 20 distinct
values is refused as continuous. Levels without data leave the legend and the
palette, and rows with a missing group are dropped with a warning. The colours
pinned here are pinned in the R twin's tests/testthat/test-group-handling.R as
well, so that a level takes the same colour in both packages. The R twin also
groups `outlier_plot()` and `explore_pairs()`, which take no group here.
"""

import copy
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import depictr as dp
from depictr.palette import NA_VALUE, OKABE_ITO

# Each plot is called on a frame with numeric `x` and `y`, a categorical `cat`
# for explore_categorical() to count and the grouping column `grp`, and draws
# the group in the aesthetics listed.
GROUPED_PLOTS = {
    "scatter_trend": (lambda d: dp.scatter_trend(d, "x", "y", group="grp"),
                      ("color", "fill")),
    "ecdf_plot": (lambda d: dp.ecdf_plot(d, "y", group="grp"), ("color",)),
    "explore_distribution": (
        lambda d: dp.explore_distribution(d, "y", group="grp", kind="both"),
        ("color", "fill")),
    "raincloud_plot": (lambda d: dp.raincloud_plot(d, "y", group="grp"),
                       ("color", "fill")),
    "group_comparison_plot": (lambda d: dp.group_comparison_plot(d, "y", "grp"),
                              ("color",)),
    "explore_categorical": (
        lambda d: dp.explore_categorical(d, "cat", group="grp"), ("fill",)),
    "ridgeline_plot": (lambda d: dp.ridgeline_plot(d, "y", "grp"), ("fill",)),
}

UNUSED_COLOURS = {"East": "#005b96", "South": "#e69f00", "West": "#009e73"}
NUMERIC_COLOURS = {"1": "#005b96", "2": "#e69f00"}
TREATMENT_COLOURS = {"standard": "#005b96", "enhanced": "#e69f00"}
MISSING_MESSAGE = "5 observation(s) with a missing group were dropped."


def _built(p):
    # A built copy, as R's ggplot_build() gives it, leaving the plot as it was.
    built = copy.deepcopy(p)
    built._build()
    return built


def _level_colours(built, aesthetic):
    """The colour each level takes, read from the trained scale."""
    scale = built.scales.get_scales(aesthetic)
    limits = list(scale.final_limits)
    colours = scale.map(limits)
    return {str(level): str(c).lower() for level, c in zip(limits, colours,
                                                             strict=True)}


def _drawn_palette(built, aesthetic):
    """The palette colours drawn in any layer.

    An unused level that kept a slot shows up here as a colour the legend does
    not explain.
    """
    drawn = set()
    for layer in built.layers:
        if aesthetic in layer.data:
            drawn |= {str(c).lower() for c in layer.data[aesthetic]}
    return drawn & (set(OKABE_ITO) | {NA_VALUE})


def _assert_level_colours(name, built, expected):
    for aesthetic in GROUPED_PLOTS[name][1]:
        assert _level_colours(built, aesthetic) == expected, (name, aesthetic)
        assert _drawn_palette(built, aesthetic) == set(expected.values()), (
            name, aesthetic)


def _unused_frame():
    # wellbeing_survey without its North respondents: the categorical keeps
    # North as an unused category between East and South.
    wb = dp.wellbeing_survey()
    sub = wb[wb["region"] != "North"].reset_index(drop=True)
    return pd.DataFrame({"x": sub["age"], "y": sub["income"],
                         "cat": sub["education"], "grp": sub["region"]})


def _numeric_frame():
    # crop_yield with a block coded 1 and 2.
    cy = dp.crop_yield()
    n = len(cy)
    return pd.DataFrame({"x": cy["fertiliser"], "y": cy["yield"],
                         "cat": np.resize(["a", "b", "c", "d"], n),
                         "grp": np.resize([1, 2], n)})


def _missing_frame():
    # crop_yield with the treatment of its first five fields missing.
    cy = dp.crop_yield()
    d = pd.DataFrame({"x": cy["fertiliser"], "y": cy["yield"],
                      "cat": np.resize(["a", "b", "c", "d"], len(cy)),
                      "grp": cy["treatment"]})
    d.loc[d.index[:5], "grp"] = np.nan
    return d


@pytest.mark.parametrize("name", list(GROUPED_PLOTS))
def test_an_unused_level_takes_no_colour_and_no_slot(name):
    d = _unused_frame()
    assert list(d["grp"].cat.categories) == ["East", "North", "South", "West"]
    # depictr's warnings and plotnine's are UserWarnings. The rows missing an
    # income are left out before the groups are formed, as in the R twin, so
    # plotnine no longer warns that it removed them.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        built = _built(GROUPED_PLOTS[name][0](d))
    # An unused category kept its slot, so South took the third colour, where
    # the R twin gives it the second.
    _assert_level_colours(name, built, UNUSED_COLOURS)


@pytest.mark.parametrize("name", list(GROUPED_PLOTS))
def test_a_numeric_code_becomes_a_group(name):
    # scatter_trend(), raincloud_plot(), group_comparison_plot() and
    # explore_categorical() raised TypeError, and explore_distribution() and
    # ecdf_plot() drew one ungrouped curve.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        built = _built(GROUPED_PLOTS[name][0](_numeric_frame()))
    _assert_level_colours(name, built, NUMERIC_COLOURS)


def test_a_float_code_is_labelled_as_a_whole_number():
    # A missing value makes a column of codes float, and the codes would read
    # 1.0 and 2.0, where the R twin reads 1 and 2.
    d = _numeric_frame()
    d["grp"] = d["grp"].astype(float)
    d.loc[0, "grp"] = np.nan
    with pytest.warns(UserWarning, match=re.escape(
            "1 observation(s) with a missing group were dropped.")):
        built = _built(dp.scatter_trend(d, "x", "y", group="grp"))
    assert _level_colours(built, "color") == NUMERIC_COLOURS


@pytest.mark.parametrize("name", list(GROUPED_PLOTS))
def test_rows_with_a_missing_group_are_dropped_with_a_warning(name):
    # scatter_trend(), ecdf_plot(), explore_distribution() and raincloud_plot()
    # drew the missing rows as a grey group of their own, and the others
    # dropped them without a word.
    with pytest.warns(UserWarning, match=re.escape(MISSING_MESSAGE)):
        p = GROUPED_PLOTS[name][0](_missing_frame())
    _assert_level_colours(name, _built(p), TREATMENT_COLOURS)


@pytest.mark.parametrize("name", list(GROUPED_PLOTS))
def test_a_continuous_or_wholly_missing_group_is_refused(name):
    plot = GROUPED_PLOTS[name][0]
    d = _numeric_frame()
    d["grp"] = d["x"]
    msg = (f"`group` (grp) is numeric with {d['x'].nunique()} distinct values; "
           "it looks continuous, not categorical. Bin it, or convert it to a "
           "categorical first.")
    with pytest.raises(ValueError, match=re.escape(msg)):
        plot(d)
    d = _missing_frame()
    d["grp"] = np.nan
    with pytest.raises(ValueError,
                       match=re.escape("`group` is missing for every observation.")):
        plot(d)


def test_twenty_distinct_codes_are_still_groups():
    d = _numeric_frame()
    d["grp"] = np.resize(np.arange(1, 21), len(d))
    # Twenty groups are past the eight colours of the palette, which says so.
    with pytest.warns(UserWarning, match="20 colours interpolated"):
        p = dp.scatter_trend(d, "x", "y", group="grp")
    assert len(_level_colours(_built(p), "color")) == 20


def test_an_unused_level_leaves_the_axis_of_group_comparison_plot():
    # The trained position scales live on the layout, one per panel.
    built = _built(dp.group_comparison_plot(_unused_frame(), "y", "grp"))
    assert list(built.layout.panel_scales_x[0].final_limits) == [
        "East", "South", "West"]


def test_explore_categorical_orders_and_colours_the_groups_by_level():
    # The R twin ordered the groups alphabetically and gave postgraduate the
    # first colour. Both twins now give it to secondary, the first level.
    built = _built(dp.explore_categorical(dp.wellbeing_survey(), "region",
                                          group="education"))
    colours = _level_colours(built, "fill")
    assert list(colours.items()) == [("secondary", "#005b96"),
                                     ("undergraduate", "#e69f00"),
                                     ("postgraduate", "#009e73")]


def test_ridgeline_plot_colours_by_level_and_orders_by_median():
    # The ridges are stacked by median, as before, but each takes the colour of
    # its level, as in the R twin, where the median order used to decide it.
    wb = dp.wellbeing_survey()
    built = _built(dp.ridgeline_plot(wb, "life_satisfaction", "region"))
    assert _level_colours(built, "fill") == dict(
        zip(["East", "North", "South", "West"], OKABE_ITO[:4], strict=True))
    medians = wb.groupby("region", observed=True)["life_satisfaction"].median()
    order = list(medians.sort_values().index)[::-1]
    assert list(built.data["region"].cat.categories) == order


def test_prepare_group_names_the_palette_by_level():
    from depictr._groups import _prepare_group

    d = _unused_frame()
    data, levels, palette = _prepare_group(d, "grp")
    assert list(data["grp"].cat.categories) == ["East", "South", "West"]
    assert levels == ["East", "South", "West"]
    assert palette == UNUSED_COLOURS
    # The caller's frame is left as it was.
    assert list(d["grp"].cat.categories) == ["East", "North", "South", "West"]
