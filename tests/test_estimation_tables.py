"""Tests for the estimation plot and the descriptive summary table.

The plot tests smoke-build each figure (conftest forces the Agg backend); the
table tests check the expected rows and columns are present.
"""

import numpy as np
import pytest
from plotnine import ggplot

import depictr.data as data
from depictr.estimation import estimation_plot
from depictr.tables import summary_table

CY = data.crop_yield()
WB = data.wellbeing_survey()


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


# ---- estimation_plot -------------------------------------------------------

def test_estimation_two_group_builds_and_carries_differences():
    p = estimation_plot(CY, "yield", "treatment", seed=1, n_boot=300)
    assert _builds(p)
    diffs = p.differences
    # One row per non-reference group.
    assert len(diffs) == 1
    assert {"diff", "lower", "upper", "cohens_d", "hedges_g"} <= set(diffs.columns)
    # The bootstrap interval brackets the point estimate.
    row = diffs.iloc[0]
    assert row["lower"] <= row["diff"] <= row["upper"]


def test_estimation_multi_group_with_reference():
    p = estimation_plot(WB, "life_satisfaction", "region",
                        reference="North", seed=2, n_boot=300)
    assert _builds(p)
    # Three non-reference regions out of four.
    assert len(p.differences) == 3
    assert (p.differences["reference"] == "North").all()


@pytest.mark.parametrize("effsize", ["hedges_g", "cohens_d", "none"])
def test_estimation_effsize_variants_build(effsize):
    assert _builds(estimation_plot(CY, "yield", "treatment",
                                   effsize=effsize, seed=1, n_boot=200))


def test_estimation_two_panel_composes_and_carries_differences():
    # The Gardner-Altman layout returns a composition (not a bare ggplot) that
    # still draws and still carries the differences table.
    p = estimation_plot(CY, "yield", "treatment", two_panel=True,
                        seed=1, n_boot=200)
    p.draw(show=False)
    assert len(p.differences) == 1


def test_estimation_two_panel_omits_the_interval_for_a_single_observation():
    # The top panel used to draw lo == hi == mean for a one-observation group,
    # implying perfect precision, while the difference panel below correctly
    # reported the same group as undefined. Both must now abstain.
    import pandas as pd

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 9.0],
                       "g": ["a", "a", "a", "c", "c", "b"]})
    p = estimation_plot(df, "y", "g", two_panel=True, seed=1, n_boot=200)
    top, _ = list(p)
    singleton = top.data[top.data["group"] == "b"].iloc[0]
    assert np.isnan(singleton["lo"]) and np.isnan(singleton["hi"])
    diff = p.differences.set_index("group").loc["b"]
    assert np.isnan(diff["lower"]) and np.isnan(diff["upper"])
    p.draw(show=False)


def test_estimation_bootstrap_is_reproducible_with_seed():
    a = estimation_plot(CY, "yield", "treatment", seed=7, n_boot=300).differences
    b = estimation_plot(CY, "yield", "treatment", seed=7, n_boot=300).differences
    np.testing.assert_allclose(a["lower"], b["lower"])
    np.testing.assert_allclose(a["upper"], b["upper"])


def test_estimation_bad_inputs():
    with pytest.raises(KeyError):
        estimation_plot(CY, "not_a_column", "treatment")
    with pytest.raises(TypeError):
        estimation_plot(CY, "treatment", "treatment")  # y not numeric
    with pytest.raises(ValueError):
        estimation_plot(CY, "yield", "treatment", reference="nope")


# ---- summary_table ---------------------------------------------------------

def test_summary_table_basic_rows():
    t = summary_table(CY, vars=["yield", "rainfall", "treatment"])
    assert list(t.columns) == ["variable", "statistic", "Overall"]
    # First row is the sample size.
    assert t.iloc[0]["variable"] == "N"
    assert t.iloc[0]["Overall"] == str(len(CY))
    # Numeric variables get a Mean (SD) row.
    assert "Mean (SD)" in set(t["statistic"])
    # The categorical variable expands to one row per level.
    cats = set(CY["treatment"].unique())
    assert cats <= set(t["statistic"])


def test_summary_table_grouped_columns_and_missing_row():
    t = summary_table(WB, vars=["life_satisfaction", "income", "education"],
                      group="region")
    # An Overall column plus one per region level.
    for col in ["variable", "statistic", "Overall"]:
        assert col in t.columns
    for level in WB["region"].unique():
        assert str(level) in t.columns
    # income has missing values, so a missing-count row is present.
    assert "Missing, n (%)" in set(t["statistic"])
    # The N row reports per-group sizes that sum to the overall N.
    n_row = t[t["variable"] == "N"].iloc[0]
    group_n = sum(int(n_row[str(g)]) for g in WB["region"].unique())
    assert group_n == int(n_row["Overall"]) == len(WB)


def test_summary_table_accounts_for_a_missing_group_value():
    # A record with no group label used to drop out of every group column while
    # Overall still counted it, so the group sizes no longer summed to N.
    import pandas as pd

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", None]})
    t = summary_table(df, vars=["x"], group="g")
    n_row = t[t["variable"] == "N"].iloc[0]
    assert int(n_row["Overall"]) == len(df)
    assert int(n_row["a"]) + int(n_row["b"]) + int(n_row["Missing"]) == len(df)


def test_summary_table_group_columns_are_ordered_not_first_seen():
    # The same data in a different row order must give the same table.
    import pandas as pd

    shuffled = WB.sample(frac=1, random_state=3)
    a = summary_table(WB, vars=["stress"], group="region")
    b = summary_table(shuffled, vars=["stress"], group="region")
    assert list(a.columns) == ["variable", "statistic", "Overall",
                               "East", "North", "South", "West"]
    pd.testing.assert_frame_equal(a, b)


def test_summary_table_suppress_missing():
    t = summary_table(WB, vars=["income"], group=None, missing=False)
    assert "Missing, n (%)" not in set(t["statistic"])


def test_summary_table_default_vars_skips_nothing_unexpected():
    # With no vars given, every non-group column appears (none are identifiers).
    t = summary_table(CY)
    listed = set(t["variable"]) - {"", "N"}
    assert listed == set(CY.columns)
