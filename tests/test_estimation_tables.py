"""Tests for the estimation plot and the descriptive summary table.

The plot tests smoke-build each figure (conftest forces the Agg backend); the
table tests check the expected rows and columns are present.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from plotnine import ggplot

import depictr.data as data
from depictr.estimation import estimation_plot
from depictr.tables import summary_table

CY = data.crop_yield()
WB = data.wellbeing_survey()
LD = data.lexical_decision()


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
    # The two-panel layout returns a composition (not a bare ggplot) that
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


# ---- where each difference is drawn ----------------------------------------
# A reader matches a difference to its group by the column it sits in, so every
# group must take the same place across the width of both panels. plotnine's
# composition gives the stacked panels the same width, so a column's place is
# read from the drawn axes as a fraction of the panel's x range.

def _draw(p):
    fig = p.draw(show=False)
    plt.close(fig)
    return fig


def _columns(ax):
    """Each tick label's place across the panel, from 0 (left) to 1 (right)."""
    lo, hi = ax.get_xlim()
    return {label.get_text(): (x - lo) / (hi - lo)
            for x, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=True)}


def _points(ax):
    """The drawn points, as (place across the panel, y) pairs."""
    lo, hi = ax.get_xlim()
    return [((x - lo) / (hi - lo), y)
            for c in ax.collections if isinstance(c, PathCollection)
            for x, y in c.get_offsets()]


def test_estimation_differences_sit_beneath_their_groups():
    p = estimation_plot(WB, "life_satisfaction", "region", reference="North",
                        two_panel=True, seed=2, n_boot=200)
    top_ax, bottom_ax = _draw(p).axes
    top, bottom = _columns(top_ax), _columns(bottom_ax)
    # The reference comes first above, as in the R twin, and keeps an empty,
    # unlabelled column below.
    assert list(top) == ["North", "East", "South", "West"]
    assert list(bottom) == ["", "East", "South", "West"]
    assert bottom[""] == pytest.approx(top["North"])
    for g in ["East", "South", "West"]:
        assert bottom[g] == pytest.approx(top[g])
    # Each difference is drawn in its own group's column.
    diffs = p.differences.set_index("group")["diff"]
    points = _points(bottom_ax)
    assert len(points) == 3
    for x, y in points:
        group = min(top, key=lambda g, x=x: abs(top[g] - x))
        assert x == pytest.approx(top[group])
        assert y == pytest.approx(diffs[group])


def test_estimation_single_panel_keeps_the_reference_column():
    p = estimation_plot(WB, "life_satisfaction", "region", reference="North",
                        seed=2, n_boot=200)
    (ax,) = _draw(p).axes
    columns = _columns(ax)
    assert list(columns) == ["", "East", "South", "West"]
    diffs = p.differences.set_index("group")["diff"]
    for x, y in _points(ax):
        group = min(columns, key=lambda g, x=x: abs(columns[g] - x))
        assert y == pytest.approx(diffs[group])


def test_estimation_two_groups_difference_sits_beneath_its_group():
    p = estimation_plot(LD, "RT", "condition", two_panel=True, seed=1, n_boot=200)
    top_ax, bottom_ax = _draw(p).axes
    top, bottom = _columns(top_ax), _columns(bottom_ax)
    assert list(top) == ["related", "unrelated"]
    assert bottom["unrelated"] == pytest.approx(top["unrelated"])
    ((x, _),) = _points(bottom_ax)
    assert x == pytest.approx(top["unrelated"])


# ---- the default reference -------------------------------------------------

def test_estimation_default_reference_ignores_row_order():
    # 0.3.0 took the reference from the first row, so shuffling the rows
    # flipped the sign of every difference.
    shuffled = CY.sample(frac=1, random_state=4)
    assert shuffled["treatment"].iloc[0] == "enhanced"
    with pytest.warns(FutureWarning):
        a = estimation_plot(shuffled, "yield", "treatment", seed=1, n_boot=200)
    b = estimation_plot(CY, "yield", "treatment", seed=1, n_boot=200)
    for d in (a.differences, b.differences):
        assert d["reference"].tolist() == ["standard"]
        assert d["group"].tolist() == ["enhanced"]
    for col in ("diff", "cohens_d", "hedges_g"):
        assert a.differences[col].iloc[0] == pytest.approx(b.differences[col].iloc[0])


def test_estimation_default_reference_follows_the_declared_order():
    rng = np.random.default_rng(3)
    y = rng.normal(size=8)
    # A categorical keeps its declared order wherever its rows sit, and a
    # category with no observations is passed over.
    cat = pd.Categorical(["enhanced", "standard"] * 4,
                         categories=["absent", "standard", "enhanced"])
    with pytest.warns(FutureWarning):
        p = estimation_plot(pd.DataFrame({"y": y, "g": cat}), "y", "g",
                            seed=1, n_boot=50)
    assert p.differences["reference"].tolist() == ["standard"]
    # Anything else is sorted.
    plain = ["b", "a"] * 4
    with pytest.warns(FutureWarning):
        p = estimation_plot(pd.DataFrame({"y": y, "g": plain}), "y", "g",
                            seed=1, n_boot=50)
    assert p.differences["reference"].tolist() == ["a"]


def test_estimation_warns_only_where_the_old_default_differs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The first row is "standard", which the new rule also chooses.
        estimation_plot(CY, "yield", "treatment", seed=1, n_boot=50)
        # A reference passed explicitly is never questioned.
        estimation_plot(WB, "life_satisfaction", "region", reference="South",
                        seed=1, n_boot=50)
    # wellbeing_survey's first complete row is in the South, which 0.3.0 took
    # as the reference. The first level is East.
    with pytest.warns(FutureWarning, match="'East'.*'South'"):
        p = estimation_plot(WB, "life_satisfaction", "region", seed=1, n_boot=50)
    assert (p.differences["reference"] == "East").all()


# ---- labelling and reproducibility -----------------------------------------

def test_estimation_caption_names_the_bootstrap():
    # The R twin's caption, word for word, at its default of 5,000 resamples.
    p = estimation_plot(CY, "yield", "treatment", seed=1)
    assert "Bootstrap 95% CI (5,000 resamples)" in [t.get_text() for t in _draw(p).texts]
    p = estimation_plot(CY, "yield", "treatment", conf_level=0.9, n_boot=200,
                        two_panel=True, seed=1)
    assert "Bootstrap 90% CI (200 resamples)" in [t.get_text() for t in _draw(p).texts]
    # A round number of resamples is written out in full, as dev/goldens/
    # estimation.R prints it from R. A group of one skips the bootstrap.
    tiny = pd.DataFrame({"y": [1.0, 2.0, 3.0], "g": ["a", "a", "b"]})
    p = estimation_plot(tiny, "y", "g", n_boot=100_000, seed=1)
    assert "Bootstrap 95% CI (100,000 resamples)" in [t.get_text() for t in _draw(p).texts]


@pytest.mark.parametrize("seed", [0, 7])
def test_estimation_seed_fixes_the_jitter(seed):
    def jittered_points(seed):
        p = estimation_plot(CY, "yield", "treatment", two_panel=True,
                            seed=seed, n_boot=50)
        top_ax = _draw(p).axes[0]
        return max((c.get_offsets() for c in top_ax.collections
                    if isinstance(c, PathCollection)), key=len)

    a, b = jittered_points(seed), jittered_points(seed)
    assert len(a) == len(CY)
    np.testing.assert_array_equal(a, b)


# ---- parity with the R twin ------------------------------------------------
# Printed by dev/goldens/estimation.R, run against the R twin with the change
# that lines up the estimation plot's columns. Each entry gives the reference a
# call takes, the columns of its upper panel and the tick labels of its lower
# one.
R_LAYOUT = [
    (("wellbeing_survey", "life_satisfaction", "region", None),
     "East", ["East", "North", "South", "West"], ["", "North", "South", "West"]),
    (("wellbeing_survey", "life_satisfaction", "region", "North"),
     "North", ["North", "East", "South", "West"], ["", "East", "South", "West"]),
    (("crop_yield", "yield", "treatment", None),
     "standard", ["standard", "enhanced"], ["", "enhanced"]),
    (("lexical_decision", "RT", "condition", None),
     "related", ["related", "unrelated"], ["", "unrelated"]),
    (("lexical_decision", "RT", "modality", None),
     "visual", ["visual", "auditory"], ["", "auditory"]),
]


@pytest.mark.parametrize(("call", "reference", "top", "bottom"), R_LAYOUT)
def test_estimation_layout_matches_the_r_twin(call, reference, top, bottom):
    name, y, group, ref = call
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        p = estimation_plot(getattr(data, name)(), y, group, reference=ref,
                            two_panel=True, seed=1, n_boot=50)
    top_ax, bottom_ax = _draw(p).axes
    assert (p.differences["reference"] == reference).all()
    assert [t.get_text() for t in top_ax.get_xticklabels()] == top
    assert [t.get_text() for t in bottom_ax.get_xticklabels()] == bottom


def test_estimation_numeric_groups_follow_the_r_twin():
    # R's factor() sorts a numeric group by value, as dev/goldens/estimation.R
    # prints for these rows. Sorted as text, 10 would come before 5 and be the
    # reference.
    df = pd.DataFrame({"y": np.arange(1.0, 13.0), "g": [20, 5, 10] * 4})
    with pytest.warns(FutureWarning, match="took 5 as the reference"):
        p = estimation_plot(df, "y", "g", two_panel=True, seed=1, n_boot=50)
    top_ax, bottom_ax = _draw(p).axes
    assert (p.differences["reference"] == 5).all()
    assert [t.get_text() for t in top_ax.get_xticklabels()] == ["5", "10", "20"]
    assert [t.get_text() for t in bottom_ax.get_xticklabels()] == ["", "10", "20"]
    # Each raw observation is drawn within the jitter of its own group's column.
    # A numeric group's data used to be drawn at their values on a continuous
    # axis. The vertical jitter is under 0.5, so rounding recovers each value.
    columns = _columns(top_ax)
    group_of = dict(zip(df["y"], df["g"], strict=True))
    raw = max((c for c in top_ax.collections if isinstance(c, PathCollection)),
              key=lambda c: len(c.get_offsets()))
    lo, hi = top_ax.get_xlim()
    for x, y in raw.get_offsets():
        place = (x - lo) / (hi - lo)
        nearest = min(columns, key=lambda g, place=place: abs(columns[g] - place))
        assert nearest == str(group_of[round(y)])
        assert abs(columns[nearest] - place) < 0.13 / (hi - lo)


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
    # With no vars given, every non-group column appears except the field
    # identifier, which has one distinct value per row and is named in a
    # warning.
    with pytest.warns(UserWarning, match=r"skipping high-cardinality column\(s\): field\."):
        t = summary_table(CY)
    listed = set(t["variable"]) - {"", "N"}
    assert listed == set(CY.columns) - {"field"}
