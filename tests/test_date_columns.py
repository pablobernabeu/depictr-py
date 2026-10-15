"""Dates, date-times, durations and other columns that are not plain numbers.

``summary_table()`` gives a datetime column one row, its range, and a timedelta
column its mean (SD) in a stated unit; ``explore_bivariate()`` draws both on a
continuous axis; ``correlation_heatmap()`` refuses a column it cannot correlate.
The frame and the cells pinned here are the same as in the R twin's
tests/testthat/test-date-columns.R.
"""

import copy

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

import depictr as dp


def date_frame():
    """Forty visits three days apart, two arms of twenty, a time stamp every
    90 minutes and a duration that cycles through one to eight days."""
    i = np.arange(40)
    return pd.DataFrame({
        "id": [f"p{k:02d}" for k in i],
        "arm": ["a"] * 20 + ["b"] * 20,
        "visit": pd.Timestamp("2024-01-01") + pd.to_timedelta(3 * i, unit="D"),
        "stamp": pd.Timestamp("2024-03-01 08:30", tz="UTC")
        + pd.to_timedelta(90 * i, unit="min"),
        "dur": pd.to_timedelta(i % 8 + 1, unit="D"),
        "score": i / 10,
    })


def _cells(tab, row, cols=("Overall", "a", "b")):
    return [tab.iloc[row][c] for c in cols]


def _geoms(p):
    return [type(layer.geom).__name__ for layer in p.layers]


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


# ---- summary_table ---------------------------------------------------------

def test_summary_table_gives_a_date_column_one_row_its_range():
    d = date_frame()
    tab = dp.summary_table(d, vars=["visit"])
    # One row for N and one for the dates, where there was one per day.
    assert len(tab) == 2
    assert list(tab["statistic"]) == ["", "Range"]
    assert tab.iloc[1]["Overall"] == "2024-01-01 to 2024-04-27"

    grouped = dp.summary_table(d, vars=["visit"], group="arm")
    assert _cells(grouped, 1) == ["2024-01-01 to 2024-04-27",
                                  "2024-01-01 to 2024-02-27",
                                  "2024-03-01 to 2024-04-27"]


def test_summary_table_writes_a_datetime_range_to_the_precision_it_needs():
    d = date_frame()
    grouped = dp.summary_table(d, vars=["stamp"], group="arm")
    assert grouped.iloc[1]["statistic"] == "Range"
    # No time has seconds, so the cells stop at minutes.
    assert _cells(grouped, 1) == ["2024-03-01 08:30 to 2024-03-03 19:00",
                                  "2024-03-01 08:30 to 2024-03-02 13:00",
                                  "2024-03-02 14:30 to 2024-03-03 19:00"]

    # Times that all fall at midnight are written as dates.
    midnight = pd.DataFrame({"t": pd.to_datetime(["2024-05-01", "2024-05-09"])})
    assert dp.summary_table(midnight, vars=["t"]).iloc[1]["Overall"] == (
        "2024-05-01 to 2024-05-09")
    # A time with seconds keeps them.
    seconds = pd.DataFrame(
        {"t": pd.to_datetime(["2024-05-01 10:00:05", "2024-05-01 10:30:00"])})
    assert dp.summary_table(seconds, vars=["t"]).iloc[1]["Overall"] == (
        "2024-05-01 10:00:05 to 2024-05-01 10:30:00")


def test_summary_table_summarises_a_duration_as_mean_sd_in_a_unit():
    d = date_frame()
    grouped = dp.summary_table(d, vars=["dur"], group="arm")
    assert grouped.iloc[1]["statistic"] == "Mean (SD), days"
    assert _cells(grouped, 1) == ["4.5 (2.3)", "4.1 (2.3)", "4.9 (2.3)"]

    # A timedelta carries no unit of its own, so the median picks one.
    hours = pd.DataFrame({"h": pd.to_timedelta([1.5, 2.5, 4], unit="h")})
    tab = dp.summary_table(hours, vars=["h"])
    assert tab.iloc[1]["statistic"] == "Mean (SD), hours"
    assert tab.iloc[1]["Overall"] == "2.7 (1.3)"

    minutes = pd.DataFrame({"m": pd.to_timedelta([90, 150, 2000], unit="s")})
    assert dp.summary_table(minutes, vars=["m"]).iloc[1]["statistic"] == (
        "Mean (SD), mins")
    seconds = pd.DataFrame({"s": pd.to_timedelta([0, 5, 59], unit="s")})
    assert dp.summary_table(seconds, vars=["s"]).iloc[1]["statistic"] == (
        "Mean (SD), secs")


def test_summary_table_keeps_dates_in_one_row_each_and_warns_on_a_skip():
    d = date_frame()
    with pytest.warns(UserWarning,
                      match=r"skipping high-cardinality column\(s\): id\."):
        tab = dp.summary_table(d, group="arm")
    # N, then one row each for visit, stamp, dur and score. The datetime and
    # timedelta columns were skipped as identifiers without a word.
    assert list(tab["variable"]) == ["N", "visit", "stamp", "dur", "score"]
    assert list(tab["statistic"]) == ["", "Range", "Range", "Mean (SD), days",
                                      "Mean (SD)"]


def test_summary_table_reports_missing_dates_and_an_empty_group():
    d = date_frame()
    d.loc[20:, "visit"] = pd.NaT
    tab = dp.summary_table(d, vars=["visit"], group="arm")
    assert list(tab["statistic"]) == ["", "Range", "Missing, n (%)"]
    assert _cells(tab, 1) == ["2024-01-01 to 2024-02-27",
                              "2024-01-01 to 2024-02-27", "--"]
    assert tab.iloc[2]["b"] == "20 (100%)"


# ---- explore_bivariate -----------------------------------------------------

@pytest.mark.parametrize("xy", [("visit", "score"), ("score", "stamp"),
                                ("dur", "score")])
def test_explore_bivariate_draws_dates_and_durations_as_a_scatter(xy):
    d = date_frame()
    # Such a column was taken for a category, and drawing the boxes over it
    # raised "Continuous value supplied to discrete scale".
    p = dp.explore_bivariate(d, *xy)
    assert _geoms(p) == ["geom_point", "geom_smooth"]
    assert _builds(p)


def test_explore_bivariate_draws_a_duration_in_a_stated_unit():
    # As in the R twin: a number in the unit summary_table() reports, which
    # the axis title names.
    d = date_frame()
    p = dp.explore_bivariate(d, "dur", "score")
    assert p.labels.x == "dur (days)"
    assert p.labels.y == "score"
    assert list(p.data["dur"].head(3)) == [1.0, 2.0, 3.0]
    # The caller's frame keeps its timedelta column.
    assert d["dur"].dtype.kind == "m"
    box = dp.explore_bivariate(d, "arm", "dur")
    assert box.labels.y == "dur (days)"


def test_explore_bivariate_draws_dates_by_category_as_box_plots():
    d = date_frame()
    p = dp.explore_bivariate(d, "arm", "visit")
    assert _geoms(p)[0] == "geom_boxplot"
    # One box per arm, where there was one per day; the copy keeps the build
    # from altering the plot under test.
    built = copy.deepcopy(p)
    built._build()
    assert len(built.layers[0].data) == 2
    assert _builds(p)


# ---- correlation_heatmap ---------------------------------------------------

def test_correlation_heatmap_refuses_columns_it_cannot_correlate():
    cy = dp.crop_yield()
    with pytest.raises(TypeError) as err:
        dp.correlation_heatmap(cy, cols=["yield", "treatment", "rainfall"])
    assert str(err.value) == (
        "correlation_heatmap() needs numeric columns; these are not: treatment.")
    d = date_frame()
    with pytest.raises(TypeError) as err:
        dp.correlation_heatmap(d, cols=["score", "visit", "arm"])
    assert str(err.value) == (
        "correlation_heatmap() needs numeric columns; these are not: visit, arm.")
    # A boolean column correlates as 0 and 1, as it did before.
    d["late"] = d["score"] > 2
    assert _builds(dp.correlation_heatmap(d, cols=["score", "late"]))
