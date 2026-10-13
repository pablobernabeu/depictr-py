"""Smoke tests for the time-series plots: each builds a figure without error."""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.lines import Line2D
from plotnine import ggplot
from plotnine.exceptions import PlotnineWarning

import depictr.timeseries as ts


def _monthly_series(n=120, seed=0):
    """A monthly series with a trend, a 12-period seasonal term and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)
    idx = pd.period_range("2010-01", periods=n, freq="M")
    return pd.Series(values, index=idx)


S = _monthly_series()
V = S.to_numpy()


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


def test_acf_plot_kinds():
    pytest.importorskip("statsmodels")
    assert _builds(ts.acf_plot(S, kind="acf"))
    assert _builds(ts.acf_plot(S, kind="pacf"))
    assert _builds(ts.acf_plot(V, kind="acf", lags=20))


def test_acf_plot_bad_kind():
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError):
        ts.acf_plot(S, kind="nonsense")


def test_decompose_plot():
    pytest.importorskip("statsmodels")
    assert _builds(ts.decompose_plot(S, model="additive"))
    assert _builds(ts.decompose_plot(V, period=12, model="additive"))


def test_decompose_plot_needs_period_for_bare_array():
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError):
        ts.decompose_plot(V)  # no index, no period to infer from


def test_seasonal_plot():
    assert _builds(ts.seasonal_plot(S))
    assert _builds(ts.seasonal_plot(V, period=12))


def test_seasonal_cycles_are_ordered_numerically():
    # A plain categorical over strings sorts "10" between "1" and "2", which
    # both scrambles the legend and misassigns the colour ramp.
    p = ts.seasonal_plot(_monthly_series(144), period=12)
    categories = list(p.data["cycle"].cat.categories)
    assert p.data["cycle"].cat.ordered
    assert categories == [str(c) for c in range(12)]


def test_seasonal_cycles_use_the_sequential_ramp():
    # Twelve cycles is past the eight colourblind-safe qualitative colours, so
    # the cycle scale must be the sequential ramp rather than an interpolated
    # qualitative palette (which would also warn).
    import depictr as dp

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = ts.seasonal_plot(_monthly_series(144), period=12)
    scale = next(s for s in p.scales if "color" in s.aesthetics)
    assert list(scale.palette(12)) == dp.depictr_palette(12, kind="sequential")


def test_annual_index_is_not_a_seasonal_period():
    # An annual index once mapped to period 1, which is not a seasonal period:
    # seasonal_plot drew one point per "cycle" and neither function complained.
    annual = pd.Series(
        np.arange(20.0), index=pd.period_range("2000", periods=20, freq="Y")
    )
    with pytest.raises(ValueError, match="Could not infer the seasonal period"):
        ts.seasonal_plot(annual)
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match="Could not infer the seasonal period"):
        ts.decompose_plot(annual)


def test_seasonal_plot_rejects_an_empty_series():
    # The cycle count now comes from cycle.max(), so an empty series has to be
    # turned away by name rather than by a numpy reduction error.
    with pytest.raises(ValueError, match="no non-missing values"):
        ts.seasonal_plot(np.array([]), period=12)


def test_internal_missing_values_are_refused():
    # Dropping an internal NaN closed up the gap silently: the ACF correlated
    # values across it, the decomposition misaligned, and every post-gap
    # observation landed on the wrong within-period position.
    s = _monthly_series()
    s.iloc[40] = np.nan
    with pytest.raises(ValueError, match="internal missing values"):
        ts.seasonal_plot(s)
    with pytest.raises(ValueError, match="internal missing values"):
        ts.seasonal_plot(s.to_numpy(), period=12)
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match="internal missing values"):
        ts.acf_plot(s)
    with pytest.raises(ValueError, match="internal missing values"):
        ts.decompose_plot(s)


def test_missing_values_at_the_ends_are_trimmed():
    # Trimming the ends only shortens the series; no observation moves relative
    # to another, so it stays legitimate where an internal drop is not.
    s = _monthly_series()
    s.iloc[[0, -1]] = np.nan
    assert _builds(ts.seasonal_plot(s))
    pytest.importorskip("statsmodels")
    assert _builds(ts.acf_plot(s))
    assert _builds(ts.decompose_plot(s))


def test_all_missing_series_keeps_its_existing_error():
    # An all-missing series trims to nothing, so it reaches the same named
    # refusal an empty series does rather than the internal-missing error.
    with pytest.raises(ValueError, match="no non-missing values"):
        ts.seasonal_plot(pd.Series([np.nan] * 24), period=12)
    # acf_plot took log10 of a zero length and raised an OverflowError from the
    # int() around it, so it is held to the same named refusal.
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match="no non-missing values"):
        ts.acf_plot(pd.Series([np.nan] * 24))
    with pytest.raises(ValueError, match="no non-missing values"):
        ts.acf_plot(np.array([]))


def test_daily_index_infers_a_weekly_period():
    daily = pd.Series(
        np.arange(40.0),
        index=pd.period_range("2000-01-01", periods=40, freq="D"),
    )
    assert ts._infer_period(daily, None) == 7


def test_timeseries_plot_raw_and_rolling():
    assert _builds(ts.timeseries_plot(S))
    assert _builds(ts.timeseries_plot(V, rolling=12))


def _draw(p):
    """Draw `p` once, returning its lines as drawn and plotnine's warnings.

    Each line is an (n, 2) array of x and y, in layer order. plotnine hands
    matplotlib one Line2D per path and keeps an interior missing value as a NaN
    vertex, where matplotlib leaves a gap. A NaN row inside an array is
    therefore a break in the drawn line. A plot is drawn only once here, since
    drawing it again adds its layers to the same figure a second time.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = p.draw(show=False)
    try:
        lines = [a.get_xydata() for a in fig.axes[0].get_children()
                 if isinstance(a, Line2D)]
    finally:
        plt.close(fig)
    notes = [str(w.message) for w in caught if issubclass(w.category, PlotnineWarning)]
    return lines, notes


def test_timeseries_plot_keeps_a_gap_in_place():
    # Dropping the missing value closed up the gap: 6.0 was drawn at x = 5,
    # every later point one step to the left, and the line ran from 4 to 6.
    values = np.arange(20.0)
    values[5] = np.nan
    p = ts.timeseries_plot(pd.Series(values))
    assert p.data["x"].tolist() == list(range(20))
    assert p.data.loc[p.data["x"] == 6, "value"].item() == 6.0
    (line,), _ = _draw(p)
    np.testing.assert_array_equal(line[:, 0], np.arange(20.0))
    assert np.isnan(line[5, 1])
    np.testing.assert_array_equal(np.delete(line[:, 1], 5), np.delete(values, 5))


def test_timeseries_plot_rolling_mean_is_missing_across_a_gap():
    # The centred mean was taken over the closed-up series, so the windows next
    # to the gap averaged values from both sides of it (4.33 and 5.67 here).
    values = np.arange(20.0)
    values[5] = np.nan
    (_, roll), _ = _draw(ts.timeseries_plot(pd.Series(values), rolling=3))
    # The first and last windows run off the series and are not drawn.
    np.testing.assert_array_equal(roll[:, 0], np.arange(1.0, 19.0))
    # The windows centred on 4, 5 and 6 reach the gap, so the overlay breaks
    # there. Elsewhere, the centred mean of a straight line is the line itself.
    gap = np.isin(roll[:, 0], [4, 5, 6])
    assert np.isnan(roll[gap, 1]).all()
    np.testing.assert_allclose(roll[~gap, 1], roll[~gap, 0])


def test_timeseries_plot_draws_an_object_column_with_its_gap():
    # Python floats and None arrive with the object dtype. plotnine read such
    # values as categories, one group each, so no line was drawn at all, with
    # or without the None.
    s = pd.Series([1.0, None, 3.0, 4.0], dtype=object)
    (line,), _ = _draw(ts.timeseries_plot(s))
    np.testing.assert_array_equal(line[:, 0], [0, 1, 2, 3])
    assert np.isnan(line[1, 1])


def test_timeseries_plot_keeps_a_missing_month_on_the_date_axis():
    # With June missing, the line ran straight from May to July.
    idx = pd.period_range("2020-01", periods=12, freq="M")
    values = np.arange(1.0, 13.0)
    values[5] = np.nan
    p = ts.timeseries_plot(pd.Series(values, index=idx))
    assert list(p.data["x"]) == list(idx.to_timestamp())
    (line,), _ = _draw(p)
    assert len(line) == 12
    assert np.isnan(line[5, 1])


def test_timeseries_plot_counts_positions_from_the_first_entry():
    # Missing values at either end keep their positions too, and the ends that
    # cannot be drawn are left out without a warning.
    values = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, np.nan])
    p = ts.timeseries_plot(pd.Series(values), rolling=3)
    assert p.data["x"].tolist() == list(range(7))
    (line, roll), notes = _draw(p)
    assert notes == []
    np.testing.assert_array_equal(line, [[2, 1], [3, 2], [4, 3], [5, 4]])
    np.testing.assert_array_equal(roll, [[3, 2], [4, 3]])


def test_timeseries_plot_rolling_overlay_does_not_warn_on_complete_data():
    # The undrawable ends of the overlay raised "geom_path: Removed 11 rows
    # containing missing values" for every complete series.
    _, notes = _draw(ts.timeseries_plot(V, rolling=12))
    assert notes == []


def test_several_columns_are_refused():
    # A frame with two columns was read row by row into one series (1, 100, 2,
    # 200 and so on), and timeseries_plot() dropped its index as well.
    frame = pd.DataFrame({"a": V, "b": V + 100}, index=S.index)
    two = re.escape("`x` has 2 columns; pass one series at a time.")
    three = re.escape("`x` has 3 columns; pass one series at a time.")
    array = np.column_stack([V, V, V])
    with pytest.raises(ValueError, match=two):
        ts.timeseries_plot(frame)
    with pytest.raises(ValueError, match=three):
        ts.timeseries_plot(array)
    with pytest.raises(ValueError, match=two):
        ts.seasonal_plot(frame)
    with pytest.raises(ValueError, match=three):
        ts.seasonal_plot(array, period=12)
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match=two):
        ts.acf_plot(frame)
    with pytest.raises(ValueError, match=three):
        ts.acf_plot(array, kind="pacf")
    with pytest.raises(ValueError, match=two):
        ts.decompose_plot(frame)
    with pytest.raises(ValueError, match=three):
        ts.decompose_plot(array, period=12)


def test_a_single_column_is_the_series():
    # A one-column frame keeps its index, so the period and the date axis come
    # from it as they do from the Series.
    frame = S.to_frame("sales")
    pd.testing.assert_frame_equal(ts.timeseries_plot(frame).data,
                                  ts.timeseries_plot(S).data)
    pd.testing.assert_frame_equal(ts.seasonal_plot(frame).data,
                                  ts.seasonal_plot(S).data)
    pd.testing.assert_frame_equal(ts.timeseries_plot(V.reshape(-1, 1)).data,
                                  ts.timeseries_plot(V).data)
    pytest.importorskip("statsmodels")
    pd.testing.assert_frame_equal(ts.acf_plot(frame).data, ts.acf_plot(S).data)
    pd.testing.assert_frame_equal(ts.decompose_plot(frame).data,
                                  ts.decompose_plot(S).data)


def test_object_values_are_read_as_numbers():
    # The seasonal plot took the values of an object column as they came, and
    # plotnine drew them as categories, one evenly spaced tick per value.
    obj = S.astype(object)
    expected = ts.seasonal_plot(S).data
    pd.testing.assert_frame_equal(ts.seasonal_plot(obj).data, expected)
    pd.testing.assert_frame_equal(ts.seasonal_plot(obj.to_frame("sales")).data,
                                  expected)
