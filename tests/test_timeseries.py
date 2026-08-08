"""Smoke tests for the time-series plots: each builds a figure without error."""

import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

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


def test_daily_index_infers_a_weekly_period():
    daily = pd.Series(
        np.arange(40.0),
        index=pd.period_range("2000-01-01", periods=40, freq="D"),
    )
    assert ts._infer_period(daily, None) == 7


def test_timeseries_plot_raw_and_rolling():
    assert _builds(ts.timeseries_plot(S))
    assert _builds(ts.timeseries_plot(V, rolling=12))
