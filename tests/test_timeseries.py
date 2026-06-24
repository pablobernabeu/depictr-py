"""Smoke tests for the time-series plots: each builds a figure without error."""

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


def test_timeseries_plot_raw_and_rolling():
    assert _builds(ts.timeseries_plot(S))
    assert _builds(ts.timeseries_plot(V, rolling=12))
