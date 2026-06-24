"""Smoke tests: the posterior/bootstrap forest plots build without error."""

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from depictr.data import crop_yield
from depictr.posterior import frequentist_bayesian_plot, posterior_plot


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


@pytest.fixture(scope="module")
def draws_frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Intercept": rng.normal(1.5, 0.2, 2000),
        "fertiliser": rng.normal(0.01, 0.002, 2000),
        "rainfall": rng.normal(0.002, 0.0008, 2000),
    })


@pytest.fixture(scope="module")
def ols():
    smf = pytest.importorskip("statsmodels.formula.api")
    # "yield" is a Python keyword, so the formula needs Q("yield").
    return smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph',
                   crop_yield()).fit()


def test_posterior_plot_from_frame(draws_frame):
    assert _builds(posterior_plot(draws_frame, title="Posterior"))


def test_posterior_plot_from_dict_with_ragged_arrays_and_labels():
    rng = np.random.default_rng(1)
    # Different lengths exercise the ragged-array path.
    draws = {"a": rng.normal(0, 1, 1500), "b": rng.normal(2, 0.5, 1200)}
    assert _builds(posterior_plot(draws, labels={"a": "Alpha", "b": "Beta"}))


def test_posterior_plot_rejects_bad_input():
    with pytest.raises(TypeError):
        posterior_plot([1.0, 2.0, 3.0])


def test_summary_bands_are_nested_and_ordered():
    # The inner 66% band must sit inside the outer 95% band, around the median.
    from depictr.posterior import _summarise_draws

    rng = np.random.default_rng(2)
    out = _summarise_draws({"x": rng.normal(0, 1, 50000)})
    row = out.iloc[0]
    assert row["outer_low"] < row["inner_low"] < row["median"]
    assert row["median"] < row["inner_high"] < row["outer_high"]
    # Central 95% of a standard normal is roughly +/- 1.96.
    assert abs(row["outer_low"] + 1.96) < 0.1
    assert abs(row["outer_high"] - 1.96) < 0.1


def test_frequentist_bayesian_plot_with_model(ols):
    rng = np.random.default_rng(3)
    params = ols.params
    draws = {name: rng.normal(params[name], abs(params[name]) * 0.2 + 0.01, 1000)
             for name in params.index}
    assert _builds(frequentist_bayesian_plot(ols, draws, title="Freq vs Bayes"))


def test_frequentist_bayesian_plot_with_tidy_frame():
    rng = np.random.default_rng(4)
    tidy = pd.DataFrame({
        "term": ["x", "z"], "estimate": [1.0, -0.5],
        "conf_low": [0.5, -1.0], "conf_high": [1.5, 0.0],
    })
    draws = {"x": rng.normal(1.0, 0.3, 800), "z": rng.normal(-0.5, 0.2, 800)}
    assert _builds(frequentist_bayesian_plot(tidy, draws))


def test_frequentist_bayesian_plot_rejects_disjoint_terms():
    tidy = pd.DataFrame({
        "term": ["x"], "estimate": [1.0], "conf_low": [0.5], "conf_high": [1.5],
    })
    with pytest.raises(ValueError):
        frequentist_bayesian_plot(tidy, {"other": np.zeros(100)})
