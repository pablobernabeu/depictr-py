"""Tests for depictr.mixed (random-effects caterpillar plot)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; the default Tk backend is broken here

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from plotnine import ggplot

from depictr.data import lexical_decision
from depictr.mixed import random_effects_plot


def _grouped_data(seed: int = 7):
    """lexical_decision with a participant column carrying a random intercept."""
    df = lexical_decision()
    rng = np.random.default_rng(seed)
    n = len(df)
    n_part = 24
    part_id = rng.integers(0, n_part, n)
    part_offset = rng.normal(0, 40, n_part)
    df = df.assign(participant=[f"P{i:02d}" for i in part_id])
    df["RT"] = df["RT"] + part_offset[part_id]
    return df


@pytest.fixture(scope="module")
def fitted_model():
    df = _grouped_data()
    return smf.mixedlm("RT ~ condition", df, groups=df["participant"]).fit()


def test_returns_ggplot_and_builds(fitted_model):
    p = random_effects_plot(fitted_model, title="Participant intercepts")
    assert isinstance(p, ggplot)
    p.draw(show=False)  # raises if the plot cannot be rendered


def test_one_point_per_group(fitted_model):
    # A random-intercept model has one BLUP per participant; the plotted data
    # should carry exactly that many rows.
    p = random_effects_plot(fitted_model)
    assert len(p.data) == len(fitted_model.random_effects)


def test_interval_present_for_mixedlm(fitted_model):
    # statsmodels exposes random_effects_cov, so the plot should add an interval.
    p = random_effects_plot(fitted_model)
    assert p.data["low"].notna().all()
    assert (p.data["high"] >= p.data["low"]).all()


def test_random_slope_facets():
    df = _grouped_data()
    res = smf.mixedlm("RT ~ condition", df, groups=df["participant"],
                      re_formula="~word_frequency").fit()
    p = random_effects_plot(res)
    p.draw(show=False)
    # Two random-effect terms (intercept + slope) -> two distinct terms plotted.
    assert p.data["term"].nunique() == 2


def test_no_interval_fallback():
    # An object with random_effects but no random_effects_cov: points only.
    class Stub:
        random_effects = {
            "A": pd.Series([1.2], index=["Group"]),
            "B": pd.Series([-0.7], index=["Group"]),
            "C": pd.Series([0.3], index=["Group"]),
        }

    p = random_effects_plot(Stub())
    assert isinstance(p, ggplot)
    p.draw(show=False)
    assert p.data["low"].isna().all()


def test_rejects_non_model():
    with pytest.raises(TypeError):
        random_effects_plot(pd.DataFrame({"x": [1, 2, 3]}))


def test_rejects_empty_random_effects():
    class Empty:
        random_effects = {}

    with pytest.raises(ValueError):
        random_effects_plot(Empty())
