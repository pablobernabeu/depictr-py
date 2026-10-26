"""Tests for depictr.mixed (random-effects caterpillar plot)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; the default Tk backend is broken here

import warnings

import pandas as pd
import pytest
import statsmodels.formula.api as smf
from matplotlib.collections import PathCollection
from plotnine import ggplot

from depictr.data import lexical_decision
from depictr.mixed import random_effects_plot


def _grouped_data():
    """lexical_decision, whose 24 participants each carry a random intercept."""
    return lexical_decision()


@pytest.fixture(scope="module")
def fitted_model():
    df = _grouped_data()
    return smf.mixedlm("RT ~ condition", df, groups=df["participant"]).fit()


def test_participant_model_matches_lme4_and_is_not_singular():
    # The documented example. The one it replaced grouped the simulated
    # wellbeing survey by region, which had no variance to find: cov_re was
    # 4.7e-11 and statsmodels warned that the covariance was singular. Here
    # each participant carries a random intercept, and R's lme4 on the same data,
    # lmer(RT ~ condition + (1 | participant)), gives 6050.6824
    # (dev/goldens/datasets.R).
    df = _grouped_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = smf.mixedlm("RT ~ condition", df, groups=df["participant"]).fit()
    assert float(fit.cov_re.iloc[0, 0]) == pytest.approx(6050.6824, rel=1e-3)
    assert len(fit.random_effects) == 24


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


def test_each_panel_sorted_by_its_own_term():
    # A random intercept and a random slope for word frequency, which order the
    # participants differently. Each panel runs up the axis from its own term's
    # most negative effect, as in the R twin, and each label sits beside its
    # own group's point.
    df = _grouped_data()
    res = smf.mixedlm("RT ~ condition", df, groups=df["participant"],
                      re_formula="~word_frequency").fit()
    re = pd.DataFrame(res.random_effects).T
    expected = {term: [str(g) for g in re[term].sort_values().index]
                for term in re.columns}
    assert expected["Group"] != expected["word_frequency"]

    p = random_effects_plot(res)
    p.draw(show=False)
    assert list(p.layout.layout["term"]) == ["Group", "word_frequency"]
    for term, ax in zip(p.layout.layout["term"], p.axs, strict=True):
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == expected[term]
        points = next(c for c in ax.collections
                      if isinstance(c, PathCollection)).get_offsets()
        drawn = [x for x, _ in sorted(points.tolist(), key=lambda xy: xy[1])]
        assert drawn == pytest.approx([re.loc[g, term] for g in labels])


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
