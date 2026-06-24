"""Smoke tests: every regression-diagnostic plot builds without error."""

import numpy as np
import pytest
from plotnine import ggplot

from depictr.data import clinical_trial, crop_yield
from depictr.diagnostics import (
    binned_residual_plot,
    influence_plot,
    qq_plot,
    vif_plot,
)


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


@pytest.fixture(scope="module")
def ols():
    smf = pytest.importorskip("statsmodels.formula.api")
    # "yield" is a Python keyword, so the formula needs Q("yield").
    return smf.ols('Q("yield") ~ fertiliser + treatment + rainfall + soil_ph',
                   crop_yield()).fit()


@pytest.fixture(scope="module")
def glm():
    smf = pytest.importorskip("statsmodels.formula.api")
    sm = pytest.importorskip("statsmodels.api")
    return smf.glm("adverse_event ~ biomarker + age + arm", clinical_trial(),
                   family=sm.families.Binomial()).fit()


def test_qq_plot_from_model_and_from_array(ols):
    assert _builds(qq_plot(ols, title="QQ"))
    # Also accepts a 1-D array of residuals directly.
    assert _builds(qq_plot(np.asarray(ols.resid)))


def test_influence_plot(ols, glm):
    assert _builds(influence_plot(ols))
    assert _builds(influence_plot(glm))


def test_influence_plot_rejects_raw_array():
    with pytest.raises(TypeError):
        influence_plot([1.0, 2.0, 3.0])


def test_vif_plot(ols, glm):
    assert _builds(vif_plot(ols, title="VIF"))
    assert _builds(vif_plot(glm))


def test_binned_residual_plot(glm):
    assert _builds(binned_residual_plot(glm))
    assert _builds(binned_residual_plot(glm, n_bins=8))


def test_binned_residual_plot_rejects_non_binomial(ols):
    with pytest.raises(ValueError):
        binned_residual_plot(ols)
