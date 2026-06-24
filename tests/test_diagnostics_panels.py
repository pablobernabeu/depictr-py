"""Smoke tests: the multi-panel regression reports build and save."""

import os
import tempfile

import pytest

from depictr.data import crop_yield
from depictr.diagnostics_panels import model_report, residual_diagnostics_plot


def _builds(composition):
    """A composition is fine if it draws off-screen without error."""
    composition.draw(show=False)
    return True


@pytest.fixture(scope="module")
def ols():
    smf = pytest.importorskip("statsmodels.formula.api")
    # "yield" is a Python keyword, so the formula needs Q("yield").
    return smf.ols('Q("yield") ~ fertiliser + treatment + rainfall + soil_ph',
                   crop_yield()).fit()


def test_residual_diagnostics_plot(ols):
    assert _builds(residual_diagnostics_plot(ols))
    assert _builds(residual_diagnostics_plot(ols, title="Residual diagnostics"))


def test_model_report(ols):
    assert _builds(model_report(ols))
    assert _builds(model_report(ols, title="Model report"))


def test_residual_diagnostics_plot_saves(ols, tmp_path):
    fn = tmp_path / "diagnostics.png"
    residual_diagnostics_plot(ols).save(fn, width=8, height=6, dpi=80,
                                        verbose=False)
    assert fn.exists() and fn.stat().st_size > 0


def test_panels_reject_raw_array():
    with pytest.raises(TypeError):
        residual_diagnostics_plot([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        model_report([1.0, 2.0, 3.0])
