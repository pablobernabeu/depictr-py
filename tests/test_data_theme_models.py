"""Tests for the bundled data, the theme/scales, and the model-or-frame input."""

import pandas as pd
import pytest
from plotnine import theme
from plotnine.scales.scale import scale

from depictr import (
    DATASETS,
    coefficient_plot,
    scale_colour_depictr,
    scale_fill_depictr,
    theme_depictr,
    tidy_estimates,
    wellbeing_survey,
)


def test_datasets_load_and_are_reproducible():
    for name, loader in DATASETS.items():
        df = loader()
        assert isinstance(df, pd.DataFrame) and len(df) > 0, name
        # Deterministic: two calls give identical data.
        pd.testing.assert_frame_equal(loader(), df)


def test_wellbeing_has_missingness():
    df = wellbeing_survey()
    assert df["income"].isna().any()


def test_theme_returns_a_theme_and_validates_grid():
    assert isinstance(theme_depictr(), theme)
    assert isinstance(theme_depictr(grid="x"), theme)
    with pytest.raises(ValueError):
        theme_depictr(grid="diagonal")


def test_scales_return_scale_objects():
    assert isinstance(scale_colour_depictr(), scale)
    assert isinstance(scale_fill_depictr(3), scale)


def test_tidy_estimates_from_frame_normalises_columns():
    raw = pd.DataFrame({
        "term": ["a", "b"], "estimate": [1.0, -1.0],
        "conf.low": [0.5, -1.5], "conf.high": [1.5, -0.5],
    })
    tidy = tidy_estimates(raw)
    assert list(tidy.columns) == ["term", "estimate", "conf_low", "conf_high"]


def test_tidy_estimates_rejects_bad_input():
    with pytest.raises((TypeError, ValueError)):
        tidy_estimates(pd.DataFrame({"term": ["a"]}))
    with pytest.raises(TypeError):
        tidy_estimates(42)


def test_coefficient_plot_from_fitted_model():
    sm = pytest.importorskip("statsmodels.formula.api")
    fit = sm.ols('Q("yield") ~ fertiliser + treatment', DATASETS["crop_yield"]()).fit()
    p = coefficient_plot(fit)
    p.draw(show=False)  # builds without error
