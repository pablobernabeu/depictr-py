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


def test_coefficient_plot_order_increases_up_the_axis_as_in_r():
    # Regression: "ascending" and "descending" were drawn the opposite way round
    # from the docstring and from the R twin. plotnine, like ggplot2, draws the
    # first category of a discrete y axis at the bottom, so "ascending" must list
    # the smallest estimate first and put the largest at the top. The R suite
    # pins the same direction (test-coefficient_plot.R).
    sm = pytest.importorskip("statsmodels.formula.api")
    fit = sm.ols('Q("yield") ~ rainfall + fertiliser + soil_ph',
                 DATASETS["crop_yield"]()).fit()
    slopes = fit.params.drop("Intercept")
    p = coefficient_plot(fit, order="ascending")
    assert list(p.data["term"].cat.categories) == list(slopes.sort_values().index)
    p = coefficient_plot(fit, order="descending")
    assert list(p.data["term"].cat.categories) == list(
        slopes.sort_values(ascending=False).index)
    # "model" keeps the model's order from top to bottom.
    p = coefficient_plot(fit)
    assert list(p.data["term"].cat.categories) == list(slopes.index)[::-1]


def test_coefficient_plot_ties_keep_model_order_as_in_r():
    # R's order() is stable, and a stable sort here stacks tied terms the same
    # way. The R suite pins the same vectors in test-coefficient_plot.R. There
    # are 24 terms because with only a few, pandas' default sort, which is not
    # stable, can leave the ties in order by chance.
    est = pd.DataFrame({"term": [f"t{i:02d}" for i in range(1, 25)],
                        "estimate": ([0.2, 0.4, 0.1, 0.3, 0.0] * 5)[:24]})
    est["conf_low"] = est["estimate"] - 0.05
    est["conf_high"] = est["estimate"] + 0.05
    up = [5, 10, 15, 20, 3, 8, 13, 18, 23, 1, 6, 11, 16, 21, 4, 9, 14, 19, 24,
          2, 7, 12, 17, 22]
    down = [2, 7, 12, 17, 22, 4, 9, 14, 19, 24, 1, 6, 11, 16, 21, 3, 8, 13, 18,
            23, 5, 10, 15, 20]
    p = coefficient_plot(est, order="ascending")
    assert list(p.data["term"].cat.categories) == [f"t{i:02d}" for i in up]
    p = coefficient_plot(est, order="descending")
    assert list(p.data["term"].cat.categories) == [f"t{i:02d}" for i in down]
