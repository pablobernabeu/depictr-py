"""Tests for the bundled data, the theme/scales, and the model-or-frame input."""

import copy
import warnings
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from plotnine import aes, geom_bar, geom_point, ggplot, theme
from plotnine.scales.scale import scale

from depictr import (
    DATASETS,
    coefficient_plot,
    compare_models,
    depictr_palette,
    ecdf_plot,
    scale_colour_depictr,
    scale_fill_depictr,
    theme_depictr,
    tidy_estimates,
    wellbeing_survey,
)
from depictr.palette import NA_VALUE, OKABE_ITO


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


def _drawn(p, aesthetic):
    """The colours a plot's first layer is drawn in, read back after drawing."""
    built = copy.deepcopy(p)
    figure = built.draw(show=False)
    try:
        return [str(c).lower() for c in built.layers[0].data[aesthetic]]
    finally:
        plt.close(figure)


def _groups(k, per_group=5):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "g": np.repeat([f"g{i}" for i in range(1, k + 1)], per_group),
        "x": rng.normal(size=k * per_group),
    })


def test_a_ninth_group_is_interpolated_not_drawn_as_missing():
    # The scale used to hold eight fixed values, so plotnine drew a ninth group
    # in the missing-value grey, with only its own generic warning. The plot
    # names a colour for each group when it is made, so the interpolation
    # warning comes then, not when the figure is drawn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = ecdf_plot(_groups(9), "x", group="g")
        colours = set(_drawn(p, "color"))
    messages = [str(w.message) for w in caught]
    assert any("9 colours interpolated" in m for m in messages)
    assert not any("can return a maximum of" in m for m in messages)
    assert len(colours) == 9
    assert NA_VALUE not in colours


def test_the_fill_scale_interpolates_past_eight_groups():
    p = ggplot(_groups(9), aes("g", fill="g")) + geom_bar() + scale_fill_depictr()
    with pytest.warns(UserWarning, match="9 colours interpolated"):
        fills = set(_drawn(p, "fill"))
    assert len(fills) == 9
    assert NA_VALUE not in fills


def test_eight_groups_keep_the_okabe_ito_colours_without_a_warning():
    p = (ggplot(_groups(8), aes("x", "x", colour="g")) + geom_point()
         + scale_colour_depictr())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        colours = _drawn(p, "color")
    assert not any("interpolated" in str(w.message) for w in caught)
    assert list(dict.fromkeys(colours)) == OKABE_ITO


def test_missing_rows_stay_grey_beside_nine_interpolated_groups():
    df = _groups(9)
    df.loc[0, "g"] = None
    p = (ggplot(df, aes("x", "x", colour="g")) + geom_point()
         + scale_colour_depictr())
    with pytest.warns(UserWarning, match="9 colours interpolated"):
        colours = _drawn(p, "color")
    assert colours[0] == NA_VALUE
    assert NA_VALUE not in colours[1:]
    assert len(set(colours[1:])) == 9


def test_an_explicit_n_still_fixes_the_palette():
    p = (ggplot(_groups(3), aes("x", "x", colour="g")) + geom_point()
         + scale_colour_depictr(n=5))
    assert list(dict.fromkeys(_drawn(p, "color"))) == depictr_palette(5)[:3]


def test_the_scales_take_the_contrast_first_order():
    contrast_first = depictr_palette(order="contrast")
    p = (ggplot(_groups(3), aes("x", "x", colour="g")) + geom_point()
         + scale_colour_depictr(order="contrast"))
    assert list(dict.fromkeys(_drawn(p, "color"))) == contrast_first[:3]
    p = ggplot(_groups(3), aes("g", fill="g")) + geom_bar() + scale_fill_depictr(
        order="contrast")
    assert list(dict.fromkeys(_drawn(p, "fill"))) == contrast_first[:3]
    with pytest.raises(ValueError, match="`order` must be"):
        scale_colour_depictr(order="brightest")


def test_the_contrast_first_scale_warns_past_four_groups():
    # As past eight, the warning comes when the plot is drawn, since that is
    # when the scale learns how many groups there are.
    p = (ggplot(_groups(5), aes("x", "x", colour="g")) + geom_point()
         + scale_colour_depictr(order="contrast"))
    with pytest.warns(UserWarning, match="^Past four colours"):
        colours = list(dict.fromkeys(_drawn(p, "color")))
    assert colours == depictr_palette(order="contrast")[:5]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _drawn(ggplot(_groups(4), aes("x", "x", colour="g")) + geom_point()
               + scale_colour_depictr(order="contrast"), "color")
    assert not any("Past four" in str(w.message) for w in caught)


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


def _mixed_data():
    # lexical_decision has no participant column, so borrow the one the
    # mixed-model tests simulate, with its random intercept.
    from test_mixed import _grouped_data

    return _grouped_data()


def test_tidy_estimates_keeps_only_the_fixed_effects_of_a_mixedlm():
    # Regression: every entry of params was kept, so the random-effect variance
    # and covariance entries, which statsmodels divides by the residual scale,
    # were drawn as coefficients. The R twin keeps lme4::fixef() only.
    smf = pytest.importorskip("statsmodels.formula.api")
    df = _mixed_data()
    fit = smf.mixedlm("RT ~ condition", df, groups=df["participant"],
                      re_formula="~word_frequency").fit()
    tidy = tidy_estimates(fit)
    assert list(tidy["term"]) == list(fit.fe_params.index)
    assert not tidy["term"].str.contains("Var|Cov").any()
    np.testing.assert_allclose(tidy["estimate"], fit.fe_params)
    # Wald intervals on the normal distribution, as the R twin's merMod method
    # computes them from lme4::fixef() and its standard errors.
    z = NormalDist().inv_cdf(0.975)
    np.testing.assert_allclose(tidy["conf_low"], fit.fe_params - z * fit.bse_fe)
    np.testing.assert_allclose(tidy["conf_high"], fit.fe_params + z * fit.bse_fe)
    assert set(coefficient_plot(fit).data["term"]) == {"condition[T.unrelated]"}
    assert set(compare_models({"mixed": fit}).data["term"]) == {
        "condition[T.unrelated]"}


def test_tidy_estimates_reads_a_mixedlm_fitted_from_arrays():
    # Regression: fitted from arrays, params is a bare ndarray, and reading its
    # index raised AttributeError. The fixed-effect names then come from the
    # design matrix, and the variance entry is dropped as for a formula fit.
    sm = pytest.importorskip("statsmodels.api")
    df = _mixed_data()
    fit = sm.MixedLM(df["RT"].to_numpy(),
                     sm.add_constant(df["word_frequency"].to_numpy()),
                     groups=df["participant"].to_numpy()).fit()
    tidy = tidy_estimates(fit)
    assert list(tidy["term"]) == ["const", "x1"]
    np.testing.assert_allclose(tidy["estimate"], fit.fe_params)
    np.testing.assert_allclose(tidy[["conf_low", "conf_high"]],
                               np.asarray(fit.conf_int())[: fit.k_fe])


def test_tidy_estimates_reads_an_ols_fitted_from_arrays():
    # Regression: the same AttributeError broke every plot built on
    # tidy_estimates() for a model fitted from arrays.
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 2))
    y = 1 + x @ np.array([0.5, -0.3]) + rng.normal(size=60)
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    tidy = tidy_estimates(fit)
    assert list(tidy["term"]) == ["const", "x1", "x2"]
    np.testing.assert_allclose(tidy["estimate"], fit.params)
    np.testing.assert_allclose(tidy[["conf_low", "conf_high"]], fit.conf_int())
    p = coefficient_plot(fit)
    p.draw(show=False)
    assert set(p.data["term"]) == {"x1", "x2"}
    compare_models({"arrays": fit}).draw(show=False)


def test_tidy_estimates_returns_string_terms_for_numbered_columns():
    # Regression: a model fitted on a data frame with numbered columns, as
    # pd.DataFrame() makes from an array, kept integer terms. The intercept
    # filter in coefficient_plot() and compare_models() then failed on them.
    # The R twin returns character terms.
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(60, 2)))
    y = 1 + x.to_numpy() @ np.array([0.5, -0.3]) + rng.normal(size=60)
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    assert list(tidy_estimates(fit)["term"]) == ["const", "0", "1"]
    p = coefficient_plot(fit)
    p.draw(show=False)
    assert set(p.data["term"]) == {"0", "1"}
    compare_models({"numbered": fit}).draw(show=False)


def test_tidy_estimates_refuses_estimates_it_cannot_name():
    # Bare estimates with no design matrix to name them, held in an array and
    # in a list. A list's index attribute is a method, which must not pass for
    # a set of names.
    class Unnamed:
        params = np.array([1.0, 2.0])

        def conf_int(self, alpha=0.05):
            return np.array([[0.5, 1.5], [1.5, 2.5]])

    class Listed(Unnamed):
        params = [1.0, 2.0]

    for model in (Unnamed(), Listed()):
        with pytest.raises(TypeError, match="Cannot name the estimates"):
            tidy_estimates(model)


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
