"""Arguments are checked where they enter, with the R twin's messages.

A `conf_level` given as a percentage, 95, gave NaN intervals in most of the
functions that take one and an unrelated error in the rest. The R twin's
tests/testthat/test-conf-level.R pins the same message. The other checks here
cover arguments that were ignored, misread or left to fail deep inside
matplotlib or NumPy.
"""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from plotnine import ggplot

import depictr as dp

CY = dp.crop_yield()
LD = dp.lexical_decision()
CT = dp.clinical_trial()

CONF_LEVEL_MESSAGE = "`conf_level` must be a single number strictly between 0 and 1."
CONF_LEVEL_MESSAGE_NONE = (
    "`conf_level` must be a single number strictly between 0 and 1, or None to "
    "leave the limits out."
)
BAD_LEVELS = [95, 0, 1, -0.5, np.inf, np.nan, True, "0.95", [0.9, 0.95]]


def _ols():
    smf = pytest.importorskip("statsmodels.formula.api")
    return smf.ols('Q("yield") ~ fertiliser + rainfall + treatment', CY).fit()


def _conf_level_calls():
    fit = _ols()
    estimates = dp.tidy_estimates(fit)
    return {
        "tidy_estimates on a model": lambda cl: dp.tidy_estimates(fit, conf_level=cl),
        "tidy_estimates on a frame": lambda cl: dp.tidy_estimates(estimates,
                                                                  conf_level=cl),
        "coefficient_plot": lambda cl: dp.coefficient_plot(fit, conf_level=cl),
        "compare_models": lambda cl: dp.compare_models({"A": fit, "B": fit},
                                                       conf_level=cl),
        "estimation_plot": lambda cl: dp.estimation_plot(
            CY, "yield", "treatment", conf_level=cl, n_boot=20, seed=1),
        "effects_plot": lambda cl: dp.effects_plot(fit, "fertiliser", conf_level=cl),
        "interaction_plot": lambda cl: dp.interaction_plot(
            fit, "fertiliser", "treatment", conf_level=cl),
        "group_comparison_plot": lambda cl: dp.group_comparison_plot(
            CY, "yield", "treatment", conf_level=cl),
    }


CONF_LEVEL_FUNCTIONS = [
    "tidy_estimates on a model", "tidy_estimates on a frame", "coefficient_plot",
    "compare_models", "estimation_plot", "effects_plot", "interaction_plot",
    "group_comparison_plot",
]


@pytest.mark.parametrize("bad", [*BAD_LEVELS, None], ids=repr)
@pytest.mark.parametrize("name", CONF_LEVEL_FUNCTIONS)
def test_a_level_outside_zero_and_one_is_refused(name, bad):
    call = _conf_level_calls()[name]
    with pytest.raises(ValueError, match=re.escape(CONF_LEVEL_MESSAGE)):
        call(bad)


def test_a_level_inside_zero_and_one_is_still_accepted():
    for call in _conf_level_calls().values():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            call(np.float64(0.9))


def _scores():
    rng = np.random.default_rng(1)
    score = rng.uniform(0.05, 0.95, 400)
    return (rng.uniform(size=400) < score).astype(int), score


@pytest.mark.parametrize("bad", BAD_LEVELS, ids=repr)
def test_survival_and_calibration_keep_none_to_leave_the_limits_out(bad):
    y, score = _scores()
    with pytest.raises(ValueError, match=re.escape(CONF_LEVEL_MESSAGE_NONE)):
        dp.calibration_plot(y, score, conf_level=bad)
    pytest.importorskip("lifelines")
    with pytest.raises(ValueError, match=re.escape(CONF_LEVEL_MESSAGE_NONE)):
        dp.survival_plot(CT["time"], CT["event"], conf_level=bad, logrank=False)


def _density_alpha(p):
    for layer in p.layers:
        if "density" in type(layer.geom).__name__:
            return layer.geom.aes_params["alpha"]
    raise AssertionError("no density layer")


def test_explore_distribution_honours_an_explicit_alpha():
    # A grouped density was drawn at 0.2 whatever `alpha` said.
    p = dp.explore_distribution(LD, "RT", group="condition", alpha=0.9)
    assert _density_alpha(p) == 0.9
    # The defaults are unchanged: 0.2 for a grouped density, 0.6 otherwise.
    assert _density_alpha(dp.explore_distribution(LD, "RT", group="condition")) == 0.2
    assert _density_alpha(dp.explore_distribution(LD, "RT")) == 0.6
    hist = dp.explore_distribution(LD, "RT", kind="histogram")
    assert hist.layers[0].geom.aes_params["alpha"] == 0.6
    # Over a histogram, the grouped density stays unfilled, and the bars take
    # the alpha.
    both = dp.explore_distribution(LD, "RT", group="condition", kind="both",
                                   alpha=0.9)
    assert both.layers[0].geom.aes_params["alpha"] == 0.9
    assert _density_alpha(both) == 0


@pytest.mark.parametrize("bad", [-1, 0, 0.5, np.nan, np.inf, True, "3"], ids=repr)
@pytest.mark.parametrize("kind", ["qualitative", "sequential", "diverging"])
def test_depictr_palette_refuses_an_n_below_one(bad, kind):
    # depictr_palette(-1) gave seven colours, (0) none and (0, "sequential")
    # seven. The message is the R twin's.
    with pytest.raises(ValueError,
                       match=re.escape("`n` must be a single positive number.")):
        dp.depictr_palette(bad, kind=kind)


@pytest.mark.parametrize("kind", ["qualitative", "sequential", "diverging"])
def test_depictr_palette_truncates_a_fractional_n_as_r_does(kind):
    assert dp.depictr_palette(2.7, kind=kind) == dp.depictr_palette(2, kind=kind)
    assert dp.depictr_palette(np.int64(3), kind=kind) == dp.depictr_palette(3, kind=kind)
    assert dp.depictr_palette(1.0, kind=kind) == dp.depictr_palette(1, kind=kind)


def test_simulate_cvd_takes_one_colour_string():
    # A string was read character by character: '#005b96' failed on '#', and
    # 'rgb' came back as three colours. The R twin returns #275295.
    assert dp.simulate_cvd("#005b96", "deutan") == ["#275295"]
    assert dp.simulate_cvd("red", "deutan") == dp.simulate_cvd(["red"], "deutan")
    # "rgb" names no colour, and is now refused as the one colour it is.
    with pytest.raises(ValueError, match="'rgb'"):
        dp.simulate_cvd("rgb", "deutan")


def test_simulate_cvd_returns_an_empty_list_for_no_colours():
    assert dp.simulate_cvd([], "deutan") == []
    # The deficiency is still checked.
    with pytest.raises(ValueError, match="`deficiency` must be one of"):
        dp.simulate_cvd([], "deuteranopia")


def test_palette_safety_reads_a_string_as_one_colour():
    # 'cmyk' was read as four colours and reported safe.
    for one in ("cmyk", "#005b96"):
        with pytest.raises(ValueError, match="needs at least two colours"):
            dp.palette_safety(one)


def test_coefficient_plot_refuses_an_unknown_order():
    fit = _ols()
    with pytest.raises(ValueError, match="`order` must be one of"):
        dp.coefficient_plot(fit, order="descendng")


def test_coefficient_plot_takes_none_as_model_order():
    # "none" is the R twin's name for model order.
    fit = _ols()
    model = dp.coefficient_plot(fit, order="model")
    none = dp.coefficient_plot(fit, order="none")
    assert isinstance(none, ggplot)
    assert (list(none.data["label"].cat.categories)
            == list(model.data["label"].cat.categories))


@pytest.mark.parametrize("bad", [2.5, 0, -3, np.nan, np.inf, True, "200"], ids=repr)
def test_estimation_plot_refuses_an_n_boot_that_is_not_a_positive_integer(bad):
    with pytest.raises(ValueError, match=re.escape("`n_boot` must be a positive integer.")):
        dp.estimation_plot(CY, "yield", "treatment", n_boot=bad)


def test_estimation_plot_takes_a_whole_number_n_boot_given_as_a_float():
    p = dp.estimation_plot(CY, "yield", "treatment", n_boot=200.0, seed=1)
    q = dp.estimation_plot(CY, "yield", "treatment", n_boot=200, seed=1)
    assert p.differences.equals(q.differences)
    figure = p.draw(show=False)
    try:
        assert "Bootstrap 95% CI (200 resamples)" in [t.get_text() for t in figure.texts]
    finally:
        plt.close(figure)
