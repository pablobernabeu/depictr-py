"""The multi-panel regression reports: they build, save and read GLMs right."""

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from depictr.data import clinical_trial, crop_yield
from depictr.diagnostics_panels import model_report, residual_diagnostics_plot


def _builds(composition):
    """A composition is fine if it draws off-screen without error."""
    composition.draw(show=False)
    return True


def _panels(composition):
    """The plots of a composition, row by row."""
    if isinstance(composition, ggplot):
        return [composition]
    return [plot for item in composition.items for plot in _panels(item)]


def _share_outside_band(values, level=0.95):
    """Share of a sample outside the pointwise band of R's qq_plot().

    The band is the R twin's default: the reference line through the quartiles
    and, at each of R's ppoints() positions, the large-sample standard error of
    the order statistic, slope / dnorm(z) * sqrt(p * (1 - p) / n).
    """
    from scipy import stats

    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    a = 3 / 8 if n <= 10 else 0.5
    p = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
    z = stats.norm.ppf(p)
    q_theory = stats.norm.ppf([0.25, 0.75])
    q_sample = np.percentile(v, [25, 75])
    slope = (q_sample[1] - q_sample[0]) / (q_theory[1] - q_theory[0])
    centre = q_sample[0] + slope * (z - q_theory[0])
    se = slope / stats.norm.pdf(z) * np.sqrt(p * (1 - p) / n)
    crit = stats.norm.ppf(1 - (1 - level) / 2)
    return float(np.mean(np.abs(v - centre) > crit * se))


@pytest.fixture(scope="module")
def ols():
    smf = pytest.importorskip("statsmodels.formula.api")
    # "yield" is a Python keyword, so the formula needs Q("yield").
    return smf.ols('Q("yield") ~ fertiliser + treatment + rainfall + soil_ph',
                   crop_yield()).fit()


@pytest.fixture(scope="module")
def glm_fits():
    sm = pytest.importorskip("statsmodels.api")
    smf = pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(5)
    n = 300
    d = pd.DataFrame({"x": rng.normal(size=n)})
    d["y_binary"] = rng.binomial(1, 1 / (1 + np.exp(-(-0.2 + 0.9 * d["x"]))))
    d["y_count"] = rng.poisson(np.exp(0.4 + 0.5 * d["x"]))
    d["y_cont"] = 1 + 0.5 * d["x"] + rng.normal(size=n)
    return {
        "binomial": smf.glm("y_binary ~ x", d,
                            family=sm.families.Binomial()).fit(),
        "logit": smf.logit("y_binary ~ x", d).fit(disp=0),
        "probit": smf.probit("y_binary ~ x", d).fit(disp=0),
        "poisson": smf.glm("y_count ~ x", d, family=sm.families.Poisson()).fit(),
        "gaussian": smf.glm("y_cont ~ x", d).fit(),
    }


@pytest.fixture(scope="module")
def correct_logistic():
    """A logistic model fitted to data simulated from it, n = 400."""
    sm = pytest.importorskip("statsmodels.api")
    smf = pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(2026)
    n = 400
    d = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    eta = -0.3 + 1.2 * d["x1"] - 0.8 * d["x2"]
    d["y"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return {
        "glm": smf.glm("y ~ x1 + x2", d, family=sm.families.Binomial()).fit(),
        "logit": smf.logit("y ~ x1 + x2", d).fit(disp=0),
    }


def test_residual_diagnostics_plot(ols):
    assert _builds(residual_diagnostics_plot(ols))
    # A title on a multi-panel grid is dropped with a warning: plotnine
    # compositions cannot carry a figure-level title.
    with pytest.warns(UserWarning, match="figure-level title"):
        assert _builds(residual_diagnostics_plot(ols,
                                                 title="Residual diagnostics"))


def test_model_report(ols):
    assert _builds(model_report(ols))
    with pytest.warns(UserWarning, match="figure-level title"):
        assert _builds(model_report(ols, title="Model report"))


def test_residual_diagnostics_plot_saves(ols, tmp_path):
    fn = tmp_path / "diagnostics.png"
    residual_diagnostics_plot(ols).save(fn, width=8, height=6, dpi=80,
                                        verbose=False)
    assert fn.exists() and fn.stat().st_size > 0


def test_residual_smoothers_are_drawn_in_the_reference_grey(ols, glm_fits):
    # The lowess smoothers were drawn in the orange #e69f00, a group colour of
    # the palette, where the R twin draws them in depictr_reference(), grey60.
    from plotnine import geom_smooth

    for fit in (ols, glm_fits["poisson"]):
        smooths = [layer.geom.aes_params.get("color")
                   for panel in _panels(residual_diagnostics_plot(fit, seed=1))
                   for layer in panel.layers
                   if isinstance(layer.geom, geom_smooth)]
        assert smooths == ["#999999"] * 3


def test_panels_reject_raw_array():
    with pytest.raises(TypeError):
        residual_diagnostics_plot([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        model_report([1.0, 2.0, 3.0])


@pytest.mark.parametrize("name",
                         ["binomial", "logit", "probit", "poisson", "gaussian"])
def test_glm_panels(glm_fits, name):
    # residual_diagnostics_plot() and model_report() read model.resid, which
    # no GLM, Logit or Probit result has, so both raised AttributeError for
    # every one of them, called as here with the default arguments.
    fit = glm_fits[name]
    assert _builds(residual_diagnostics_plot(fit))
    assert _builds(model_report(fit))

    binomial = name in {"binomial", "logit", "probit"}
    first = "Binned residuals" if binomial else "Residuals vs fitted"
    report = _panels(residual_diagnostics_plot(fit, seed=1))
    assert [p.labels.title for p in report] == [
        first, "Scale-location", "Normal Q-Q", "Residuals vs leverage"]
    assert report[2].labels.y == "Quantile residuals"
    if not binomial:
        assert report[0].labels.y == "Deviance residuals"
    dashboard = _panels(model_report(fit, seed=1))
    assert [p.labels.title for p in dashboard] == [
        "Coefficients", first, "Normal Q-Q"]
    # One seed, one set of randomised residuals, in either function.
    np.testing.assert_array_equal(dashboard[2].data["sample"],
                                  report[2].data["sample"])


@pytest.mark.parametrize("kind", ["glm", "logit"])
def test_correct_logistic_model_stays_inside_the_band(correct_logistic, kind):
    # Quantile residuals are standard normal under a correct model, so about
    # 5% of them fall outside a pointwise 95% band. The studentised Pearson
    # residuals of a binary model are not normal and leave it far more often.
    from depictr.diagnostics import _studentised_residuals

    fit = correct_logistic[kind]
    qq = _panels(residual_diagnostics_plot(fit, seed=1))[2]
    assert len(qq.data) == 400
    assert _share_outside_band(qq.data["sample"]) < 0.10
    assert _share_outside_band(_studentised_residuals(fit)) > 0.10


def test_glm_and_logit_fits_draw_the_same_quantile_residuals(correct_logistic):
    from depictr.diagnostics import _quantile_residuals

    by_glm, glm_ok = _quantile_residuals(correct_logistic["glm"], seed=4)
    by_logit, logit_ok = _quantile_residuals(correct_logistic["logit"], seed=4)
    assert glm_ok and logit_ok
    np.testing.assert_allclose(by_glm, by_logit, atol=1e-6)


def test_classic_panels_for_ols_and_on_request(ols, glm_fits):
    report = _panels(residual_diagnostics_plot(ols))
    assert report[0].labels.title == "Residuals vs fitted"
    assert report[0].labels.y == "Residuals"
    assert report[2].labels.y == "Sample quantiles"

    for name in ("binomial", "logit"):
        classic = _panels(residual_diagnostics_plot(glm_fits[name],
                                                    glm_panels=False))
        assert classic[0].labels.title == "Residuals vs fitted"
        assert classic[0].labels.y == "Deviance residuals"
        assert classic[2].labels.y == "Sample quantiles"
    assert _builds(residual_diagnostics_plot(glm_fits["logit"],
                                             glm_panels=False))
    dashboard = _panels(model_report(glm_fits["poisson"], glm_panels=False))
    assert dashboard[1].labels.y == "Deviance residuals"
    assert dashboard[2].labels.y == "Sample quantiles"


def test_logit_fitted_values_are_probabilities(glm_fits):
    # A Logit result's fittedvalues is the linear predictor. The panels plot
    # the fitted probabilities, as for the binomial GLM of the same model.
    by_logit = _panels(residual_diagnostics_plot(glm_fits["logit"],
                                                 glm_panels=False))[0]
    by_glm = _panels(residual_diagnostics_plot(glm_fits["binomial"],
                                               glm_panels=False))[0]
    assert by_logit.data["fitted"].between(0, 1).all()
    np.testing.assert_allclose(by_logit.data["fitted"], by_glm.data["fitted"],
                               atol=1e-6)
    np.testing.assert_allclose(by_logit.data["resid"], by_glm.data["resid"],
                               atol=1e-6)


def test_quantile_residuals_are_reproducible_and_leave_numpy_alone(glm_fits):
    from depictr.diagnostics import _quantile_residuals

    fit = glm_fits["binomial"]
    a, _ = _quantile_residuals(fit, seed=7)
    b, _ = _quantile_residuals(fit, seed=7)
    np.testing.assert_array_equal(a, b)
    # The randomisation draws from a generator of its own and leaves numpy's
    # global one alone.
    np.random.seed(1)
    before = np.random.random()
    np.random.seed(1)
    _quantile_residuals(fit, seed=7)
    assert np.random.random() == before


def test_quantile_residuals_are_standard_normal_under_the_model():
    sm = pytest.importorskip("statsmodels.api")
    from scipy import stats

    from depictr.diagnostics import _quantile_residuals

    rng = np.random.default_rng(11)
    n = 1200
    x = rng.normal(size=n)
    X = sm.add_constant(x)
    mu = np.exp(0.5 + 0.7 * x)
    log_link = sm.families.links.Log()
    fits = {
        "poisson": sm.GLM(rng.poisson(mu), X,
                          family=sm.families.Poisson()).fit(),
        "gamma": sm.GLM(rng.gamma(shape=4, scale=mu / 4), X,
                        family=sm.families.Gamma(link=log_link)).fit(),
        "gaussian": sm.GLM(1 + 0.5 * x + rng.normal(size=n), X).fit(),
        "binomial": sm.GLM(rng.binomial(1, 1 / (1 + np.exp(-x))), X,
                           family=sm.families.Binomial()).fit(),
    }
    for name, fit in fits.items():
        r, is_quantile = _quantile_residuals(fit, seed=1)
        assert is_quantile, name
        assert r.shape == (n,), name
        assert stats.kstest(r, "norm").pvalue > 0.05, name

    # Overdispersed counts fitted as Poisson are caught.
    y_over = rng.negative_binomial(1.2, 1.2 / (1.2 + mu))
    bad = sm.GLM(y_over, X, family=sm.families.Poisson()).fit()
    r, _ = _quantile_residuals(bad, seed=1)
    assert stats.kstest(r, "norm").pvalue < 0.01


def test_quantile_residuals_count_binomial_trials_either_way():
    # Trials come from a two-column (successes, failures) response or from
    # var_weights on a proportion, the two ways statsmodels takes them.
    sm = pytest.importorskip("statsmodels.api")
    from scipy import stats

    from depictr.diagnostics import _quantile_residuals

    rng = np.random.default_rng(21)
    n = 400
    x = rng.normal(size=n)
    X = sm.add_constant(x)
    trials = rng.integers(5, 21, size=n)
    succ = rng.binomial(trials, 1 / (1 + np.exp(-(-0.4 + 0.8 * x))))
    by_columns = sm.GLM(np.column_stack([succ, trials - succ]), X,
                        family=sm.families.Binomial()).fit()
    by_weights = sm.GLM(succ / trials, X, family=sm.families.Binomial(),
                        var_weights=trials).fit()
    r, is_quantile = _quantile_residuals(by_columns, seed=1)
    assert is_quantile
    assert abs(r.mean()) < 0.15
    assert stats.kstest(r, "norm").pvalue > 0.05
    np.testing.assert_allclose(_quantile_residuals(by_weights, seed=1)[0], r,
                               atol=1e-8)


def test_quantile_residuals_weight_the_gaussian_and_gamma_scales():
    # Under variance weights w, observation i has variance
    # scale * V(mu_i) / w_i, so each group's residuals have unit spread.
    sm = pytest.importorskip("statsmodels.api")

    from depictr.diagnostics import _quantile_residuals

    rng = np.random.default_rng(2026)
    n = 2000
    x = rng.normal(size=n)
    X = sm.add_constant(x)
    w = np.repeat([1.0, 16.0], n // 2)

    gauss = sm.GLM(1 + 0.5 * x + rng.normal(scale=2 / np.sqrt(w)), X,
                   var_weights=w).fit()
    r, _ = _quantile_residuals(gauss)
    for group in (1.0, 16.0):
        assert 0.9 < r[w == group].std(ddof=1) < 1.1
    # For a Gaussian model, the quantile residual is the weighted residual
    # divided by sigma.
    np.testing.assert_allclose(
        r, np.sqrt(w) * gauss.resid_response / np.sqrt(gauss.scale), atol=1e-8)

    mu = np.exp(0.5 + 0.3 * x)
    gamma = sm.GLM(rng.gamma(shape=4 * w, scale=mu / (4 * w)), X,
                   family=sm.families.Gamma(link=sm.families.links.Log()),
                   var_weights=w).fit()
    r, _ = _quantile_residuals(gamma)
    for group in (1.0, 16.0):
        assert 0.9 < r[w == group].std(ddof=1) < 1.1


def test_zero_weight_observations_have_no_quantile_residual():
    sm = pytest.importorskip("statsmodels.api")

    from depictr.diagnostics import _quantile_residuals

    rng = np.random.default_rng(17)
    n = 200
    x = rng.normal(size=n)
    X = sm.add_constant(x)
    w = np.ones(n)
    w[[2, 49]] = 0
    fit = sm.GLM(1 + x + rng.normal(size=n), X, var_weights=w).fit()
    r, _ = _quantile_residuals(fit)
    assert np.flatnonzero(np.isnan(r)).tolist() == [2, 49]
    # The Q-Q panel draws the others. model_report() is the route here because
    # statsmodels' get_influence(), which the leverage panels of
    # residual_diagnostics_plot() need, fails on a zero weight.
    qq = _panels(model_report(fit))[2]
    assert len(qq.data) == n - 2


def test_continuous_quantile_residuals_match_the_r_twin():
    # Printed by dev/goldens/quantile_residuals.R from the R twin's
    # quantile_residuals() at the commit of change set CS11, on fits converged
    # to the estimate statsmodels reaches. Neither family is randomised, so the
    # twins must agree.
    sm = pytest.importorskip("statsmodels.api")
    smf = pytest.importorskip("statsmodels.formula.api")

    from depictr.diagnostics import _quantile_residuals

    cy = crop_yield()
    w = np.resize([1.0, 2.0, 3.0], len(cy))
    gauss = smf.glm('Q("yield") ~ rainfall + fertiliser', cy,
                    var_weights=w).fit()
    gamma = smf.glm('Q("yield") ~ rainfall + fertiliser', cy,
                    family=sm.families.Gamma(link=sm.families.links.Log())
                    ).fit()
    pinned = {
        "gaussian": (gauss, [0.1998652750, 0.2434533383, -0.8194369197,
                             0.6792600704, -0.2612067488], 197.0),
        "gamma": (gamma, [0.4251697830, 0.4861144348, -0.5923176388,
                          0.9611763808, -0.1834470517], 261.4223406080),
    }
    for name, (fit, first_five, sum_of_squares) in pinned.items():
        r, is_quantile = _quantile_residuals(fit)
        assert is_quantile, name
        np.testing.assert_allclose(r[:5], first_five, atol=1e-6, err_msg=name)
        np.testing.assert_allclose(np.sum(r ** 2), sum_of_squares, rtol=1e-6,
                                   err_msg=name)


def test_unsupported_family_falls_back_to_deviance_residuals():
    sm = pytest.importorskip("statsmodels.api")

    from depictr.diagnostics import _quantile_residuals

    rng = np.random.default_rng(8)
    x = rng.normal(size=200)
    X = sm.add_constant(x)
    y = rng.negative_binomial(2, 2 / (2 + np.exp(0.5 + 0.4 * x)))
    fit = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=0.5)).fit()
    r, is_quantile = _quantile_residuals(fit, seed=1)
    assert not is_quantile
    np.testing.assert_allclose(r, fit.resid_deviance)
    qq = _panels(residual_diagnostics_plot(fit))[2]
    assert qq.labels.y == "Deviance residuals"


def test_clinical_trial_glm_report_draws():
    # The R twin's own example model, the binomial GLM of the adverse events.
    sm = pytest.importorskip("statsmodels.api")
    smf = pytest.importorskip("statsmodels.formula.api")
    glm = smf.glm("adverse_event ~ biomarker + age + arm", clinical_trial(),
                  family=sm.families.Binomial()).fit()
    assert _builds(model_report(glm, seed=1))
    assert _builds(residual_diagnostics_plot(glm, seed=1))
