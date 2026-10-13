"""Smoke tests: every regression-diagnostic plot builds without error."""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from depictr.data import clinical_trial, crop_yield, wellbeing_survey
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


# ---- variance inflation factors --------------------------------------------


@pytest.fixture(params=["patsy", "formulaic"])
def formula_engine(request):
    """Fit under each formula engine that the installed statsmodels offers.

    statsmodels 0.15 added formulaic beside patsy, chosen through
    ``statsmodels.formula.options``. Before 0.15, patsy was the only engine.
    """
    formula = pytest.importorskip("statsmodels.formula")
    options = getattr(formula, "options", None)
    if options is None:
        if request.param != "patsy":
            pytest.skip("statsmodels before 0.15 has no formulaic engine")
        yield request.param
        return
    pytest.importorskip(request.param)
    previous = options.formula_engine
    options.formula_engine = request.param
    try:
        yield request.param
    finally:
        options.formula_engine = previous


def _gvif_terms(model):
    # The port of the R twin's internal gvif_terms(): one row per term with its
    # df, GVIF and GVIF^(1/(2 df)).
    from depictr.diagnostics import _gvif_terms as gvif_terms

    return gvif_terms(model)


def _gvif(model):
    res = _gvif_terms(model)
    return dict(zip(res["term"], res["gvif"], strict=True))


def _hline(p):
    # The reference line as built: its positions, colour and line type. The
    # copy keeps the build from altering the plot under test.
    built = copy.deepcopy(p)
    built._build()
    layer = next(lay for lay in built.layers
                 if type(lay.geom).__name__ == "geom_hline")
    data = layer.data
    return (list(data["yintercept"]), *data["color"].unique(),
            *data["linetype"].unique())


def test_vif_gives_the_r_twins_generalised_vif_per_term(formula_engine):
    # Printed by dev/goldens/vif.R from the R twin's gvif_terms() at the commit
    # of change set CS30. Each factor is one term with one GVIF, whatever the
    # engine names its columns. patsy orders the terms otherwise than R, so
    # they are compared by name.
    smf = pytest.importorskip("statsmodels.formula.api")
    fit = smf.ols("life_satisfaction ~ stress + region + education",
                  wellbeing_survey()).fit()
    res = _gvif_terms(fit).set_index("term")
    r_twin = {"stress": (1, 1.21638178777, 1.10289699781),
              "region": (3, 1.22488550981, 1.03438587836),
              "education": (2, 1.01798089228, 1.00446522657)}
    assert sorted(res.index) == sorted(r_twin)
    for term, (df, gvif, adjusted) in r_twin.items():
        assert res.loc[term, "df"] == df
        assert res.loc[term, "gvif"] == pytest.approx(gvif, rel=1e-9)
        assert res.loc[term, "gvif_adj"] == pytest.approx(adjusted, rel=1e-9)
    # The figure draws GVIF^(1/(2 df)), since two terms span several columns,
    # against the square root of the threshold, as the R twin does.
    p = vif_plot(fit)
    drawn = dict(zip(p.data["term"].astype(str), p.data["vif"], strict=True))
    assert drawn == pytest.approx({t: v[2] for t, v in r_twin.items()}, rel=1e-9)
    assert list(p.data["term"].cat.categories) == ["education", "region", "stress"]
    assert p.labels.y == r"$\mathrm{GVIF}^{1/(2\,\mathrm{df})}$"
    line, colour, linetype = _hline(p)
    assert line == pytest.approx([np.sqrt(5)])
    assert (colour, linetype) == ("#666666", "solid")
    # Drawn last: drawing a flipped plot swaps the labels it holds.
    assert _builds(p)


def test_vif_of_a_factor_does_not_depend_on_its_reference_level():
    # A three-level factor independent of x, with a reference level of 14 rows
    # in 400. Each dummy's own VIF was 7.7 with "a" as the reference, past the
    # line at 5, and 1.04 with "b". The GVIF is 1.005 either way.
    smf = pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(30)
    site = np.repeat(["a", "b", "c"], [14, 193, 193])
    x = rng.normal(size=400)
    y = x + rng.normal(size=400)
    drawn = {}
    for levels in (["a", "b", "c"], ["b", "a", "c"]):
        frame = pd.DataFrame({"y": y, "x": x,
                              "site": pd.Categorical(site, categories=levels)})
        p = vif_plot(smf.ols("y ~ x + site", frame).fit())
        drawn[levels[0]] = dict(zip(p.data["term"].astype(str), p.data["vif"],
                                    strict=True))
    assert set(drawn["a"]) == set(drawn["b"]) == {"x", "site"}
    assert drawn["a"] == pytest.approx(drawn["b"], rel=1e-10)
    assert max(drawn["a"].values()) < 1.1


def test_vif_weights_a_weighted_fit_and_a_glm_as_the_r_twin_does(glm):
    # The GVIF describes the estimated coefficients, whose covariance carries
    # the weights of the fit. The correlations of the unweighted design matrix
    # gave other values. The values are printed by dev/goldens/vif.R, and
    # car::vif() gives the same in R.
    smf = pytest.importorskip("statsmodels.formula.api")
    ws = wellbeing_survey()
    weights = np.resize([1.0, 2.0, 3.0], len(ws))
    wls = smf.wls("life_satisfaction ~ stress + region + education", ws,
                  weights=weights).fit()
    weighted = {"stress": 1.22771766362, "region": 1.22763102351,
                "education": 1.02413689177}
    assert _gvif(wls) == pytest.approx(weighted, rel=1e-9)
    # vif_plot() takes the WLS result, which lacks the influence machinery the
    # other diagnostics need and was refused as not fitted, and draws those
    # values. A raw array is still refused.
    p = vif_plot(wls)
    drawn = dict(zip(p.data["term"].astype(str), p.data["gvif"], strict=True))
    assert drawn == pytest.approx(weighted, rel=1e-9)
    with pytest.raises(TypeError, match="fitted statsmodels result"):
        vif_plot(np.ones((4, 2)))
    # A logistic GLM, and the same model fitted by Logit. The R fit is
    # converged tightly, to the estimate statsmodels reaches.
    expected = {"biomarker": 1.00264843600, "age": 1.00461360059,
                "arm": 1.00348848503}
    assert _gvif(glm) == pytest.approx(expected, rel=1e-7)
    logit = smf.logit("adverse_event ~ biomarker + age + arm",
                      clinical_trial()).fit(disp=0)
    assert _gvif(logit) == pytest.approx(expected, rel=1e-7)


def test_vif_plot_refuses_aliased_coefficients(formula_engine):
    # rain2 is twice rainfall, so the design is rank-deficient and no VIF
    # exists. statsmodels fitted it through a pseudo-inverse, and the plot drew
    # infinite VIFs for both columns, or about 1e15 under statsmodels 0.15,
    # with no more than a warning. The R twin refuses the same model with the
    # same sentence.
    smf = pytest.importorskip("statsmodels.formula.api")
    sm = pytest.importorskip("statsmodels.api")
    cy = crop_yield()
    cy["rain2"] = 2 * cy["rainfall"]
    cy["treatment2"] = cy["treatment"]
    cy["arm"] = cy["treatment"].astype("category").cat.add_categories("unused")
    ct = clinical_trial()
    ct["age_months"] = 12 * ct["age"]
    with warnings.catch_warnings():
        # statsmodels 0.15 warns that the design is rank-deficient.
        warnings.simplefilter("ignore")
        doubled = smf.ols('Q("yield") ~ rainfall + rain2 + fertiliser', cy).fit()
        copied = smf.ols('Q("yield") ~ rainfall + treatment + treatment2', cy).fit()
        logistic = smf.glm("adverse_event ~ biomarker + age + age_months", ct,
                           family=sm.families.Binomial()).fit()
        unused = smf.ols('Q("yield") ~ rainfall + fertiliser + arm', cy).fit()
    with pytest.raises(ValueError, match=(
            r"^VIF is undefined when coefficients are aliased: rain2\. "
            r"Drop the redundant term\(s\) and refit\.$")):
        vif_plot(doubled)
    # A copy of a factor is aliased as a whole term, and a GLM is refused alike.
    with pytest.raises(ValueError, match=r"aliased: treatment2\. "):
        vif_plot(copied)
    with pytest.raises(ValueError, match=r"aliased: age_months\. "):
        vif_plot(logistic)
    # A category that no row takes fills a column of zeros, as the docstring
    # warns. R's lm() drops such a level, but this fit kept its coefficient.
    with pytest.raises(ValueError, match=r"aliased: arm\. "):
        vif_plot(unused)


def test_vif_skips_a_term_that_fills_no_columns(formula_engine):
    # A categorical with a single level fills no column under either engine,
    # so it has no coefficient and no VIF. Computing one for it raised
    # ZeroDivisionError.
    smf = pytest.importorskip("statsmodels.formula.api")
    cy = crop_yield()
    cy["farm"] = pd.Categorical(["only"] * len(cy))
    with_farm = vif_plot(smf.ols('Q("yield") ~ rainfall + fertiliser + farm', cy).fit())
    without = vif_plot(smf.ols('Q("yield") ~ rainfall + fertiliser', cy).fit())
    drawn = dict(zip(with_farm.data["term"].astype(str), with_farm.data["vif"],
                     strict=True))
    expected = dict(zip(without.data["term"].astype(str), without.data["vif"],
                        strict=True))
    assert drawn == pytest.approx(expected, rel=1e-12)


def test_vif_plot_draws_the_threshold_on_the_scale_of_its_bars(ols):
    # Every term of this model has one column, so the bars are ordinary VIFs
    # and the line marks the threshold itself, in the R twin's grey.
    p = vif_plot(ols)
    assert p.labels.y == "Variance inflation factor"
    assert _hline(p) == ([5], "#666666", "solid")
    assert _hline(vif_plot(ols, threshold=10))[0] == [10]
    for bad in (0, -1, float("nan"), "VIF"):
        with pytest.raises(ValueError, match="`threshold` must be"):
            vif_plot(ols, threshold=bad)


def test_vif_plot_warns_that_a_model_without_an_intercept_is_uncentred():
    # Without an intercept, the coefficients' covariance comes from uncentred
    # columns. car::vif() warns here too, and so does the R twin.
    smf = pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(13)
    x1 = rng.normal(5, 1, size=100)
    frame = pd.DataFrame({"x1": x1, "x2": 0.8 * x1 + rng.normal(0, 0.5, size=100)})
    frame["y"] = frame["x1"] - frame["x2"] + rng.normal(size=100)
    fit = smf.ols("y ~ 0 + x1 + x2", frame).fit()
    with pytest.warns(UserWarning, match="no intercept"):
        res = _gvif_terms(fit)
    # Uncentred VIFs: each diagonal entry of X'X times that of its inverse.
    gram = fit.model.exog.T @ fit.model.exog
    np.testing.assert_allclose(res["gvif"],
                               np.diag(gram) * np.diag(np.linalg.inv(gram)),
                               rtol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _gvif_terms(smf.ols("y ~ x1 + x2", frame).fit())


def test_vif_takes_each_column_of_an_array_fit_as_a_term():
    # A fit from arrays has no formula to group its columns, so each column is
    # a term, named as statsmodels names it, and the constant is left out.
    sm = pytest.importorskip("statsmodels.api")
    # A formula drops incomplete rows itself, but arrays have to arrive complete.
    ws = wellbeing_survey().dropna(
        subset=["life_satisfaction", "stress", "age", "income"])
    exog = sm.add_constant(ws[["stress", "age", "income"]].to_numpy(dtype=float))
    fit = sm.OLS(ws["life_satisfaction"].to_numpy(), exog).fit()
    res = _gvif_terms(fit)
    assert list(res["term"]) == ["x1", "x2", "x3"]
    assert (res["df"] == 1).all()
    smf = pytest.importorskip("statsmodels.formula.api")
    by_formula = smf.ols("life_satisfaction ~ stress + age + income", ws).fit()
    np.testing.assert_allclose(res["gvif"], _gvif_terms(by_formula)["gvif"],
                               rtol=1e-10)


def test_vif_needs_two_predictor_columns():
    # A lone predictor has nothing to be collinear with, and the R twin refuses
    # it too.
    smf = pytest.importorskip("statsmodels.formula.api")
    fit = smf.ols('Q("yield") ~ rainfall', crop_yield()).fit()
    with pytest.raises(ValueError, match="at least two predictor columns"):
        vif_plot(fit)


def test_binned_residual_plot(glm):
    assert _builds(binned_residual_plot(glm))
    assert _builds(binned_residual_plot(glm, n_bins=8))


def test_binned_residual_plot_rejects_non_binomial(ols):
    with pytest.raises(ValueError):
        binned_residual_plot(ols)


def test_binned_residual_plot_accepts_logit_and_probit(glm):
    # Logit and Probit fits are binomial models fitted another way, and the
    # check on a GLM family turned them away. A Logit fit is the same model as
    # the binomial GLM, so its bins are the same.
    smf = pytest.importorskip("statsmodels.formula.api")
    ct = clinical_trial()
    logit = smf.logit("adverse_event ~ biomarker + age + arm", ct).fit(disp=0)
    probit = smf.probit("adverse_event ~ biomarker + age + arm", ct).fit(disp=0)
    assert _builds(binned_residual_plot(logit))
    assert _builds(binned_residual_plot(probit))
    by_logit = binned_residual_plot(logit).data.reset_index(drop=True)
    by_glm = binned_residual_plot(glm).data.reset_index(drop=True)
    pd.testing.assert_frame_equal(by_logit, by_glm, check_exact=False,
                                  atol=1e-6)


# ---- binned residuals: the R twin's bins and band ---------------------------
# Printed by dev/goldens/binned_residuals.R from the R twin's
# binned_residual_plot() at the commit of change set CS28. Each row is a bin's
# mean fitted value, mean residual, count and band half-width (se2) under the
# default band="model". The rows of the documented example add the half-width
# under band="empirical".

R_BINS_EXAMPLE = [
    (0.0142420795, -0.0142420795, 17, 0.0574492525, 0.0017657912),
    (0.0230750271, -0.0230750271, 18, 0.0707659642, 0.0013170924),
    (0.0312556123, -0.0312556123, 17, 0.0843983288, 0.0011851072),
    (0.0384348443, 0.0726762668, 18, 0.0906184780, 0.1524965163),
    (0.0459856867, -0.0459856867, 18, 0.0987323512, 0.0010466797),
    (0.0526670975, 0.0649799614, 17, 0.1083441832, 0.1609988499),
    (0.0621326966, 0.0489784145, 18, 0.1137915470, 0.1529117365),
    (0.0665399618, 0.0445711493, 18, 0.1174837589, 0.1522959059),
    (0.0741992223, -0.0153756929, 17, 0.1271293258, 0.1175670234),
    (0.0834799798, 0.0276311313, 18, 0.1303825327, 0.1524724896),
    (0.0953429861, -0.0397874306, 18, 0.1384349818, 0.1110805751),
    (0.1065722997, -0.0477487703, 17, 0.1496713500, 0.1177669393),
    (0.1245439398, 0.0421227269, 18, 0.1556236571, 0.1789673061),
    (0.1497020591, -0.1497020591, 18, 0.1680923077, 0.0058156997),
    (0.1876487908, -0.0111782026, 17, 0.1893435630, 0.1906454287),
    (0.2353108099, 0.0424669679, 18, 0.1997584984, 0.2170236029),
    (0.3570105354, 0.0318783535, 18, 0.2222172685, 0.2352705929),
]

# adverse_event ~ arm: one fitted value per arm, so the ranks within an arm,
# and with them the bins, follow row order.
R_BINS_TIED = [
    (0.0857142857, 0.0319327731, 17, 0.1357916630),
    (0.0857142857, 0.0253968254, 18, 0.1319657758),
    (0.0857142857, -0.0268907563, 17, 0.1357916630),
    (0.0857142857, 0.0253968254, 18, 0.1319657758),
    (0.0857142857, 0.0809523810, 18, 0.1319657758),
    (0.0857142857, -0.0857142857, 17, 0.1357916630),
    (0.0857142857, 0.0253968254, 18, 0.1319657758),
    (0.0875496032, -0.0875496032, 18, 0.1331894286),
    (0.1187500000, 0.1165441176, 17, 0.1569176911),
    (0.1187500000, -0.0631944444, 18, 0.1524965847),
    (0.1187500000, -0.0076388889, 18, 0.1524965847),
    (0.1187500000, 0.0577205882, 17, 0.1569176911),
    (0.1187500000, -0.1187500000, 18, 0.1524965847),
    (0.1187500000, -0.0076388889, 18, 0.1524965847),
    (0.1187500000, 0.0577205882, 17, 0.1569176911),
    (0.1187500000, 0.0479166667, 18, 0.1524965847),
    (0.1187500000, -0.0631944444, 18, 0.1524965847),
]

# cbind(successes, trials - successes) ~ x on the integer data of _grouped().
R_BINS_GROUPED = [
    (0.4229536084, -0.1109827089, 6, 0.1514291221),
    (0.4244059355, 0.0415031554, 7, 0.1453063692),
    (0.4259714223, 0.1159477696, 7, 0.1414302926),
    (0.4274264031, -0.1264163021, 6, 0.1332731027),
    (0.4288826727, 0.1933962388, 7, 0.1480060090),
    (0.4304523234, -0.0584517050, 7, 0.1455670330),
]

# The same model fitted as a quasibinomial one, whose dispersion, 2.958,
# widens every band by its square root.
R_BINS_GROUPED_QUASI = [
    (0.4229536084, -0.1109827089, 6, 0.2604387692),
    (0.4244059355, 0.0415031554, 7, 0.2499084155),
    (0.4259714223, 0.1159477696, 7, 0.2432420582),
    (0.4274264031, -0.1264163021, 6, 0.2292127323),
    (0.4288826727, 0.1933962388, 7, 0.2545514515),
    (0.4304523234, -0.0584517050, 7, 0.2503567240),
]


def _assert_bins(p, pinned):
    """Compare a binned-residual plot's per-bin data with rows pinned from R."""
    bins = p.data
    pinned = np.asarray(pinned, dtype=float)
    assert len(bins) == len(pinned)
    np.testing.assert_allclose(bins["fitted"], pinned[:, 0], atol=1e-6)
    np.testing.assert_allclose(bins["resid"], pinned[:, 1], atol=1e-6)
    assert bins["n"].tolist() == pinned[:, 2].astype(int).tolist()
    np.testing.assert_allclose(bins["se2"], pinned[:, 3], atol=1e-6)


def _grouped():
    """The grouped binomial data of dev/goldens/binned_residuals.R."""
    i = np.arange(1, 41)
    trials = 4 + i % 9
    return pd.DataFrame({"x": (i - 20.5) / 10, "trials": trials,
                         "successes": (trials * (i % 11)) // 10})


def test_binned_residual_plot_matches_the_r_twin(glm):
    # The twins took different numbers of bins (round against floor of
    # sqrt(n)) and cut the ranks at different places, so on this example the
    # first bin held 18 rows here and 17 in R.
    p = binned_residual_plot(glm)
    _assert_bins(p, [row[:4] for row in R_BINS_EXAMPLE])
    # A correctly specified model: the data were simulated from it.
    assert p.data["outside"].sum() <= 1

    empirical = binned_residual_plot(glm, band="empirical").data
    np.testing.assert_allclose(empirical["se2"],
                               [row[4] for row in R_BINS_EXAMPLE], atol=1e-6)
    # The empirical band collapses in the bins without events and flags them,
    # as the R twin's flags its bins 1, 2, 3, 5 and 14.
    assert np.flatnonzero(empirical["outside"]).tolist() == [0, 1, 2, 4, 13]
    np.testing.assert_allclose(empirical["upper"], empirical["se2"])
    np.testing.assert_allclose(empirical["lower"], -empirical["se2"])


def test_binned_residual_plot_breaks_ties_by_row_order():
    # numpy's default sort leaves the order of tied fitted values unspecified,
    # and R's rank(ties.method = "first") follows row order.
    smf = pytest.importorskip("statsmodels.formula.api")
    sm = pytest.importorskip("statsmodels.api")
    tied = smf.glm("adverse_event ~ arm", clinical_trial(),
                   family=sm.families.Binomial()).fit()
    _assert_bins(binned_residual_plot(tied), R_BINS_TIED)


def _grouped_fits(**fit_kwargs):
    """The grouped model of _grouped(), fitted from two columns and with var_weights."""
    sm = pytest.importorskip("statsmodels.api")
    d = _grouped()
    X = sm.add_constant(d["x"].to_numpy())
    by_columns = sm.GLM(np.column_stack([d["successes"],
                                         d["trials"] - d["successes"]]),
                        X, family=sm.families.Binomial()).fit(**fit_kwargs)
    by_weights = sm.GLM(d["successes"] / d["trials"], X,
                        family=sm.families.Binomial(),
                        var_weights=d["trials"].to_numpy()).fit(**fit_kwargs)
    return by_columns, by_weights


def test_binned_residual_plot_reads_grouped_binomials_as_proportions():
    # For a (successes, failures) response, statsmodels' resid_response counts
    # successes while its fitted values are probabilities, and the band took
    # each row as a single trial. The residuals are now proportions, as in the
    # R twin, and each row's variance is divided by its trials.
    for fit in _grouped_fits():
        _assert_bins(binned_residual_plot(fit), R_BINS_GROUPED)


def test_binned_residual_plot_takes_floor_sqrt_n_bins():
    # floor(sqrt(n)) bins, at least 2 and at most n // 2, as in the R twin.
    # round(sqrt(n)) took 6 bins for 31 rows and 3 for 7.
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(3)
    x = rng.normal(size=31)
    y = (rng.random(31) < 1 / (1 + np.exp(-x))).astype(float)
    fit = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
    assert len(binned_residual_plot(fit).data) == 5
    small = sm.GLM(np.array([0.0, 1, 0, 1, 1, 0, 1]),
                   sm.add_constant(np.arange(7.0)),
                   family=sm.families.Binomial()).fit()
    assert len(binned_residual_plot(small).data) == 2

    with pytest.raises(ValueError, match="n_bins"):
        binned_residual_plot(fit, n_bins=1)
    with pytest.raises(ValueError, match="band"):
        binned_residual_plot(fit, band="nope")
    # Fewer than four rows, as in R.
    three = sm.GLM(np.array([0.0, 1, 0]), np.ones((3, 1)),
                   family=sm.families.Binomial()).fit()
    with pytest.raises(ValueError, match="Too few"):
        binned_residual_plot(three)


def test_binned_residual_plot_leaves_out_rows_with_zero_weight():
    # A row with zero weight takes no part in the fit, and the model gives its
    # residual no finite variance, so it is left out of the bins, as in R.
    smf = pytest.importorskip("statsmodels.formula.api")
    sm = pytest.importorskip("statsmodels.api")
    ct = clinical_trial()
    w = np.ones(len(ct))
    w[[4, 59, 139]] = 0
    formula = "adverse_event ~ biomarker + age + arm"
    weighted = smf.glm(formula, ct, family=sm.families.Binomial(),
                       var_weights=w).fit()
    kept = smf.glm(formula, ct[w > 0], family=sm.families.Binomial()).fit()
    for band in ("model", "empirical"):
        pd.testing.assert_frame_equal(
            binned_residual_plot(weighted, band=band).data,
            binned_residual_plot(kept, band=band).data,
            check_exact=False, atol=1e-6,
        )


def test_binned_residual_band_scales_with_an_estimated_scale(glm):
    # A binomial GLM fitted with scale="X2" estimates its scale, as a
    # quasibinomial fit estimates its dispersion in the R twin, and its
    # residual variances are multiplied by it. The model is fitted afresh, so
    # that the fixture's model keeps its own scale setting.
    smf = pytest.importorskip("statsmodels.formula.api")
    sm = pytest.importorskip("statsmodels.api")
    quasi = smf.glm("adverse_event ~ biomarker + age + arm", clinical_trial(),
                    family=sm.families.Binomial()).fit(scale="X2")
    assert quasi.scale != 1
    np.testing.assert_allclose(binned_residual_plot(quasi).data["se2"],
                               binned_residual_plot(glm).data["se2"]
                               * np.sqrt(quasi.scale))


def test_binned_residual_band_counts_the_trials_in_an_estimated_scale():
    # statsmodels' fit(scale="X2") leaves out the trials of a two-column
    # response, so it put the scale of the grouped data at 0.39, below the
    # binomial 1, where R's quasibinomial dispersion is 2.96. The band was then
    # narrower than the binomial one. Fitted with var_weights, the same model
    # gets 2.96 from statsmodels. Both now match R.
    for fit in _grouped_fits(scale="X2"):
        _assert_bins(binned_residual_plot(fit), R_BINS_GROUPED_QUASI)
    # A scale estimated from the deviance, or fixed by the caller, is the
    # fit's own and is kept.
    binomial_se2 = np.asarray(R_BINS_GROUPED)[:, 3]
    for scale in ("dev", 2.0):
        for fit in _grouped_fits(scale=scale):
            np.testing.assert_allclose(binned_residual_plot(fit).data["se2"],
                                       binomial_se2 * np.sqrt(fit.scale),
                                       atol=1e-6)
