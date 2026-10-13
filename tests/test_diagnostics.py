"""Smoke tests: every regression-diagnostic plot builds without error."""

import numpy as np
import pandas as pd
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
