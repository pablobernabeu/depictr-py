"""Smoke and behaviour tests for the model-prediction plots."""

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from depictr.data import crop_yield, lexical_decision
from depictr.models import tidy_estimates
from depictr.predictions import compare_models, effects_plot, interaction_plot


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


@pytest.fixture(scope="module")
def ols():
    smf = pytest.importorskip("statsmodels.formula.api")
    # "yield" is a Python keyword, so the formula needs Q("yield").
    return smf.ols('Q("yield") ~ fertiliser * treatment + rainfall + soil_ph',
                   crop_yield()).fit()


@pytest.fixture(scope="module")
def ols_additive():
    smf = pytest.importorskip("statsmodels.formula.api")
    return smf.ols('Q("yield") ~ fertiliser + treatment + rainfall',
                   crop_yield()).fit()


@pytest.fixture(scope="module")
def glm():
    smf = pytest.importorskip("statsmodels.formula.api")
    sm = pytest.importorskip("statsmodels.api")
    return smf.glm("accuracy ~ word_frequency * condition + modality",
                   lexical_decision(), family=sm.families.Binomial()).fit()


def test_effects_plot_ols_and_glm(ols, glm):
    assert _builds(effects_plot(ols, "fertiliser", title="Effect"))
    # A GLM predicts on the response (probability) scale.
    assert _builds(effects_plot(glm, "word_frequency"))


def test_effects_plot_rejects_categorical_var(ols):
    with pytest.raises(TypeError):
        effects_plot(ols, "treatment")


def test_effects_plot_rejects_unknown_var(ols):
    with pytest.raises(KeyError):
        effects_plot(ols, "not_a_column")


def test_effects_plot_rejects_non_model():
    with pytest.raises(TypeError):
        effects_plot([1.0, 2.0, 3.0], "x")


def test_interaction_plot_ols_and_glm(ols, glm):
    assert _builds(interaction_plot(ols, "fertiliser", "treatment", title="Inter"))
    assert _builds(interaction_plot(glm, "word_frequency", "condition", band=False))


def test_interaction_plot_needs_numeric_x(ols):
    with pytest.raises(TypeError):
        interaction_plot(ols, "treatment", "fertiliser")


def test_interaction_plot_rejects_unknown_columns(ols):
    with pytest.raises(KeyError):
        interaction_plot(ols, "fertiliser", "nope")


def test_interaction_plot_ignores_a_missing_group_value():
    # A missing grouping value used to survive as the string "nan" and be drawn
    # as a third coloured line with its own legend entry.
    import numpy as np

    smf = pytest.importorskip("statsmodels.formula.api")
    cy = crop_yield().copy()
    cy.loc[:4, "treatment"] = np.nan
    model = smf.ols('Q("yield") ~ fertiliser * treatment', cy).fit()
    p = interaction_plot(model, "fertiliser", "treatment")
    assert sorted(p.data["treatment"].unique()) == ["enhanced", "standard"]


def test_compare_models_builds(ols, ols_additive):
    p = compare_models({"additive": ols_additive, "interaction": ols},
                       title="Comparison")
    assert _builds(p)


def test_compare_models_accepts_tidy_frame(ols, glm):
    # A fitted model and a pre-tidied frame should mix freely.
    frame = tidy_estimates(ols)
    assert _builds(compare_models({"glm": glm, "frame": frame}))


def test_compare_models_dodges_models_apart(ols_additive, ols):
    # Two models sharing a term must not land on the same y position.
    p = compare_models({"a": ols_additive, "b": ols})
    fig = p.draw(show=False)
    import numpy as np
    ys = []
    for coll in fig.axes[0].collections:
        off = np.asarray(coll.get_offsets())
        if len(off):
            ys.extend(off[:, 1].tolist())
    # More distinct y positions than terms means the models are separated.
    assert len(set(round(y, 3) for y in ys)) > 1


def test_compare_models_single_model(ols_additive):
    assert _builds(compare_models({"only": ols_additive}))


def test_compare_models_rejects_empty():
    with pytest.raises(ValueError):
        compare_models({})


# Estimation rows, formula variables and offsets. The grid was built over every
# row and every column of the source frame. Rows the fit dropped moved the held
# means and the plotted range, a column outside the formula counted as a
# predictor (an all-missing one raised IndexError), and offset and exposure fits
# were predicted at an offset of zero.


def _without_wettest_yields(n=60):
    """crop_yield with the response missing for the n wettest fields."""
    cy = crop_yield().copy()
    cy.loc[cy["rainfall"].nlargest(n).index, "yield"] = np.nan
    return cy


def test_effects_plot_holds_means_over_the_estimation_rows():
    smf = pytest.importorskip("statsmodels.formula.api")
    cy = _without_wettest_yields()
    model = smf.ols('Q("yield") ~ rainfall + fertiliser', cy).fit()
    est = cy[cy["yield"].notna()]
    p = effects_plot(model, "fertiliser")
    b = model.params
    expected = (b["Intercept"] + b["rainfall"] * est["rainfall"].mean()
                + b["fertiliser"] * p.data["fertiliser"])
    np.testing.assert_allclose(p.data["fit"], expected, rtol=1e-10)
    assert p.data["fertiliser"].min() == est["fertiliser"].min()
    assert p.data["fertiliser"].max() == est["fertiliser"].max()
    assert p.labels.y == "Predicted yield"


def test_interaction_plot_uses_the_estimation_rows():
    smf = pytest.importorskip("statsmodels.formula.api")
    cy = _without_wettest_yields()
    # A level seen only in rows the fit dropped has no estimate to draw.
    # treatment is categorical in R's level order, so "standard" is the
    # reference level and the new level has to be added as a category first.
    dropped = cy.index[cy["yield"].isna()]
    cy["treatment"] = cy["treatment"].cat.add_categories("organic")
    cy.loc[dropped[:5], "treatment"] = "organic"
    model = smf.ols('Q("yield") ~ fertiliser * treatment + rainfall', cy).fit()
    p = interaction_plot(model, "fertiliser", "treatment")
    assert sorted(p.data["treatment"].unique()) == ["enhanced", "standard"]
    est = cy[cy["yield"].notna()]
    ref = p.data[p.data["treatment"] == "standard"]
    b = model.params
    expected = (b["Intercept"] + b["fertiliser"] * ref["fertiliser"]
                + b["rainfall"] * est["rainfall"].mean())
    np.testing.assert_allclose(ref["fit"], expected, rtol=1e-10)
    assert ref["fertiliser"].max() == est["fertiliser"].max()


def test_a_repeated_index_label_still_picks_the_estimation_rows():
    smf = pytest.importorskip("statsmodels.formula.api")
    cy = _without_wettest_yields()
    # pd.concat() without ignore_index repeats labels, which row_labels cannot
    # tell apart. The positions of the dropped rows can.
    cy.index = list(range(100)) * 2
    model = smf.ols('Q("yield") ~ rainfall + fertiliser', cy).fit()
    est = cy[cy["yield"].notna()]
    p = effects_plot(model, "fertiliser")
    b = model.params
    expected = (b["Intercept"] + b["rainfall"] * est["rainfall"].mean()
                + b["fertiliser"] * p.data["fertiliser"])
    np.testing.assert_allclose(p.data["fit"], expected, rtol=1e-10)
    assert p.data["fertiliser"].max() == est["fertiliser"].max()


def test_a_term_computed_from_statistics_of_its_rows_is_refused():
    smf = pytest.importorskip("statsmodels.formula.api")
    cy = crop_yield()
    # get_prediction() rebuilt I(rainfall - rainfall.mean()) from the grid's
    # rows, so the centring moved with the grid and the line shifted.
    centred = smf.ols('Q("yield") ~ I(rainfall - rainfall.mean()) + fertiliser',
                      cy).fit()
    with pytest.raises(ValueError,
                       match=r"`I\(rainfall - rainfall\.mean\(\)\)` depends on statistics"):
        effects_plot(centred, "rainfall")
    with pytest.raises(ValueError, match="depends on statistics"):
        effects_plot(centred, "fertiliser")
    scaled = smf.ols('Q("yield") ~ fertiliser * treatment + I(rainfall / rainfall.max())',
                     cy).fit()
    with pytest.raises(ValueError, match="depends on statistics"):
        interaction_plot(scaled, "fertiliser", "treatment")
    # patsy's stateful transforms keep the values of the fit, so they stay.
    stateful = smf.ols('Q("yield") ~ center(rainfall) + standardize(fertiliser)',
                       cy).fit()
    p = effects_plot(stateful, "rainfall")
    b = stateful.params.to_numpy()
    expected = b[0] + b[1] * (p.data["rainfall"] - cy["rainfall"].mean())
    np.testing.assert_allclose(p.data["fit"], expected, rtol=1e-10)


def test_a_column_outside_the_formula_is_no_predictor():
    smf = pytest.importorskip("statsmodels.formula.api")
    cy = crop_yield().copy()
    cy["notes"] = pd.Series([None] * len(cy), dtype="object")
    model = smf.ols('Q("yield") ~ rainfall + fertiliser * treatment', cy).fit()
    # The all-missing notes column used to raise IndexError.
    assert _builds(effects_plot(model, "fertiliser"))
    assert _builds(interaction_plot(model, "fertiliser", "treatment"))
    # soil_ph is in the frame but not in the model, so sweeping it drew a flat line.
    with pytest.raises(KeyError):
        effects_plot(model, "soil_ph")
    with pytest.raises(KeyError):
        interaction_plot(model, "soil_ph", "treatment")


def test_quoted_and_transformed_predictors_are_found_in_the_formula():
    smf = pytest.importorskip("statsmodels.formula.api")
    cy = crop_yield()
    # The app quotes every column with Q("..."), which must still name it.
    quoted = smf.ols('Q("yield") ~ Q("rainfall") + Q("fertiliser")', cy).fit()
    p = effects_plot(quoted, "fertiliser")
    b = quoted.params
    expected = (b["Intercept"] + b['Q("rainfall")'] * cy["rainfall"].mean()
                + b['Q("fertiliser")'] * p.data["fertiliser"])
    np.testing.assert_allclose(p.data["fit"], expected, rtol=1e-10)

    # Transformed terms are held on the raw scale, and patsy re-applies them.
    logged = smf.ols('Q("yield") ~ np.log(rainfall) + fertiliser + C(treatment)',
                     cy).fit()
    p = effects_plot(logged, "fertiliser")
    grid = pd.DataFrame({"fertiliser": p.data["fertiliser"],
                         "rainfall": cy["rainfall"].mean(),
                         "treatment": cy["treatment"].mode().iloc[0]})
    np.testing.assert_allclose(p.data["fit"], logged.predict(grid), rtol=1e-10)
    assert _builds(effects_plot(logged, "rainfall"))


@pytest.fixture(scope="module")
def counts():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"x": rng.normal(size=200), "expo": rng.uniform(1, 10, 200),
                      "arm": np.tile(["a", "b"], 100)})
    d["y"] = rng.poisson(d["expo"] * np.exp(0.2 + 0.5 * d["x"]))
    return d


def test_offset_and_exposure_fits_are_predicted_at_the_mean_offset(counts):
    smf = pytest.importorskip("statsmodels.formula.api")
    sm = pytest.importorskip("statsmodels.api")
    d = counts
    mean_offset = np.log(d["expo"]).mean()
    poisson = sm.families.Poisson()
    fits = {
        "offset": smf.glm("y ~ x", d, family=poisson,
                          offset=np.log(d["expo"])).fit(),
        # An exposure enters as its log, so the mean offset is the log of the
        # geometric mean exposure.
        "exposure": smf.glm("y ~ x", d, family=poisson, exposure=d["expo"]).fit(),
    }
    for name, model in fits.items():
        p = effects_plot(model, "x")
        b = model.params
        # At x = 0 the curve is exp(b0 + mean offset), where it used to be exp(b0).
        expected = np.exp(b["Intercept"] + b["x"] * p.data["x"] + mean_offset)
        np.testing.assert_allclose(p.data["fit"], expected, rtol=1e-8, err_msg=name)
        assert p.labels.y == "Predicted y at the mean offset"

    model = smf.glm("y ~ x * arm", d, family=poisson,
                    offset=np.log(d["expo"])).fit()
    p = interaction_plot(model, "x", "arm")
    b = model.params
    a = p.data[p.data["arm"] == "a"]
    np.testing.assert_allclose(
        a["fit"], np.exp(b["Intercept"] + b["x"] * a["x"] + mean_offset), rtol=1e-8)
    assert p.labels.y == "Predicted y at the mean offset"
