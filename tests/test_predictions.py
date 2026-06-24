"""Smoke and behaviour tests for the model-prediction plots."""

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
