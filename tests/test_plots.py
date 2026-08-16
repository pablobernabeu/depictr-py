"""Smoke tests: every plotting function builds a figure without error."""

import re

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

import depictr as dp

CT = dp.clinical_trial()
WB = dp.wellbeing_survey()
LD = dp.lexical_decision()


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


def test_explore_distribution_kinds():
    for kind in ("density", "histogram", "both"):
        assert _builds(dp.explore_distribution(LD, "RT", group="condition", kind=kind))


def test_explore_distribution_bad_column():
    with pytest.raises(KeyError):
        dp.explore_distribution(LD, "not_a_column")


def test_na_group_level_draws():
    # A missing value in a mapped column takes the NA colour, which must be
    # one matplotlib accepts (a hex value, not an R colour name).
    ld = LD.copy()
    ld.loc[ld.index[:20], "condition"] = np.nan
    assert _builds(dp.explore_distribution(ld, "RT", group="condition"))


def test_explore_categorical():
    assert _builds(dp.explore_categorical(WB, "education", group="region"))
    assert _builds(dp.explore_categorical(WB, "region"))


def test_correlation_heatmap_and_missingness():
    assert _builds(dp.correlation_heatmap(WB))
    assert _builds(dp.missingness_map(WB))


def test_correlation_heatmap_handles_an_undefined_correlation():
    # A constant column has no correlation, so pandas returned a band of NaN
    # that the label formatter printed as the literal string "nan". The
    # messages are the R twin's, word for word.
    df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 5, 9],
                       "flat": [3.0, 3, 3, 3], "also_flat": [7.0, 7, 7, 7]})
    with pytest.warns(UserWarning) as rec:
        p = dp.correlation_heatmap(df)
    assert str(rec[0].message) == (
        "correlation_heatmap(): dropping zero-variance column(s): "
        "flat, also_flat."
    )
    assert set(p.data["var1"].cat.categories) == {"a", "b"}
    assert "nan" not in set(p.data["label"])
    assert _builds(p)

    # Too little pairwise overlap leaves an NaN the column drop cannot catch,
    # so the label itself has to say so.
    sparse = pd.DataFrame({"x": [1.0, 2, np.nan, np.nan],
                           "y": [np.nan, np.nan, 3.0, 7.0]})
    q = dp.correlation_heatmap(sparse)
    assert "n/a" in set(q.data["label"])


def test_legend_inside_variants_build():
    # The opt-in inside-legend path builds for every function that offers it.
    assert _builds(dp.explore_distribution(LD, "RT", group="condition",
                                           legend_inside=True))
    # "both" exercises the line-keyed density geom and the guides() override.
    assert _builds(dp.explore_distribution(LD, "RT", group="condition",
                                           kind="both", legend_inside=True))
    assert _builds(dp.ecdf_plot(LD, "RT", group="condition", legend_inside=True))
    assert _builds(dp.missingness_map(WB, legend_inside=True))
    wb = WB.assign(grp=(WB["age"] < WB["age"].median()).map(
        {True: "younger", False: "older"}))
    assert _builds(dp.dumbbell_plot(wb, "region", "life_satisfaction", "grp",
                                    legend_inside=True))


def test_legend_inside_helper_validates_corner():
    from plotnine import theme

    from depictr.theme import legend_inside
    assert isinstance(legend_inside("bottom left"), theme)
    with pytest.raises(ValueError):
        legend_inside("middle")


def test_classification_family():
    pytest.importorskip("sklearn")
    y, s = CT["adverse_event"], CT["biomarker"]
    assert _builds(dp.roc_curve_plot(y, s))
    assert _builds(dp.pr_curve_plot(y, s))
    assert _builds(dp.gain_plot(y, s))
    assert _builds(dp.lift_plot(y, s))
    assert _builds(dp.threshold_plot(y, 1 / (1 + np.exp(-s))))
    assert _builds(dp.confusion_matrix_plot(y, (s > 0).astype(int)))
    assert _builds(dp.calibration_plot(y, 1 / (1 + np.exp(-s))))


def test_gain_and_lift_refuse_a_single_class_outcome():
    # Dividing by max(sum, 1) drew a curve flat along the axis, which reads as a
    # catastrophic model rather than as an undefined quantity. The message is the
    # R twin's, word for word.
    rng = np.random.default_rng(0)
    for y in (np.zeros(20, dtype=int), np.ones(20, dtype=int)):
        for fn in (dp.gain_plot, dp.lift_plot):
            with pytest.raises(
                ValueError,
                match="Gains/lift need both positive and negative outcomes.",
            ):
                fn(y, rng.random(20))


def test_curve_plots_refuse_a_single_class_outcome():
    # scikit-learn does not refuse a single class: it computes AUC = nan and
    # AP = 0.000 under a warning, which the annotations would print as if they
    # measured something. The messages are the R twin's, word for word.
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    cases = [
        (dp.roc_curve_plot,
         "ROC needs both positive and negative outcomes."),
        (dp.pr_curve_plot,
         "Precision-recall needs both positive and negative outcomes."),
        (dp.threshold_plot,
         "A threshold sweep needs both positive and negative outcomes."),
    ]
    for y in (np.zeros(20, dtype=int), np.ones(20, dtype=int)):
        for fn, message in cases:
            with pytest.raises(ValueError, match=message):
                fn(y, rng.random(20))


def test_survival_plot_rejects_non_finite_times():
    # A NaN follow-up time used to surface as a bare StopIteration from the
    # axis-break search, naming neither the argument nor the problem.
    pytest.importorskip("lifelines")
    t = CT["time"].to_numpy(dtype=float).copy()
    for bad in (np.nan, np.inf):
        t[3] = bad
        with pytest.raises(ValueError, match="`time` must be finite"):
            dp.survival_plot(t, CT["event"])


def test_survival_plot_drops_a_missing_group_rather_than_drawing_an_arm():
    # Regression: the missing value survived pd.unique(), so the loop ran with
    # it as a level, `groups == lvl` matched nothing, and a phantom arm labelled
    # "None" joined the curves. The messages are the R twin's, word for word.
    pytest.importorskip("lifelines")
    t = CT["time"].to_numpy(dtype=float)
    e = CT["event"].to_numpy(dtype=int)
    arms = set(CT["arm"].unique())
    # copy(): to_numpy() can hand back a view onto CT, and CT is shared.
    g = CT["arm"].to_numpy(dtype=object).copy()
    g[[5, 20, 40]] = None

    with pytest.warns(UserWarning) as rec:
        p = dp.survival_plot(t, e, group=pd.Series(g))
    assert str(rec[0].message) == (
        "3 observation(s) with a missing group were dropped."
    )
    assert set(p.data["group"].cat.categories) == arms
    assert set(p.at_risk["group"]) == arms
    _builds(p)

    # A categorical group keeps its declared order and loses only those rows.
    order = sorted(arms, reverse=True)
    gc = pd.Series(pd.Categorical(g, categories=order, ordered=True))
    with pytest.warns(UserWarning):
        q = dp.survival_plot(t, e, group=gc)
    assert list(q.data["group"].cat.categories) == order

    # Nothing left to plot is an error, not an empty figure.
    with pytest.warns(UserWarning), pytest.raises(
        ValueError, match=re.escape("`group` is missing for every observation.")
    ):
        dp.survival_plot(t, e, group=pd.Series([None] * len(t)))


def test_survival_plot_carries_logrank_and_at_risk():
    pytest.importorskip("lifelines")
    p = dp.survival_plot(CT["time"], CT["event"], group=CT["arm"])
    _builds(p)
    assert p.logrank_p is not None and 0 <= p.logrank_p <= 1
    assert len(p.at_risk) > 0


def test_survival_risk_table_composes():
    pytest.importorskip("lifelines")
    p = dp.survival_plot(CT["time"], CT["event"], group=CT["arm"],
                         risk_table=True)
    p.draw(show=False)  # composition (curve + at-risk table) draws
    assert len(p.at_risk) > 0
    # The risk-table path also honours an inside (top-right) legend.
    q = dp.survival_plot(CT["time"], CT["event"], group=CT["arm"],
                         risk_table=True, legend_inside=True)
    q.draw(show=False)
