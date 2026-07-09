"""Smoke tests: every plotting function builds a figure without error."""

import numpy as np
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
