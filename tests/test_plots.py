"""Smoke tests: every plotting function builds a figure without error."""

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot
from plotnine.exceptions import PlotnineWarning

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


def test_survival_plot_reads_the_surv_coding_as_the_r_twin_does():
    # Regression: the event went to lifelines as given, and lifelines counts
    # every non-zero value as an event. Under the survival::Surv() 1/2 coding
    # (1 = censored), which the R twin accepts, every subject died. Both curves
    # fell to zero, and the log-rank p was 0.0022 where the data give 7.2e-08.
    pytest.importorskip("lifelines")
    e = CT["event"].to_numpy()
    p01 = dp.survival_plot(CT["time"], e, group=CT["arm"])
    p12 = dp.survival_plot(CT["time"], e + 1, group=CT["arm"])
    pd.testing.assert_frame_equal(p12.data, p01.data)
    pd.testing.assert_frame_equal(p12.at_risk, p01.at_risk)
    assert p12.logrank_p == p01.logrank_p
    assert p01.logrank_p < 1e-7
    # A boolean indicator reads as 0/1.
    pb = dp.survival_plot(CT["time"], e == 1, group=CT["arm"])
    pd.testing.assert_frame_equal(pb.data, p01.data)

    # The coding is read over all observations. An arm with censorings only is
    # all 1s under the 1/2 coding, which must not be read as all events.
    t = np.arange(1.0, 9.0)
    g = np.repeat(["a", "b"], 4)
    e12 = np.array([2, 1, 2, 2, 1, 1, 1, 1])
    q12 = dp.survival_plot(t, e12, group=g)
    q01 = dp.survival_plot(t, e12 - 1, group=g)
    pd.testing.assert_frame_equal(q12.data, q01.data)
    assert (q12.data.loc[q12.data["group"] == "b", "surv"] == 1).all()


def test_survival_plot_refuses_an_event_coding_it_cannot_read():
    # Regression: 0/5 and 3/5 codings were accepted without a word, and 0/0.5
    # was cast to int and so read as all censored (log-rank p = 1). The R twin
    # refuses all three with this message, naming its own argument.
    pytest.importorskip("lifelines")
    e = CT["event"].to_numpy()
    message = re.escape(
        "`event` must use the 0/1 (0 = censored) or 1/2 (1 = censored) coding."
    )
    for codes in (e * 5, e * 0.5, e * 2 + 3):
        with pytest.raises(ValueError, match=message):
            dp.survival_plot(CT["time"], codes, group=CT["arm"])
    # Words say nothing about which value is the event.
    with pytest.raises(ValueError, match="recode a categorical or string"):
        dp.survival_plot(CT["time"], np.where(e == 1, "died", "censored"))


def test_survival_plot_drops_a_missing_event_with_a_warning():
    # Regression: casting a float event vector to int turned every NaN into the
    # integer sentinel numpy fills it with, which lifelines read as an observed
    # event, so an unknown outcome counted as a death. The warning is the R
    # twin's, word for word.
    pytest.importorskip("lifelines")
    t = CT["time"].to_numpy(dtype=float)
    e = CT["event"].to_numpy(dtype=float).copy()
    e[[3, 50, 200]] = np.nan

    with pytest.warns(UserWarning) as rec:
        p = dp.survival_plot(t, e, group=CT["arm"])
    assert str(rec[0].message) == (
        "3 observation(s) with a missing status were dropped."
    )
    _builds(p)

    # The figure is the one the complete rows alone give.
    keep = ~np.isnan(e)
    q = dp.survival_plot(t[keep], e[keep], group=CT["arm"][keep])
    pd.testing.assert_frame_equal(p.data, q.data)
    pd.testing.assert_frame_equal(p.at_risk, q.at_risk)
    assert p.logrank_p == q.logrank_p

    # A missing value in a nullable integer column drops the same way.
    with pytest.warns(UserWarning, match=re.escape("1 observation(s)")):
        dp.survival_plot(t[:5], pd.array([1, 0, None, 1, 0], dtype="Int64"))

    # Nothing left to plot is an error, not an empty figure.
    with pytest.warns(UserWarning), pytest.raises(
        ValueError, match=re.escape("`event` is missing for every observation.")
    ):
        dp.survival_plot(t, np.full(len(t), np.nan))


def test_survival_plot_drops_an_arm_whose_every_event_is_missing():
    # An arm left with no observations has no curve to draw. The R twin read a
    # factor's levels before dropping the missing statuses and kept such an arm
    # in its legend and risk table. Both twins now drop the arm, keep the
    # declared order for the others and run the log-rank test on what is left.
    pytest.importorskip("lifelines")
    g = pd.Series(pd.Categorical(np.repeat(["a", "b", "c"], 4),
                                 categories=["c", "b", "a"]))
    t = np.tile(np.arange(1.0, 5.0), 3) + 0.5
    e = np.array([1, 0, 1, 1, np.nan, np.nan, np.nan, np.nan, 1, 1, 0, 1])
    with pytest.warns(UserWarning, match=re.escape(
            "4 observation(s) with a missing status were dropped.")):
        p = dp.survival_plot(t, e, group=g, risk_table=True)
    assert list(p.data["group"].cat.categories) == ["c", "a"]
    assert set(p.at_risk["group"]) == {"a", "c"}
    keep = ~np.isnan(e)
    q = dp.survival_plot(t[keep], e[keep], group=g[keep].astype(str).to_numpy())
    assert p.logrank_p == q.logrank_p
    _builds(p)


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


@pytest.mark.parametrize(
    ("tmax", "times", "labels"),
    [
        (0.0158, [0, 0.005, 0.01, 0.015], ["0", "0.005", "0.01", "0.015"]),
        (1.2, [0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"]),
    ],
)
def test_survival_risk_table_keeps_its_times_exact(tmax, times, labels):
    # Regression: the at-risk times were rounded to one decimal. Follow-up up to
    # 0.0158 put every break at 0.0, so the table and the axis collapsed onto a
    # single tick. Follow-up up to 1.2 drew the counts at 0.25 and 0.75 under
    # ticks labelled 0.2 and 0.8.
    pytest.importorskip("lifelines")
    t = CT["time"].to_numpy(dtype=float)
    scaled = t / t.max() * tmax
    p = dp.survival_plot(scaled, CT["event"], group=CT["arm"], risk_table=True)
    for arm, rows in p.at_risk.groupby("group"):
        assert np.allclose(rows["time"], times)
        at_risk = [int(np.sum(scaled[CT["arm"] == arm] >= b)) for b in times]
        assert rows["n_at_risk"].tolist() == at_risk
    # The figure without the table reports the same times.
    plain = dp.survival_plot(scaled, CT["event"], group=CT["arm"])
    pd.testing.assert_frame_equal(plain.at_risk, p.at_risk)

    ax = p.draw(show=False).axes[0]
    assert np.allclose(ax.get_xticks(), times)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == labels


def test_survival_risk_table_stops_at_the_last_follow_up():
    # Regression: the break sequence ran past the longest follow-up, so the
    # at-risk table gained a column at a time nobody reached. Its counts fell
    # outside the x scale, and plotnine reported them as missing values.
    pytest.importorskip("lifelines")
    tmax = float(CT["time"].max())
    p = dp.survival_plot(CT["time"], CT["event"], group=CT["arm"],
                         risk_table=True)
    assert p.at_risk["time"].max() <= tmax
    with warnings.catch_warnings():
        warnings.simplefilter("error", PlotnineWarning)
        p.draw(show=False)
