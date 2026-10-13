"""Smoke tests: every plotting function builds a figure without error."""

import copy
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot
from plotnine.exceptions import PlotnineWarning

import depictr as dp
from depictr.audit import _contrast_ratio

CT = dp.clinical_trial()
WB = dp.wellbeing_survey()
LD = dp.lexical_decision()


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


def _label_contrasts(p):
    """Each label's colour and contrast against the tile drawn beneath it.

    Read back from the drawn tile and text layers, with the contrast the other
    of black and white would have had on the same tile.
    """
    built = copy.deepcopy(p)
    figure = built.draw(show=False)
    try:
        tiles = built.layers[0].data[["x", "y", "fill"]]
        texts = built.layers[1].data[["x", "y", "color", "label"]]
    finally:
        plt.close(figure)
    cells = tiles.merge(texts, on=["x", "y"])
    assert len(cells) == len(texts)
    colours = [str(c).lower() for c in cells["color"]]
    fills = [str(c) for c in cells["fill"]]
    other = ["#ffffff" if c == "#000000" else "#000000" for c in colours]
    return pd.DataFrame({
        "label": [str(v) for v in cells["label"]],
        "colour": colours,
        "fill": fills,
        "ratio": _contrast_ratio(colours, fills),
        "other": _contrast_ratio(other, fills),
    })


def _assert_legible(cells):
    assert set(cells["colour"]) <= {"#000000", "#ffffff"}
    assert (cells["ratio"] >= 4.5).all()
    assert (cells["ratio"] >= cells["other"]).all()


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


def test_correlation_heatmap_labels_are_legible_on_their_tiles():
    # Every label was the near-black #1a1a1a, which left the r = 1 diagonal on
    # the darkest blue of the ramp at 2.43:1. An undefined correlation's n/a
    # sat on the scale's grey #7f7f7f at 4.35:1.
    sparse = pd.DataFrame({"a": [1.0, 2, 3, np.nan, np.nan, np.nan],
                           "b": [np.nan, np.nan, np.nan, 4.0, 5, 6],
                           "c": [1.0, 2, 3, 4, 5, 6]})
    for data in (WB, sparse):
        _assert_legible(_label_contrasts(dp.correlation_heatmap(data)))
    cells = _label_contrasts(dp.correlation_heatmap(sparse))
    assert (cells["label"] == "n/a").any()
    assert (cells.loc[cells["label"] == "1.00", "colour"] == "#ffffff").all()


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


def test_confusion_matrix_counts_are_legible_on_their_tiles():
    # Every count was drawn in the near-black #1a1a1a on a ramp that ends in a
    # dark blue, so the busiest cell was the least legible. In the docstring's
    # own example it sat at 1.32:1.
    pytest.importorskip("sklearn")
    score = 1 / (1 + np.exp(-CT["biomarker"]))
    y, pred = CT["adverse_event"], (score > 0.6).astype(int)
    counts = [50, 12, 3, 9, 40, 15, 2, 18, 60]
    actual = np.repeat(np.repeat(["a", "b", "c"], 3), counts)
    predicted = np.repeat(np.tile(["a", "b", "c"], 3), counts)
    for p in (dp.confusion_matrix_plot(y, pred),
              dp.confusion_matrix_plot(y, pred, normalise="true"),
              dp.confusion_matrix_plot(actual, predicted),
              dp.confusion_matrix_plot(actual, predicted, normalise="pred")):
        _assert_legible(_label_contrasts(p))
    # The busiest cell is the darkest and takes white, the emptiest black.
    cells = _label_contrasts(dp.confusion_matrix_plot(y, pred))
    count = cells["label"].astype(float)
    assert cells.loc[count.idxmax(), "colour"] == "#ffffff"
    assert cells.loc[count.idxmin(), "colour"] == "#000000"


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


# ---- gains, lift and calibration: parity with the R twin -------------------
# The R values below are printed by dev/goldens/classification.R, run against
# the R twin for change set CS29. Its gains and calibration code for vector
# input is that of the 0.3.0 release. Both score vectors are built from integers
# by the same few floating-point operations in the same order, so the script
# and these tests build identical values.


def _tied_scores():
    """200 cases in four risk bands, as a banded or rounded score gives.

    The bands hold 23, 41, 57 and 79 cases at scores 0.85, 0.55, 0.30 and
    0.10, with 17, 22, 18 and 7 of the 64 positives. Row i is row
    77 * i % 200 of the banded layout, which spreads each band through the
    rows with its positives first.
    """
    sizes, positives = [23, 41, 57, 79], [17, 22, 18, 7]
    score = np.repeat([0.85, 0.55, 0.30, 0.10], sizes)
    y = np.concatenate([np.repeat([1, 0], [k, n - k])
                        for n, k in zip(sizes, positives, strict=True)])
    stride = (77 * np.arange(200)) % 200
    return y[stride], score[stride]


def _continuous_scores():
    """300 distinct probabilities that crowd towards zero, with fixed outcomes.

    Case i (from 1) has the score 0.9 * (i / 300)^2. It is an event when
    (37 * i mod 101) / 101 falls below its score, a fixed sequence standing in
    for random draws.
    """
    i = np.arange(1, 301)
    x = i / 300
    score = 0.9 * x * x
    y = (((37 * i) % 101) / 101 < score).astype(int)
    return y, score


def test_gain_and_lift_collapse_tied_scores_as_the_r_twin_does():
    # Regression: both curves accumulated row by row, so a run of tied scores
    # was targeted in whatever order its rows came, one case at a time. The R
    # twin targets a run all at once, giving one point per distinct score.
    y, score = _tied_scores()
    gain = dp.gain_plot(y, score).data
    np.testing.assert_allclose(gain["population"], [0, 0.115, 0.32, 0.605, 1],
                               atol=1e-10)
    np.testing.assert_allclose(gain["captured"],
                               [0, 0.265625, 0.609375, 0.890625, 1], atol=1e-10)
    lift = dp.lift_plot(y, score).data
    np.testing.assert_allclose(lift["population"], [0.115, 0.32, 0.605, 1],
                               atol=1e-10)
    np.testing.assert_allclose(
        lift["lift"], [2.3097826087, 1.9042968750, 1.4721074380, 1], atol=1e-9)


def test_gain_and_lift_do_not_depend_on_row_order():
    # In the audit's example, 200 cases with four distinct scores, the share of
    # positives captured at 10% depth ranged from 0.204 to 0.296 with the order
    # of the rows alone.
    y, score = _tied_scores()
    rng = np.random.default_rng(29)
    orders = {
        "shuffled": rng.permutation(len(y)),
        "positives first within ties": np.lexsort((-y, -score)),
        "negatives first within ties": np.lexsort((y, -score)),
    }
    gain, lift = dp.gain_plot(y, score).data, dp.lift_plot(y, score).data
    for label, o in orders.items():
        pd.testing.assert_frame_equal(dp.gain_plot(y[o], score[o]).data, gain,
                                      obj=label)
        pd.testing.assert_frame_equal(dp.lift_plot(y[o], score[o]).data, lift,
                                      obj=label)

    # The bundled biomarker is recorded to two decimals, so 162 of the 300
    # patients share their value with another.
    ct = CT.sample(frac=1, random_state=3)
    for fn in (dp.gain_plot, dp.lift_plot):
        before = fn(CT["adverse_event"], CT["biomarker"]).data
        after = fn(ct["adverse_event"], ct["biomarker"]).data
        pd.testing.assert_frame_equal(after, before)


def test_gain_and_lift_are_unchanged_when_no_scores_tie():
    # With every score distinct, each run of ties is a single case, so the
    # curves are the row-by-row curves depictr 0.3.0 drew.
    y, score = _continuous_scores()
    o = np.random.default_rng(4).permutation(len(y))
    y, score = y[o], score[o]
    population = np.arange(1, len(y) + 1) / len(y)
    captured = np.cumsum(y[np.argsort(-score)]) / y.sum()
    gain = dp.gain_plot(y, score).data
    np.testing.assert_allclose(gain["population"], np.r_[0, population])
    np.testing.assert_allclose(gain["captured"], np.r_[0, captured])
    lift = dp.lift_plot(y, score).data
    np.testing.assert_allclose(lift["population"], population)
    np.testing.assert_allclose(lift["lift"], captured / population)


_R_CALIBRATION = {
    "continuous scores, 10 bins, 95%": (_continuous_scores, {}, {
        "predicted": [0.0031516667, 0.0214516667, 0.0577516667, 0.1120516667,
                      0.1843516667, 0.2746516667, 0.3829516667, 0.5092516667,
                      0.6535516667, 0.8158516667],
        "observed": [0, 0, 0.0666666667, 0.1666666667, 0.2, 0.3, 0.4, 0.5,
                     0.6333333333, 0.8333333333],
        "successes": [0, 0, 2, 5, 6, 9, 12, 15, 19, 25],
        "n": [30] * 10,
        "conf_low": [0, 0, 0.0184770238, 0.0733654237, 0.0950510718,
                     0.1666474827, 0.2459062812, 0.3315412564, 0.4551356365,
                     0.6643564949],
        "conf_high": [0.1135133932, 0.1135133932, 0.2132345836, 0.3356435051,
                      0.3730569641, 0.4787578746, 0.5767963975, 0.6684587436,
                      0.7812607919, 0.9266345763],
    }),
    "continuous scores, 4 bins, 90%": (_continuous_scores,
                                       {"n_bins": 4, "conf_level": 0.9}, {
        "predicted": [0.0191266667, 0.1323766667, 0.3581266667, 0.6963766667],
        "observed": [0.0133333333, 0.16, 0.3866666667, 0.68],
        "successes": [1, 12, 29, 51],
        "n": [75] * 4,
        "conf_low": [0.0029802051, 0.1024142748, 0.2996576049, 0.5864650757],
        "conf_high": [0.0575758794, 0.2412618938, 0.4815677846, 0.7610004821],
    }),
    # The quantiles of tied scores fall on the scores themselves, so the
    # repeated edges merge. A bin includes its upper edge, so the cases at
    # 0.30 join those at 0.10 in the lowest bin, and three bins remain.
    "tied scores, 10 bins, 95%": (_tied_scores, {}, {
        "predicted": [0.1838235294, 0.55, 0.85],
        "observed": [0.1838235294, 0.5365853659, 0.7391304348],
        "successes": [25, 22, 17],
        "n": [136, 41, 23],
        "conf_low": [0.1277258889, 0.3874647127, 0.5352999516],
        "conf_high": [0.2572919683, 0.6794376602, 0.8745138396],
    }),
}


@pytest.mark.parametrize("case", list(_R_CALIBRATION))
def test_calibration_bins_and_intervals_match_the_r_twin(case):
    # Regression: the bins were scikit-learn's equal-width ones, with no count
    # and no interval, so the upper points of a rare outcome rested on one to
    # four cases with nothing to say so. The R twin cuts the scores at their
    # quantiles and draws a Wilson interval on each bin's observed rate.
    data, kwargs, expected = _R_CALIBRATION[case]
    y, score = data()
    got = dp.calibration_plot(y, score, **kwargs).data
    for column in ("predicted", "observed", "conf_low", "conf_high"):
        np.testing.assert_allclose(got[column], expected[column], atol=1e-9,
                                   err_msg=column)
    for column in ("successes", "n"):
        assert got[column].tolist() == expected[column], column


def test_calibration_draws_counts_and_wilson_intervals():
    y, score = _continuous_scores()
    p = dp.calibration_plot(y, score)
    geoms = [type(layer.geom).__name__ for layer in p.layers]
    assert geoms == ["geom_abline", "geom_errorbar", "geom_line", "geom_point"]
    assert p.layers[-1].mapping["size"] == "n"
    assert _builds(p)

    # conf_level=None draws the points alone, as the R twin's NA does.
    q = dp.calibration_plot(y, score, conf_level=None)
    assert "geom_errorbar" not in [type(layer.geom).__name__ for layer in q.layers]
    assert not {"conf_low", "conf_high"} & set(q.data.columns)
    assert _builds(q)


def test_calibration_uniform_strategy_keeps_the_old_bins():
    # strategy="uniform" keeps the equal-width bins of depictr 0.3.0, which
    # scikit-learn computed. The top bin is empty here and is dropped.
    calibration = pytest.importorskip("sklearn.calibration")
    y, score = _continuous_scores()
    p = dp.calibration_plot(y, score, strategy="uniform")
    observed, predicted = calibration.calibration_curve(y, score, n_bins=10)
    np.testing.assert_allclose(p.data["predicted"], predicted, rtol=1e-12)
    np.testing.assert_allclose(p.data["observed"], observed, rtol=1e-12)
    assert len(p.data) == 9
    assert p.data["n"].sum() == len(y)


def test_calibration_takes_the_r_twins_bins_argument():
    y, score = _continuous_scores()
    pd.testing.assert_frame_equal(dp.calibration_plot(y, score, bins=4).data,
                                  dp.calibration_plot(y, score, n_bins=4).data)
    with pytest.raises(TypeError, match="not both"):
        dp.calibration_plot(y, score, n_bins=5, bins=4)


def test_calibration_refuses_what_it_cannot_bin():
    y, score = _continuous_scores()
    # A score is compared with an observed frequency, so it must be a
    # probability. A missing score is no probability either.
    with_nan = score.copy()
    with_nan[7] = np.nan
    for bad in (score * 2, score - 0.5, with_nan):
        with pytest.raises(ValueError,
                           match="`y_score` must hold probabilities between 0 and 1"):
            dp.calibration_plot(y, bad)
    # The message is the R twin's, word for word.
    with pytest.raises(ValueError,
                       match=re.escape("Too few distinct scores to form bins.")):
        dp.calibration_plot(y, np.full(len(y), 0.3))
    with pytest.raises(ValueError, match="`y_true` must hold outcomes coded 0/1"):
        dp.calibration_plot(np.where(y == 1, "yes", "no"), score)
    with pytest.raises(ValueError, match="`y_true` must hold outcomes coded 0/1"):
        dp.calibration_plot(np.where(y == 1, 1.0, np.nan), score)
    # scikit-learn also took outcomes coded -1/1. The docstring never allowed
    # them, and the R twin refuses them.
    with pytest.raises(ValueError, match="`y_true` must hold outcomes coded 0/1"):
        dp.calibration_plot(np.where(y == 1, 1, -1), score)
    with pytest.raises(ValueError, match="must have the same length"):
        dp.calibration_plot(y[:-1], score)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        dp.calibration_plot(y, score, conf_level=95)
    with pytest.raises(ValueError, match="`strategy` must be"):
        dp.calibration_plot(y, score, strategy="kmeans")
    for bad in (0, 2.5, True):
        with pytest.raises(ValueError, match="`n_bins` must be a positive whole"):
            dp.calibration_plot(y, score, n_bins=bad)
    # Booleans are an outcome coded 0/1.
    pd.testing.assert_frame_equal(dp.calibration_plot(y == 1, score).data,
                                  dp.calibration_plot(y, score).data)


def test_gain_and_lift_refuse_a_length_mismatch():
    # A score vector one short used to fail inside pandas, in words that name
    # neither argument. Ranked by runs of ties, it would pair outcomes with the
    # wrong scores without a word, so the lengths are compared first.
    y, score = _tied_scores()
    for fn in (dp.gain_plot, dp.lift_plot):
        with pytest.raises(ValueError, match="must have the same length"):
            fn(y, score[:-1])


def test_gain_and_lift_refuse_a_missing_outcome_or_score():
    # Regression: a missing outcome cut the curve short at its rank, and each
    # missing score sorted last on its own, so the end of the curve followed
    # the order of the rows. Counting a missing outcome as a negative would
    # hide the gap, so both are refused.
    y, score = _tied_scores()
    y_missing = y.astype(float)
    y_missing[5] = np.nan
    y_nullable = pd.Series(y, dtype="Int64")
    y_nullable[5] = pd.NA
    score_missing = score.copy()
    score_missing[[5, 6]] = np.nan
    for fn in (dp.gain_plot, dp.lift_plot):
        for bad in (y_missing, y_nullable):
            with pytest.raises(ValueError, match="`y_true` holds a missing value"):
                fn(bad, score)
        with pytest.raises(ValueError, match="`y_score` holds a missing value"):
            fn(y, score_missing)


def test_gains_lift_and_calibration_need_no_scikit_learn(monkeypatch):
    # The three are computed in numpy, so they draw without the optional
    # [classification] extra.
    for name in ("sklearn", "sklearn.calibration", "sklearn.metrics"):
        monkeypatch.setitem(sys.modules, name, None)
    y, score = _continuous_scores()
    for p in (dp.gain_plot(y, score), dp.lift_plot(y, score),
              dp.calibration_plot(y, score)):
        assert _builds(p)


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
