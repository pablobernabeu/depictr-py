"""Smoke tests: the extra EDA plots and the power curve build without error."""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

import depictr as dp
from depictr.eda_extra import explore_pairs, raincloud_plot
from depictr.palette import REFERENCE
from depictr.power import power_curve_plot

WB = dp.wellbeing_survey()
LD = dp.lexical_decision()
NUM = ["age", "income", "stress", "sleep_hours", "life_satisfaction"]


def _builds(p):
    # Single plots are ggplots; explore_pairs returns an arrange_plots
    # composition, which is not a ggplot but still draws.
    p.draw(show=False)
    return True


def _power_frame():
    ns = np.arange(10, 210, 20)
    return pd.DataFrame({"n": ns, "power": 1 / (1 + np.exp(-(ns - 90) / 25))})


def test_raincloud_grouped():
    p = raincloud_plot(LD, "RT", group="condition", title="RT")
    assert isinstance(p, ggplot)
    assert _builds(p)
    # Four-level group on the wellbeing survey.
    assert _builds(raincloud_plot(WB, "life_satisfaction", group="region"))


def test_raincloud_ungrouped():
    p = raincloud_plot(WB, "income")
    assert isinstance(p, ggplot)
    assert _builds(p)


def test_raincloud_bad_columns():
    with pytest.raises(KeyError):
        raincloud_plot(LD, "not_a_column")
    with pytest.raises(KeyError):
        raincloud_plot(LD, "RT", group="not_a_column")


def test_explore_pairs_default_and_explicit():
    assert _builds(explore_pairs(WB, cols=NUM))
    assert _builds(explore_pairs(LD))


def test_explore_pairs_caps_columns():
    # Six columns requested, but the matrix is capped at five.
    six = ["age", "income", "stress", "sleep_hours", "life_satisfaction", "age"]
    assert _builds(explore_pairs(WB, cols=six))


def test_explore_pairs_needs_two_columns():
    with pytest.raises(ValueError):
        explore_pairs(WB, cols=["age"])


def test_explore_pairs_bad_column():
    with pytest.raises(KeyError):
        explore_pairs(WB, cols=["age", "not_a_column"])


def test_power_curve_ungrouped():
    p = power_curve_plot(_power_frame())
    assert isinstance(p, ggplot)
    assert _builds(p)


def test_power_curve_grouped():
    base = _power_frame()
    ns = base["n"]
    grouped = pd.concat([
        base.assign(effect="small", power=1 / (1 + np.exp(-(ns - 130) / 25))),
        base.assign(effect="large", power=1 / (1 + np.exp(-(ns - 60) / 20))),
    ])
    p = power_curve_plot(grouped, group="effect", title="Power by effect size")
    assert isinstance(p, ggplot)
    assert _builds(p)


def test_power_curve_custom_column_names():
    df = _power_frame().rename(columns={"n": "sample", "power": "pwr"})
    assert _builds(power_curve_plot(df, n="sample", power="pwr"))


def test_power_curve_bad_column():
    with pytest.raises(KeyError):
        power_curve_plot(_power_frame(), power="not_a_column")


def _pilotr_power_rows():
    # Rows shaped like pilotr's power_curve_mixed() output: one per sample
    # size and focal effect, interleaved by sample size, with Wilson bounds in
    # power_lo and power_hi. The R tests use the same values.
    ns = [8, 8, 12, 12, 16, 16, 20, 20]
    effects = ["cond", "freq"] * 4
    power = [0.30, 0.10, 0.55, 0.20, 0.75, 0.31, 0.88, 0.42]
    lo = [0.19, 0.04, 0.42, 0.11, 0.62, 0.20, 0.77, 0.30]
    hi = [0.44, 0.21, 0.67, 0.33, 0.85, 0.44, 0.94, 0.55]
    return [
        {"n_subject": n, "effect": e, "power": pw, "power_lo": a, "power_hi": b}
        for n, e, pw, a, b in zip(ns, effects, power, lo, hi, strict=True)
    ]


def _geoms(p):
    return [type(layer.geom).__name__ for layer in p.layers]


def _layer_data(p, geom):
    # The computed data of one layer, as R's ggplot_build() gives it; the copy
    # keeps the build from altering the plot under test.
    built = copy.deepcopy(p)
    built._build()
    return built.layers[_geoms(built).index(geom)].data


def test_power_curve_reads_a_pilotr_list_one_line_per_effect():
    with pytest.warns(UserWarning, match="`effect`"):
        p = power_curve_plot(_pilotr_power_rows())
    assert _builds(p)
    line = _layer_data(p, "geom_line")
    assert line["group"].nunique() == 2
    assert line["color"].nunique() == 2
    assert line["linetype"].nunique() == 2
    # Each curve rises with n; without a group they alternated in one line.
    for _, g in line.groupby("group"):
        assert g.sort_values("x")["y"].is_monotonic_increasing
    ribbon = _layer_data(p, "geom_ribbon")
    assert sorted(ribbon["ymin"]) == sorted(r["power_lo"]
                                            for r in _pilotr_power_rows())
    assert p.labels.y == "Power"


def test_power_curve_named_group_needs_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = power_curve_plot(pd.DataFrame(_pilotr_power_rows()), group="effect")
    assert p.labels.color == "effect"
    assert p.labels.linetype == "effect"


def test_power_curve_no_longer_joins_stacked_curves_into_one_line():
    # Columns the default arguments always read, so the old code drew this
    # table as a single line alternating between the two effects.
    df = (pd.DataFrame(_pilotr_power_rows())
          .rename(columns={"n_subject": "n"})
          .drop(columns=["power_lo", "power_hi"]))
    with pytest.warns(UserWarning, match="`effect`"):
        p = power_curve_plot(df)
    line = _layer_data(p, "geom_line")
    assert line["group"].nunique() == 2
    for _, g in line.groupby("group"):
        assert g.sort_values("x")["y"].is_monotonic_increasing


def test_power_curve_refuses_stacked_curves_it_cannot_separate():
    df = pd.DataFrame(_pilotr_power_rows()).rename(columns={"effect": "focal"})
    with pytest.raises(ValueError,
                       match="Several rows share a sample size; name the column"):
        power_curve_plot(df)
    assert _builds(power_curve_plot(df, group="focal"))


def test_power_curve_target_band_and_y_label():
    df = pd.DataFrame({"N": [10, 20, 30], "pw": [0.3, 0.6, 0.85],
                       "lo": [0.2, 0.5, 0.78], "hi": [0.4, 0.7, 0.92]})
    p = power_curve_plot(df, n="N", power="pw", lower="lo", upper="hi",
                         target=0.9)
    assert "geom_ribbon" in _geoms(p)
    assert list(_layer_data(p, "geom_hline")["yintercept"]) == [0.9]
    assert p.labels.y == "pw"
    assert power_curve_plot(df, n="N", power="pw",
                            y_lab="Power").labels.y == "Power"
    assert "geom_hline" not in _geoms(
        power_curve_plot(df, n="N", power="pw", target=None))
    bars = power_curve_plot(df, n="N", power="pw", lower="lo", upper="hi",
                            ribbon=False)
    # Capless bars, as the R twin's geom_errorbar(width = 0) draws them.
    assert "geom_linerange" in _geoms(bars)
    assert "geom_ribbon" not in _geoms(bars)
    with pytest.raises(ValueError, match="`lower` and `upper`"):
        power_curve_plot(df, n="N", power="pw", lower="lo")


def test_power_curve_reads_a_precision_curve_with_one_effect():
    df = pd.DataFrame({"n_subject": [15, 30, 60], "param": "effect",
                       "p_meaningful": [0.14, 0.29, 0.525],
                       "p_meaningful_lo": [0.10, 0.23, 0.46],
                       "p_meaningful_hi": [0.19, 0.36, 0.59]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = power_curve_plot(df, power="p_meaningful")
    assert _builds(p)
    assert _layer_data(p, "geom_line")["group"].nunique() == 1
    assert "geom_ribbon" in _geoms(p)
    assert p.labels.y == "p_meaningful"


def _pilotr_solved():
    # The fields pilotr's target_n() returns for a solved sample size, of which
    # the plot reads value, lo and hi. The R tests use the same values.
    return {"value": 120.0, "lo": 104.0, "hi": 139.0, "level": 0.95,
            "target": 0.8, "x": "n_subject", "y": "power",
            "transform": "sqrt", "n": 120, "n_lo": 104, "n_hi": 139}


def _solved_curve():
    return pd.DataFrame({"n_subject": [40, 80, 120, 160, 200],
                         "power": [0.31, 0.62, 0.80, 0.90, 0.95],
                         "power_lo": [0.25, 0.55, 0.74, 0.85, 0.91],
                         "power_hi": [0.38, 0.68, 0.85, 0.93, 0.97]})


def test_power_curve_marks_a_solved_sample_size_and_its_interval():
    p = power_curve_plot(_solved_curve(), solved=_pilotr_solved())
    assert _builds(p)
    geoms = _geoms(p)
    vline = _layer_data(p, "geom_vline")
    assert list(vline["xintercept"]) == [120]
    assert list(vline["color"]) == [REFERENCE]
    band = _layer_data(p, "geom_rect")
    assert (list(band["xmin"]), list(band["xmax"])) == ([104], [139])
    assert list(band["fill"]) == [REFERENCE]
    # The band lies beneath the confidence ribbon and the curve.
    assert geoms.index("geom_rect") < geoms.index("geom_ribbon")
    assert geoms.index("geom_rect") < geoms.index("geom_line")

    # A one-row DataFrame of the same fields is read alike, and with no
    # `solved` neither layer is drawn.
    row = pd.DataFrame([{"value": 120, "lo": 104, "hi": 139}])
    assert _geoms(power_curve_plot(_solved_curve(), solved=row)) == geoms
    plain = _geoms(power_curve_plot(_solved_curve()))
    assert "geom_vline" not in plain
    assert "geom_rect" not in plain

    # An interval that reaches past the swept sizes widens the axis to show it.
    wide = copy.deepcopy(power_curve_plot(
        _solved_curve(), solved={"value": 190, "lo": 150, "hi": 260}))
    wide._build()
    x_lo, x_hi = wide.layout.panel_params[0].x.range
    assert x_lo <= 40
    assert x_hi >= 260


@pytest.mark.parametrize("solved", [
    120,
    [120, 104, 139],
    # A Series, as the R twin refuses a named vector.
    pd.Series({"value": 120, "lo": 104, "hi": 139}),
    {"value": 120, "lo": 104},
    {"value": "120", "lo": 104, "hi": 139},
    {"value": float("nan"), "lo": 104, "hi": 139},
    {"value": True, "lo": 104, "hi": 139},
    {"value": [120, 130], "lo": 104, "hi": 139},
    pd.DataFrame({"value": [120, 130], "lo": 104, "hi": 139}),
])
def test_power_curve_refuses_a_solved_it_cannot_read(solved):
    with pytest.raises(ValueError, match=(
            r"^`solved` needs numeric `value`, `lo` and `hi`, as pilotr's "
            r"target_n\(\) returns\.$")):
        power_curve_plot(_solved_curve(), solved=solved)


def test_power_curve_one_level_effect_draws_one_line_without_a_legend():
    # pilotr's power_curve_mixed() names the focal effect even when there is
    # only one, and a column with one level separates nothing.
    rows = [r for r in _pilotr_power_rows() if r["effect"] == "cond"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = power_curve_plot(rows)
    assert _builds(p)
    assert _layer_data(p, "geom_line")["group"].nunique() == 1
    assert "color" not in p.mapping
    assert "linetype" not in p.mapping
    assert p.labels.color is None
    assert p.labels.linetype is None
