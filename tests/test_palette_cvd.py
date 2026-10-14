"""Tests for the palette and the colour-vision-deficiency machinery."""

import csv
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from depictr import depictr_accent, depictr_palette, palette_safety, simulate_cvd
from depictr.audit import _contrast_ratio
from depictr.cvd import _ciede2000, _rgb_to_lab
from depictr.palette import OKABE_ITO

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

# ggplot2's default palette for five groups, scales::hue_pal()(5).
HUE5 = ["#F8766D", "#A3A500", "#00BF7D", "#00B0F6", "#E76BF3"]


def test_qualitative_palette_defaults_to_full_set():
    pal = depictr_palette()
    assert pal == OKABE_ITO
    assert all(HEX.match(c) for c in pal)


def test_qualitative_slices_and_interpolates():
    assert depictr_palette(3) == OKABE_ITO[:3]
    with pytest.warns(UserWarning, match="guarantee holds only up to 8"):
        long = depictr_palette(12)
    assert len(long) == 12 and all(HEX.match(c) for c in long)


def test_interpolating_past_the_base_set_warns():
    # The base set itself must stay quiet: the warning is about interpolation,
    # not about asking for a slice.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        depictr_palette()
        depictr_palette(len(OKABE_ITO))
        depictr_palette(20, kind="sequential")
    with pytest.warns(UserWarning, match="colour-vision-deficiency guarantee"):
        depictr_palette(len(OKABE_ITO) + 1)


def test_interpolated_palette_is_not_colourblind_safe():
    # Pins the limit rather than leaving it to be discovered: past the eight
    # base colours the interpolated set fails the package's own check, which is
    # exactly why depictr_palette warns there.
    with pytest.warns(UserWarning):
        interpolated = depictr_palette(12)
    assert palette_safety(interpolated)["safe"] is False
    # The size the guarantee is actually made for still clears the threshold.
    assert palette_safety(depictr_palette(len(OKABE_ITO)))["safe"] is True
    # Nine colours fail too, and nine is where the warning starts. The 1976
    # distance passed them, at 5.63.
    with pytest.warns(UserWarning):
        nine = depictr_palette(9)
    assert palette_safety(nine)["safe"] is False
    assert palette_safety(nine, metric="cie76")["safe"] is True


def test_sequential_and_diverging_ramps():
    assert len(depictr_palette(5, kind="sequential")) == 5
    assert len(depictr_palette(9, kind="diverging")) == 9


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        depictr_palette(kind="nonsense")


# Pinned identically in the R twin's tests/testthat/test-t4-theming.R.
CONTRAST_FIRST = ["#005b96", "#d55e00", "#cc79a7", "#009e73",
                  "#e69f00", "#56b4e9", "#f0e442", "#999999"]


def test_the_accent_is_the_r_twins_vermillion_and_clears_3_to_1():
    # The accent was the orange #e69f00, at 2.25:1 against white, below the 3:1
    # WCAG 2.2 asks of a graphical object. The R twin's accent is the vermillion
    # (options.R), at 3.87:1.
    assert depictr_accent() == "#d55e00"
    assert _contrast_ratio([depictr_accent()], ["#ffffff"])[0] >= 3
    # It stays a palette colour, in its place in the default order.
    assert OKABE_ITO.index(depictr_accent()) == 3


def test_contrast_first_order_puts_the_darker_colours_first():
    assert depictr_palette(order="contrast") == CONTRAST_FIRST
    assert depictr_palette(3, order="contrast") == CONTRAST_FIRST[:3]
    # The same eight colours, so the guarantee about the set still holds, and
    # the default order is unchanged.
    assert sorted(depictr_palette(order="contrast")) == sorted(OKABE_ITO)
    assert depictr_palette(order="default") == OKABE_ITO
    assert depictr_palette(3) == ["#005b96", "#e69f00", "#009e73"]

    # The figures the documentation quotes. Three colours stay 14.01 apart in
    # CIEDE2000 under every vision type and 6.86 apart in lightness, and the
    # first four clear 3:1 against white, which the fifth does not.
    assert palette_safety(CONTRAST_FIRST[:3])["min_delta_e"] == 14.01
    lightness = np.sort(_rgb_to_lab(CONTRAST_FIRST[:3])[:, 0])
    assert round(float(np.min(np.diff(lightness))), 2) == 6.86
    ratios = _contrast_ratio(CONTRAST_FIRST, ["#ffffff"] * 8)
    assert np.all(ratios[:4] >= 3)
    assert ratios[4] < 3

    # Under this order the accent is the second group's colour.
    assert depictr_palette(2, order="contrast")[1] == depictr_accent()


def test_contrast_first_order_warns_past_four_colours():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        depictr_palette(4, order="contrast")
        # The whole set names no number of groups.
        depictr_palette(order="contrast")
    message = ("^Past four colours the contrast-first order no longer keeps "
               "every colour at 3:1 against white\\.$")
    with pytest.warns(UserWarning, match=message):
        five = depictr_palette(5, order="contrast")
    assert five == CONTRAST_FIRST[:5]


def test_only_the_qualitative_palette_takes_an_order():
    for kind in ("sequential", "diverging"):
        with pytest.raises(ValueError, match=(
                f"^`order` applies to the qualitative palette, not to the "
                f"{kind} ramp\\.$")):
            depictr_palette(5, kind=kind, order="contrast")
    assert len(depictr_palette(5, kind="sequential", order="default")) == 5
    with pytest.raises(ValueError,
                       match="^`order` must be 'default' or 'contrast'\\.$"):
        depictr_palette(3, order="brightest")


def test_simulate_cvd_validates_inputs():
    with pytest.raises(ValueError) as excinfo:
        simulate_cvd(["#005b96"], "not-a-deficiency")
    assert str(excinfo.value) == (
        "`deficiency` must be one of 'protan', 'deutan' or 'tritan'.")
    with pytest.raises(ValueError) as excinfo:
        simulate_cvd(["#005b96"], "deutan", severity=2)
    assert str(excinfo.value) == "`severity` must lie in [0, 1]."


def test_simulate_cvd_matches_colorspacious():
    # colorspacious implements the same Machado model from the published
    # matrices. Agreement to the eight-bit colour is the strongest external
    # check available, so it is worth the optional dependency.
    colorspacious = pytest.importorskip("colorspacious")
    import matplotlib.colors as mcolors

    palette = depictr_palette()
    rgb = np.array([mcolors.to_rgb(c) for c in palette])
    names = {"protan": "protanomaly", "deutan": "deuteranomaly",
             "tritan": "tritanomaly"}
    for deficiency, name in names.items():
        space = {"name": "sRGB1+CVD", "cvd_type": name, "severity": 100}
        theirs = [mcolors.to_hex(np.clip(row, 0, 1))
                  for row in colorspacious.cspace_convert(rgb, space, "sRGB1")]
        assert simulate_cvd(palette, deficiency) == theirs, deficiency


def test_severity_zero_is_identity():
    colours = ["#005b96", "#e69f00"]
    assert simulate_cvd(colours, "deutan", severity=0.0) == [
        c.lower() for c in colours
    ]


def test_red_and_green_converge_under_deuteranopia():
    # The defining confusion: red and green move together, blue stays apart.
    sim = simulate_cvd(["#ff0000", "#00ff00", "#0000ff"], "deutan")
    lab = _rgb_to_lab(sim)
    d_rg = np.linalg.norm(lab[0] - lab[1])
    d_rb = np.linalg.norm(lab[0] - lab[2])
    assert d_rg < d_rb


def test_default_palette_is_safe():
    report = palette_safety()
    assert report["safe"] is True
    assert set(report["by_condition"]) == {"normal", "protan", "deutan", "tritan"}
    assert report["worst_condition"] in report["by_condition"]
    # Pinned in the R suite as well. The worst case is the reddish purple
    # against the grey under deuteranopia, which the 1976 distance put at 7.4.
    assert report["by_condition"] == {"normal": 21.25, "protan": 13.44,
                                      "deutan": 6.43, "tritan": 11.13}
    assert report["min_delta_e"] == 6.43
    assert report["worst_condition"] == "deutan"
    assert report["worst_pair"] == ("#cc79a7", "#999999")
    assert palette_safety(metric="cie76")["min_delta_e"] == 7.4


def test_ciede2000_reproduces_the_pairs_sharma_wu_and_dalal_published():
    # Their supplementary test data, as the authors distribute it, with the
    # differences given to four decimals. The pairs include the
    # discontinuities in the mean hue and the hue difference, where
    # implementations go wrong. The R suite reads an identical copy.
    path = Path(__file__).parent / "data" / "sharma2005.csv"
    with path.open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 34
    for row in pairs:
        first = [float(row[k]) for k in ("L1", "a1", "b1")]
        second = [float(row[k]) for k in ("L2", "a2", "b2")]
        ours = _ciede2000(first, second)
        assert abs(ours - float(row["delta_e_2000"])) <= 1e-4, row["pair"]
        # The formula is symmetric in its two colours.
        assert _ciede2000(second, first) == pytest.approx(ours, rel=1e-12)


def test_the_1976_distance_overstates_differences_between_saturated_colours():
    # Two blues a reader can barely tell apart, measured both ways. The R suite
    # pins the same numbers.
    blues = ["#005b96", "#0060a8"]
    assert palette_safety(blues)["by_condition"]["normal"] == 2.69
    assert palette_safety(blues, metric="cie76")["by_condition"]["normal"] == 8.2


def test_ggplot2s_five_colours_pass_under_cie76_and_fail_under_ciede2000():
    # Under deuteranopia the sky blue and the pink become #709df5 and #80a3ef,
    # 7.35 apart by the 1976 distance, which certified the palette as safe,
    # and 2.74 apart in CIEDE2000.
    assert simulate_cvd(["#00B0F6", "#E76BF3"], "deutan") == [
        "#709df5", "#80a3ef"]
    now = palette_safety(HUE5)
    assert now["safe"] is False
    assert now["min_delta_e"] == 2.74
    assert now["worst_condition"] == "deutan"
    assert now["worst_pair"] == ("#00B0F6", "#E76BF3")
    assert palette_safety(HUE5, metric="ciede2000") == now
    old = palette_safety(HUE5, metric="cie76")
    assert old["safe"] is True
    assert old["min_delta_e"] == 7.35
    assert old["worst_pair"] == ("#00B0F6", "#E76BF3")


def test_palette_safety_refuses_a_metric_it_does_not_know():
    with pytest.raises(ValueError) as excinfo:
        palette_safety(metric="cie94")
    assert str(excinfo.value) == "`metric` must be 'ciede2000' or 'cie76'."


def test_palette_safety_needs_a_pair():
    # One colour has no pairwise distance. The no-pair sentinel used to survive
    # to the result as min_delta_e = inf with safe=True, and a worst_pair that
    # named the same colour twice.
    with pytest.raises(ValueError, match="at least two colours"):
        palette_safety(["#005b96"])
    # An empty list is a caller error too, not a request for the default set.
    with pytest.raises(ValueError, match="at least two colours"):
        palette_safety([])
    # Passing nothing at all still means the default palette.
    assert palette_safety()["worst_pair"] == palette_safety(OKABE_ITO)["worst_pair"]


def test_near_identical_colours_flagged_unsafe():
    # Two colours a hair apart stay close under every condition, so the check
    # must flag them. (Pure red vs green, by contrast, differ in lightness and
    # remain distinguishable under colour-vision deficiency, so they are safe.)
    report = palette_safety(["#005b96", "#015c97"])
    assert report["safe"] is False
