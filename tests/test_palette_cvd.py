"""Tests for the palette and the colour-vision-deficiency machinery."""

import re

import numpy as np
import pytest

from depictr import depictr_palette, palette_safety, simulate_cvd
from depictr.cvd import _rgb_to_lab
from depictr.palette import OKABE_ITO

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_qualitative_palette_defaults_to_full_set():
    pal = depictr_palette()
    assert pal == OKABE_ITO
    assert all(HEX.match(c) for c in pal)


def test_qualitative_slices_and_interpolates():
    assert depictr_palette(3) == OKABE_ITO[:3]
    long = depictr_palette(12)
    assert len(long) == 12 and all(HEX.match(c) for c in long)


def test_sequential_and_diverging_ramps():
    assert len(depictr_palette(5, kind="sequential")) == 5
    assert len(depictr_palette(9, kind="diverging")) == 9


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        depictr_palette(kind="nonsense")


def test_simulate_cvd_validates_inputs():
    with pytest.raises(ValueError):
        simulate_cvd(["#005b96"], "not-a-deficiency")
    with pytest.raises(ValueError):
        simulate_cvd(["#005b96"], "deutan", severity=2)


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


def test_near_identical_colours_flagged_unsafe():
    # Two colours a hair apart stay close under every condition, so the check
    # must flag them. (Pure red vs green, by contrast, differ in lightness and
    # remain distinguishable under colour-vision deficiency, so they are safe.)
    report = palette_safety(["#005b96", "#015c97"])
    assert report["safe"] is False
