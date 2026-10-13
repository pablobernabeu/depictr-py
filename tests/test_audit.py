"""Tests for check_figure(), the accessibility and honesty audit."""

import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import (
    aes,
    annotate,
    element_blank,
    element_rect,
    element_text,
    facet_wrap,
    geom_bin_2d,
    geom_col,
    geom_density,
    geom_hline,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_colour_gradient,
    scale_colour_identity,
    scale_colour_manual,
    scale_fill_manual,
    scale_size_identity,
    theme,
    theme_dark,
    theme_void,
)
from plotnine.composition import plot_spacer

import depictr as dp
from depictr.cvd import _rgb_to_lab

CY = dp.crop_yield()
WB = dp.wellbeing_survey()
LD = dp.lexical_decision()

CHECKS = [
    "colour_separability", "colour_separability_protan",
    "colour_separability_deutan", "colour_separability_tritan",
    "greyscale_separability", "text_size", "text_contrast",
    "geometry_contrast", "redundant_encoding", "annotation_text_size",
]

# Every colour row, which abstains with one shared detail when nothing discrete
# encodes.
COLOUR_CHECKS = CHECKS[:5] + ["geometry_contrast", "redundant_encoding"]


# Two reference figures with known answers. The good one is separable under
# every deficiency and in greyscale, contrasts well against white, keeps its
# text at the size it was drawn and adds a redundant shape. The bad one uses a
# red and a green that collapse to the same colour under deuteranopia, shrinks
# its text far below any print floor and leaves colour to do all the work.
def good_figure(**kwargs):
    return (ggplot(CY, aes("rainfall", "yield", colour="treatment",
                           shape="treatment"))
            + geom_point()
            + scale_colour_manual(values=["#005b96", "#d55e00"])
            + dp.theme_depictr(**kwargs))


def bad_figure():
    return (ggplot(CY, aes("rainfall", "yield", colour="treatment"))
            + geom_point()
            + scale_colour_manual(values=["#d62728", "#309208"])
            + dp.theme_depictr(base_size=5))


def measured_for(report, check):
    return report.loc[report["check"] == check, "measured"].item()


def verdict_for(report, check):
    return report.loc[report["check"] == check, "verdict"].item()


def detail_for(report, check):
    return report.loc[report["check"] == check, "detail"].item()


def threshold_for(report, check):
    return report.loc[report["check"] == check, "threshold"].item()


def wcag_contrast(a, b):
    """The WCAG 2.2 ratio recomputed from the published definition alone.

    Kept independent of the package code: each sRGB channel is linearised at the
    0.04045 breakpoint, the channels are weighted, then
    (L1 + 0.05) / (L2 + 0.05).
    """
    import matplotlib.colors as mcolors

    def luminance(colour):
        channels = np.array(mcolors.to_rgb(colour))
        linear = np.where(channels <= 0.04045, channels / 12.92,
                          ((channels + 0.055) / 1.055) ** 2.4)
        return float(linear @ np.array([0.2126, 0.7152, 0.0722]))

    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def mix(colour, background, alpha):
    """A colour as it shows over a background at a given alpha, by hand.

    Each channel in [0, 1] is alpha * colour + (1 - alpha) * background, then
    floor(255 * x + 0.5) gives the 8-bit value.
    """
    import matplotlib.colors as mcolors

    fg = np.array(mcolors.to_rgb(colour))
    bg = np.array(mcolors.to_rgb(background))
    channels = np.floor(255 * (alpha * fg + (1 - alpha) * bg) + 0.5)
    return "#" + "".join(f"{int(c):02x}" for c in channels)


# Translucent versions of the good figure's colours, built the same way in both
# twins. `alpha=None` leaves the layer's alpha unset.
def bars_figure(alpha, values=("#005b96", "#d55e00")):
    two = pd.DataFrame({"g": ["a", "b"], "y": [1, 2]})
    bars = geom_col() if alpha is None else geom_col(alpha=alpha)
    return (ggplot(two, aes("g", "y", fill="g")) + bars
            + scale_fill_manual(values=list(values))
            + dp.theme_depictr())


def points_figure(alpha):
    return (ggplot(CY, aes("rainfall", "yield", colour="treatment",
                           shape="treatment"))
            + geom_point(alpha=alpha)
            + scale_colour_manual(values=["#005b96", "#d55e00"])
            + dp.theme_depictr())


def density_figure(alpha=0.2):
    # Densities whose fill is translucent and whose outline, mapped to the same
    # groups, is drawn opaque.
    return (ggplot(CY, aes("yield", colour="treatment", fill="treatment"))
            + geom_density(alpha=alpha)
            + scale_colour_manual(values=["#005b96", "#d55e00"])
            + scale_fill_manual(values=["#005b96", "#d55e00"])
            + dp.theme_depictr())


# The bad figure's red and green, drawn one group per layer. Each layer maps
# colour to a constant, so neither varies in colour on its own. plotnine
# evaluates a string in aes() as an expression, so a constant is quoted twice.
STANDARD = CY[CY["treatment"] == "standard"]
ENHANCED = CY[CY["treatment"] == "enhanced"]


def split_figure():
    return (ggplot(mapping=aes("rainfall", "yield"))
            + geom_point(aes(colour="'a'"), data=STANDARD)
            + geom_point(aes(colour="'b'"), data=ENHANCED)
            + scale_colour_manual(values=["#d62728", "#309208"])
            + dp.theme_depictr())


def annotated_figure(size=5.7):
    # The good figure with a note drawn inside the panel at 5.7 pt. plotnine
    # takes a layer's text size in points. The R suite draws the same note at
    # 5.7 / .pt, since ggplot2 takes it in millimetres.
    return good_figure() + annotate("text", x=800, y=5, label="note", size=size)


def gradient_figure():
    # Points coloured by a continuous variable, on the default gradient.
    return (ggplot(CY, aes("rainfall", "yield", colour="yield")) + geom_point()
            + dp.theme_depictr())


def encoding_rows(report):
    # The rows that depend on what a figure tells apart by colour, which is
    # every row but the three text rows.
    rows = report[~report["check"].isin(
        ["text_size", "text_contrast", "annotation_text_size"])]
    return rows.reset_index(drop=True)


# --- shape of the result -----------------------------------------------------


def test_report_has_the_standard_table_shape():
    report = dp.check_figure(good_figure())
    assert isinstance(report, pd.DataFrame)
    assert report.columns.tolist() == [
        "check", "measured", "threshold", "verdict", "detail"]
    assert report["check"].tolist() == CHECKS
    assert len(report) == 10
    assert set(report["verdict"]) <= {"pass", "fail", "not applicable"}
    assert report["detail"].map(bool).all()


# --- the two verdicts the audit must be capable of ---------------------------


def test_a_deliberately_good_figure_passes_every_check_that_applies():
    report = dp.check_figure(good_figure())
    # It draws no text inside the panel, so that row has nothing to measure.
    measured = report[report["check"] != "annotation_text_size"]
    assert set(measured["verdict"]) == {"pass"}, report.to_string()
    assert verdict_for(report, "annotation_text_size") == "not applicable"


def test_a_deliberately_bad_figure_fails_the_checks_it_should():
    report = dp.check_figure(bad_figure())
    # Red against green: fine to a normal-sighted reader, gone under
    # deuteranopia.
    assert verdict_for(report, "colour_separability") == "pass"
    assert measured_for(report, "colour_separability") > 100
    assert verdict_for(report, "colour_separability_deutan") == "fail"
    assert measured_for(report, "colour_separability_deutan") < 1
    # Base size 5 puts the axis text at 4 pt, well under the 6 pt floor.
    assert verdict_for(report, "text_size") == "fail"
    assert measured_for(report, "text_size") == 4
    # Nothing but colour separates the two groups.
    assert verdict_for(report, "redundant_encoding") == "fail"
    assert measured_for(report, "redundant_encoding") == 0
    # The rest of the figure is fine, so the audit is not simply failing it.
    assert verdict_for(report, "text_contrast") == "pass"
    assert verdict_for(report, "geometry_contrast") == "pass"


def test_the_audit_reports_the_numbers_its_r_twin_reports():
    # Pinned so the two engines cannot drift apart unnoticed. Each figure below
    # is constructible in both, and the R suite pins the same vectors. None of
    # them draws text inside the panel, so each ends in NaN, which
    # assert_array_equal, unlike ==, treats as equal to itself.
    np.testing.assert_array_equal(
        dp.check_figure(good_figure())["measured"],
        [111.87, 89.87, 107.07, 98.5, 16.92, 8.8, 8.45, 3.87, 1, np.nan])
    np.testing.assert_array_equal(
        dp.check_figure(bad_figure())["measured"],
        [116.25, 35.04, 0.37, 119.38, 6.32, 4, 8.45, 4.01, 0, np.nan])

    eight = pd.DataFrame({"g": list("abcdefgh"), "x": range(8), "y": range(8)})
    p = (ggplot(eight, aes("x", "y", colour="g")) + geom_point()
         + dp.scale_colour_depictr() + dp.theme_depictr())
    report = dp.check_figure(p)
    np.testing.assert_array_equal(
        report["measured"],
        [33.43, 18.15, 7.4, 16.18, 0.79, 8.8, 8.45, 1.32, 0, np.nan])
    # The detail strings are part of the contract too, since they carry the
    # colours the numbers came from.
    assert report["detail"].tolist() == [
        "Closest pair #d55e00 and #e69f00 of 8 encoding colours.",
        "Closest pair #999999 and #cc79a7 of 8 encoding colours.",
        "Closest pair #999999 and #cc79a7 of 8 encoding colours.",
        "Closest pair #009e73 and #56b4e9 of 8 encoding colours.",
        "Closest pair #56b4e9 and #e69f00 in CIE lightness.",
        "Smallest text 8.80 pt, drawn at 17.78 cm and printed at 17.78 cm.",
        "Lowest-contrast text #4d4d4d on #ffffff at 8.80 pt.",
        "Lowest-contrast colour #f0e442 on #ffffff.",
        "Colour alone distinguishes the groups.",
        "The figure draws no text inside the panel.",
    ]

    # A note drawn inside the panel at 5.7 pt, below the 6 pt floor. The
    # text-size check never saw it, so the report used to read 8.80 pt as the
    # smallest text.
    annotated = dp.check_figure(annotated_figure())
    np.testing.assert_array_equal(
        annotated["measured"],
        [111.87, 89.87, 107.07, 98.5, 16.92, 8.8, 8.45, 3.87, 1, 5.7])
    assert [detail_for(annotated, "text_size"),
            detail_for(annotated, "annotation_text_size")] == [
        "Smallest text 8.80 pt, drawn at 17.78 cm and printed at 17.78 cm.",
        "Smallest in-panel text 5.70 pt, drawn at 17.78 cm and printed at "
        "17.78 cm.",
    ]

    # A dark plot background behind a blank panel: the geometry is measured
    # against the plot background in both twins. This twin used to read the
    # invisible axes patch here, which is white, and passed the figure at 3.87.
    dark = dp.check_figure(good_figure()
                           + theme(plot_background=element_rect(fill="#333333")))
    np.testing.assert_array_equal(
        dark["measured"],
        [111.87, 89.87, 107.07, 98.5, 16.92, 8.8, 1.49, 1.77, 1, np.nan])
    assert (detail_for(dark, "geometry_contrast")
            == "Lowest-contrast colour #005b96 on #333333.")

    # Translucent bars, measured as drawn: #005b96 and #d55e00 at alpha 0.3
    # show as #b3cee0 and #f2cfb3. Read at full opacity, the figure passed
    # geometry at 3.87 and greyscale at 16.92.
    bars = dp.check_figure(bars_figure(alpha=0.3))
    np.testing.assert_array_equal(
        bars["measured"],
        [33.44, 27.98, 33.13, 35.37, 4.03, 8.8, 8.45, 1.46, 0, np.nan])
    assert bars["detail"].tolist() == [
        *["Closest pair #b3cee0 and #f2cfb3 of 2 encoding fill colours."] * 4,
        "Closest pair #b3cee0 and #f2cfb3 in CIE lightness.",
        "Smallest text 8.80 pt, drawn at 17.78 cm and printed at 17.78 cm.",
        "Lowest-contrast text #4d4d4d on #ffffff at 8.80 pt.",
        "Lowest-contrast colour #d55e00 at alpha 0.30, drawn as #f2cfb3, "
        "on #ffffff.",
        "Colour alone distinguishes the groups.",
        "The figure draws no text inside the panel.",
    ]

    # The good figure's points at alpha 0.7, which the accessibility page used
    # to show as mended. Its vermillion is drawn as #e28e4d, at 2.56:1.
    points = dp.check_figure(points_figure(alpha=0.7))
    np.testing.assert_array_equal(
        points["measured"],
        [83.25, 69.5, 80.89, 85.03, 10.77, 8.8, 8.45, 2.56, 1, np.nan])
    assert (detail_for(points, "geometry_contrast")
            == "Lowest-contrast colour #d55e00 at alpha 0.70, drawn as "
               "#e28e4d, on #ffffff.")

    # The bad figure's red and green with one group per layer, each layer
    # mapping colour to a constant. Every colour row used to read "not
    # applicable".
    split = dp.check_figure(split_figure())
    np.testing.assert_array_equal(
        split["measured"],
        [116.25, 35.04, 0.37, 119.38, 6.32, 8.8, 8.45, 4.01, 0, np.nan])
    assert [*split["detail"][:5], detail_for(split, "geometry_contrast")] == [
        *["Closest pair #309208 and #d62728 of 2 encoding colours."] * 4,
        "Closest pair #309208 and #d62728 in CIE lightness.",
        "Lowest-contrast colour #309208 on #ffffff.",
    ]

    # A scatter coloured by a continuous variable. Every colour row abstains
    # and says why, where it used to say that nothing is distinguished by
    # colour.
    gradient = dp.check_figure(gradient_figure())
    np.testing.assert_array_equal(
        gradient["measured"],
        [np.nan] * 5 + [8.8, 8.45] + [np.nan] * 3)
    assert set(gradient.loc[gradient["check"].isin(COLOUR_CHECKS), "detail"]) == {
        "Colour is a continuous scale, which these checks do not audit."}


# --- the measurements themselves ---------------------------------------------


def test_colour_separability_is_the_palette_check_on_the_figures_colours():
    # The figure's two colours are what palette_safety would report on them, so
    # the two exports cannot drift apart.
    report = dp.check_figure(good_figure())
    reference = dp.palette_safety(["#005b96", "#d55e00"])["by_condition"]
    assert measured_for(report, "colour_separability") == reference["normal"]
    for deficiency in ("protan", "deutan", "tritan"):
        assert (measured_for(report, f"colour_separability_{deficiency}")
                == reference[deficiency])


def test_greyscale_separability_fails_for_the_default_palette():
    eight = pd.DataFrame({"g": list("abcdefgh"), "x": range(8), "y": range(8)})
    p = (ggplot(eight, aes("x", "y", colour="g")) + geom_point()
         + dp.scale_colour_depictr() + dp.theme_depictr())
    report = dp.check_figure(p)
    # Measured independently: the orange and the sky blue of the Okabe-Ito set
    # sit 0.79 apart in CIE lightness, so they print as the same grey. The
    # threshold stays where it is and the documentation carries the limitation.
    lightness = _rgb_to_lab(["#e69f00", "#56b4e9"])[:, 0]
    assert (measured_for(report, "greyscale_separability")
            == round(float(abs(lightness[0] - lightness[1])), 2))
    assert measured_for(report, "greyscale_separability") == 0.79
    assert verdict_for(report, "greyscale_separability") == "fail"
    # The same eight colours clear every colour-vision check, which is the
    # point: the guarantee is about hue confusion, not black-and-white printing.
    for check in CHECKS[:4]:
        assert verdict_for(report, check) == "pass"


def test_text_size_scales_with_the_printed_width():
    p = good_figure()
    full = measured_for(dp.check_figure(p), "text_size")
    assert full == 8.8
    shrunk = dp.check_figure(p, width_cm=8.89)
    assert measured_for(shrunk, "text_size") == round(full / 2, 2)
    assert verdict_for(shrunk, "text_size") == "fail"
    # Drawing narrower in the first place is the other half of the ratio.
    drawn_narrow = dp.check_figure(p, width_cm=8.89, render_width_cm=8.89)
    assert measured_for(drawn_narrow, "text_size") == full


def test_contrast_ratios_match_the_wcag_definition():
    # Black on white is the textbook 21:1 anchor.
    assert round(wcag_contrast("#000000", "#ffffff"), 2) == 21
    report = dp.check_figure(good_figure())
    # theme_depictr draws the axis text in #4d4d4d on a white plot background,
    # and the vermillion is the lower-contrast of the two encoding colours.
    assert (measured_for(report, "text_contrast")
            == round(wcag_contrast("#4d4d4d", "#ffffff"), 2))
    assert (measured_for(report, "geometry_contrast")
            == round(wcag_contrast("#d55e00", "#ffffff"), 2))

    # The two ratios are measured against different backgrounds, so a theme that
    # repaints them has to move both. plotnine paints plot_background onto a
    # rectangle of its own and leaves the figure patch white, so reading the
    # patch kept reporting text against white however dark the figure was.
    repainted = dp.check_figure(good_figure() + theme(
        panel_background=element_rect(fill="#333333"),
        plot_background=element_rect(fill="#eeeeee")))
    assert (measured_for(repainted, "text_contrast")
            == round(wcag_contrast("#4d4d4d", "#eeeeee"), 2))
    assert (measured_for(repainted, "geometry_contrast")
            == round(wcag_contrast("#005b96", "#333333"), 2))
    assert (repainted.loc[repainted["check"] == "text_contrast",
                          "detail"].item()
            == "Lowest-contrast text #4d4d4d on #eeeeee at 8.80 pt.")
    # A blank plot background paints nothing, so white stands in for the paper.
    blanked = dp.check_figure(good_figure()
                              + theme(plot_background=element_blank()))
    assert (measured_for(blanked, "text_contrast")
            == round(wcag_contrast("#4d4d4d", "#ffffff"), 2))


def test_label_colour_is_black_or_white_whichever_contrasts_more():
    # The heatmaps choose each cell label's colour by this rule. Black and white
    # tie against a relative luminance of 0.179, which falls between the first
    # two greys, and each reaches 4.58:1 there. The R twin's
    # .legible_text_colour() pins the same vector.
    from depictr.audit import _legible_text_colour

    okabe_ito = ["#005b96", "#e69f00", "#009e73", "#d55e00", "#cc79a7",
                 "#56b4e9", "#f0e442", "#999999"]
    assert _legible_text_colour(["#757575", "#767676", *okabe_ito]) == (
        ["#ffffff", "#000000", "#ffffff"] + ["#000000"] * 7)
    # Every colour of a 16-level grid over the three channels.
    levels = [f"{v:02x}" for v in range(0, 256, 17)]
    fills = [f"#{r}{g}{b}" for r in levels for g in levels for b in levels]
    for fill, chosen in zip(fills, _legible_text_colour(fills), strict=True):
        other = "#ffffff" if chosen == "#000000" else "#000000"
        assert wcag_contrast(chosen, fill) >= wcag_contrast(other, fill)
        assert wcag_contrast(chosen, fill) >= 4.58


# --- each text against its own background ------------------------------------


def strip_figure(text_colour, strip_fill="#1a1a1a"):
    # A facetted figure with dark strips. The strip text size is set explicitly
    # because the twins' default strip sizes differ.
    return (good_figure() + facet_wrap("treatment")
            + theme(strip_background=element_rect(fill=strip_fill),
                    strip_text=element_text(color=text_colour, size=8.8)))


def test_strip_text_is_measured_against_its_strip():
    # White on a near-black strip reads at 17.4:1. Measured against the white
    # plot background it used to read 1.00 and fail.
    light = dp.check_figure(strip_figure("white"))
    assert measured_for(light, "text_contrast") == 8.45
    assert verdict_for(light, "text_contrast") == "pass"
    assert (detail_for(light, "text_contrast")
            == "Lowest-contrast text #4d4d4d on #ffffff at 8.80 pt.")
    # Dark grey on the same strip is unreadable. It used to pass at 8.45, the
    # value of the axis text, because it was measured against white.
    dark = dp.check_figure(strip_figure("#333333"))
    assert measured_for(dark, "text_contrast") == 1.38
    assert (measured_for(dark, "text_contrast")
            == round(wcag_contrast("#333333", "#1a1a1a"), 2))
    assert verdict_for(dark, "text_contrast") == "fail"
    assert (detail_for(dark, "text_contrast")
            == "Lowest-contrast text #333333 on #1a1a1a at 8.80 pt.")
    # A translucent strip is composited over the plot background before the
    # text is measured against it: black at alpha 128/255 over white is #7f7f7f.
    veiled = dp.check_figure(strip_figure("white", strip_fill="#00000080"))
    assert (measured_for(veiled, "text_contrast")
            == round(wcag_contrast("#ffffff", "#7f7f7f"), 2))
    assert (detail_for(veiled, "text_contrast")
            == "Lowest-contrast text #ffffff on #7f7f7f at 8.80 pt.")


def test_legend_text_is_measured_against_the_legend_box():
    def boxed(text_colour):
        return good_figure() + theme(
            legend_background=element_rect(fill="#222222"),
            legend_text=element_text(color=text_colour),
            legend_title=element_text(color="white"))

    light = dp.check_figure(boxed("white"))
    assert verdict_for(light, "text_contrast") == "pass"
    assert measured_for(light, "text_contrast") == 8.45
    dark = dp.check_figure(boxed("#333333"))
    assert (measured_for(dark, "text_contrast")
            == round(wcag_contrast("#333333", "#222222"), 2))
    assert (detail_for(dark, "text_contrast")
            == "Lowest-contrast text #333333 on #222222 at 8.80 pt.")
    # ggplot2 paints legend_box_background beneath the legends, and the R twin
    # measures against it. plotnine does not draw it (its themeable is marked
    # not implemented), so here the text sits on the plot background and the
    # white legend text fails. If plotnine starts drawing the box, this fails
    # and the box should be read as R reads it.
    outer = dp.check_figure(good_figure() + theme(
        legend_box_background=element_rect(fill="#222222"),
        legend_background=element_blank(),
        legend_text=element_text(color="white"),
        legend_title=element_text(color="white")))
    assert (detail_for(outer, "text_contrast")
            == "Lowest-contrast text #ffffff on #ffffff at 8.80 pt.")


def test_a_translucent_background_is_measured_as_the_colour_it_shows():
    # Blue at alpha 64/255 over the white plot background shows as #bfbfff. Its
    # alpha used to be dropped, so the panel was measured as opaque blue.
    report = dp.check_figure(good_figure() + theme(
        panel_background=element_rect(fill="#0000ff40")))
    assert (measured_for(report, "geometry_contrast")
            == round(wcag_contrast("#d55e00", "#bfbfff"), 2))
    assert (detail_for(report, "geometry_contrast")
            == "Lowest-contrast colour #d55e00 on #bfbfff.")


def test_stock_dark_theme_facets_pass_the_text_check():
    # theme_dark draws white strip text on a dark strip. Measured against the
    # white plot background, it used to fail at 1.00.
    report = dp.check_figure(good_figure() + facet_wrap("treatment")
                             + theme_dark())
    assert verdict_for(report, "text_contrast") == "pass"


def test_large_text_is_held_to_three_to_one_at_its_printed_size():
    def titled(**kwargs):
        return (good_figure() + labs(title="A title")
                + theme(plot_title=element_text(color="#8c8c8c", **kwargs)))

    # #8c8c8c on white is 3.36:1, short of 4.5 and clear of the 3 that WCAG 2.2
    # asks of text at 18 pt or more.
    large = dp.check_figure(titled(size=20, weight="normal"))
    assert measured_for(large, "text_contrast") == 3.36
    assert threshold_for(large, "text_contrast") == 3
    assert verdict_for(large, "text_contrast") == "pass"
    assert (detail_for(large, "text_contrast")
            == "Lowest-contrast text #8c8c8c on #ffffff at 20.00 pt.")
    # Bold text counts as large from 14 pt.
    bold = dp.check_figure(titled(size=14, weight="bold"))
    assert threshold_for(bold, "text_contrast") == 3
    assert verdict_for(bold, "text_contrast") == "pass"
    assert (detail_for(bold, "text_contrast")
            == "Lowest-contrast text #8c8c8c on #ffffff at 14.00 pt bold.")
    # The same plain title is not large at 14 pt, so it fails at 4.5.
    plain = dp.check_figure(titled(size=14, weight="normal"))
    assert threshold_for(plain, "text_contrast") == 4.5
    assert verdict_for(plain, "text_contrast") == "fail"
    # Size is judged as printed: the 20 pt title in a half-width column prints
    # at 10 pt, which is normal-sized text again.
    shrunk = dp.check_figure(titled(size=20, weight="normal"), width_cm=8.89)
    assert threshold_for(shrunk, "text_contrast") == 4.5
    assert verdict_for(shrunk, "text_contrast") == "fail"
    assert (detail_for(shrunk, "text_contrast")
            == "Lowest-contrast text #8c8c8c on #ffffff at 10.00 pt.")


def test_relative_luminance_follows_the_current_wcag_definition():
    # The definition's breakpoint is 0.04045. It read 0.03928 before May 2021,
    # and no 8-bit channel value lies between the two, so no ratio moves.
    from depictr.audit import _relative_luminance
    from depictr.cvd import _srgb_to_linear

    channels = np.arange(256) / 255
    greys = [f"#{v:02x}{v:02x}{v:02x}" for v in range(256)]
    old = np.where(channels <= 0.03928, channels / 12.92,
                   ((channels + 0.055) / 1.055) ** 2.4)
    np.testing.assert_allclose(_relative_luminance(greys), old)
    np.testing.assert_allclose(_relative_luminance(greys),
                               _srgb_to_linear(channels))


# --- translucent marks and text, measured as drawn ---------------------------


def test_translucent_fills_are_measured_as_the_colours_they_show():
    bars = dp.check_figure(bars_figure(alpha=0.3))
    assert mix("#005b96", "#ffffff", 0.3) == "#b3cee0"
    assert mix("#d55e00", "#ffffff", 0.3) == "#f2cfb3"
    assert (measured_for(bars, "geometry_contrast")
            == round(wcag_contrast("#f2cfb3", "#ffffff"), 2))
    assert verdict_for(bars, "geometry_contrast") == "fail"
    assert verdict_for(bars, "greyscale_separability") == "fail"
    # The colour rows are palette_safety() on the drawn colours.
    reference = dp.palette_safety(["#b3cee0", "#f2cfb3"])["by_condition"]
    assert measured_for(bars, "colour_separability") == reference["normal"]
    for deficiency in ("protan", "deutan", "tritan"):
        assert (measured_for(bars, f"colour_separability_{deficiency}")
                == reference[deficiency])
    # A fill is mixed with the panel it is drawn on, here a dark one.
    dark = dp.check_figure(bars_figure(alpha=0.3) + theme(
        panel_background=element_rect(fill="#333333")))
    blue = mix("#005b96", "#333333", 0.3)
    assert (measured_for(dark, "geometry_contrast")
            == round(wcag_contrast(blue, "#333333"), 2))
    assert detail_for(dark, "geometry_contrast") == (
        f"Lowest-contrast colour #005b96 at alpha 0.30, drawn as {blue}, "
        "on #333333.")
    # Opaque bars keep the detail they always had.
    assert (detail_for(dp.check_figure(bars_figure(alpha=1)),
                       "geometry_contrast")
            == "Lowest-contrast colour #d55e00 on #ffffff.")


def test_translucent_points_are_measured_as_drawn():
    points = dp.check_figure(points_figure(alpha=0.7))
    assert mix("#d55e00", "#ffffff", 0.7) == "#e28e4d"
    assert (measured_for(points, "geometry_contrast")
            == round(wcag_contrast("#e28e4d", "#ffffff"), 2))
    assert verdict_for(points, "geometry_contrast") == "fail"


def test_translucent_text_is_measured_as_drawn():
    # "#00000040" is black at 64/255, as R's alpha("black", 0.25) writes it.
    veiled = dp.check_figure(good_figure() + theme(
        axis_text=element_text(color="#00000040")))
    grey = mix("#000000", "#ffffff", 64 / 255)
    assert grey == "#bfbfbf"
    assert measured_for(veiled, "text_contrast") == 1.84
    assert (measured_for(veiled, "text_contrast")
            == round(wcag_contrast(grey, "#ffffff"), 2))
    assert verdict_for(veiled, "text_contrast") == "fail"
    assert detail_for(veiled, "text_contrast") == (
        "Lowest-contrast text #000000 at alpha 0.25, drawn as #bfbfbf, "
        "on #ffffff at 8.80 pt.")
    # When the artist has an alpha of its own, matplotlib draws the text in it
    # and ignores the colour's alpha, so this black is drawn at 0.5.
    faded = dp.check_figure(good_figure() + theme(
        axis_text=element_text(color="#00000040", alpha=0.5)))
    half = mix("#000000", "#ffffff", 0.5)
    assert (measured_for(faded, "text_contrast")
            == round(wcag_contrast(half, "#ffffff"), 2))
    assert detail_for(faded, "text_contrast") == (
        f"Lowest-contrast text #000000 at alpha 0.50, drawn as {half}, "
        "on #ffffff at 8.80 pt.")


def test_an_opaque_outline_mapped_to_the_same_groups_carries_the_contrast():
    # WCAG lets a border carry a graphical object's contrast. The fills at
    # alpha 0.2 are pale, and the outlines, drawn opaque, are what is measured.
    outlined = dp.check_figure(density_figure())
    assert (measured_for(outlined, "geometry_contrast")
            == round(wcag_contrast("#d55e00", "#ffffff"), 2))
    assert (detail_for(outlined, "geometry_contrast")
            == "Lowest-contrast colour #d55e00 on #ffffff.")
    # A constant black outline says nothing about which group a shape belongs
    # to, so it carries no credit and the pale fill is measured.
    plain = dp.check_figure(
        ggplot(CY, aes("yield", fill="treatment"))
        + geom_density(alpha=0.2, color="black")
        + scale_fill_manual(values=["#005b96", "#d55e00"])
        + dp.theme_depictr())
    pale = mix("#d55e00", "#ffffff", 0.2)
    assert (measured_for(plain, "geometry_contrast")
            == round(wcag_contrast(pale, "#ffffff"), 2))
    assert detail_for(plain, "geometry_contrast") == (
        f"Lowest-contrast colour #d55e00 at alpha 0.20, drawn as {pale}, "
        "on #ffffff.")


def test_each_aesthetics_colours_are_compared_among_themselves():
    white = "#ffffff"
    # The outlines and the pale fills of the densities form two sets, and the
    # row reports the closer of their two pairs, the fills.
    outlined = dp.check_figure(density_figure())
    fills = [mix("#005b96", white, 0.2), mix("#d55e00", white, 0.2)]
    assert (measured_for(outlined, "colour_separability")
            == dp.palette_safety(fills)["by_condition"]["normal"])
    assert detail_for(outlined, "colour_separability") == (
        f"Closest pair {fills[0]} and {fills[1]} of 2 encoding fill colours.")
    # A fill at alpha 0.9 is a hair from its own opaque outline. Pooled, that
    # pair failed the figure. It is one group, so it is never compared.
    near = dp.check_figure(density_figure(alpha=0.9))
    fills = [mix("#005b96", white, 0.9), mix("#d55e00", white, 0.9)]
    assert (measured_for(near, "colour_separability")
            == dp.palette_safety(fills)["by_condition"]["normal"])
    assert verdict_for(near, "colour_separability") == "pass"
    # Nor across layers: translucent points beside opaque lines in the same two
    # colours are compared group against group, never a group with itself.
    layered = dp.check_figure(
        ggplot(CY, aes("rainfall", "yield", colour="treatment"))
        + geom_point(alpha=0.95) + geom_line()
        + scale_colour_manual(values=["#005b96", "#d55e00"])
        + dp.theme_depictr())
    blue = ["#005b96", mix("#005b96", white, 0.95)]
    vermillion = ["#d55e00", mix("#d55e00", white, 0.95)]
    across = min(dp.palette_safety([b, v])["by_condition"]["normal"]
                 for b in blue for v in vermillion)
    assert measured_for(layered, "colour_separability") == across
    assert verdict_for(layered, "colour_separability") == "pass"


def test_verdicts_on_translucent_fills_can_move_in_either_direction():
    # Red and green fills at alpha 0.6 are lighter by different amounts, so a
    # deuteranope tells them apart by lightness. Read at full opacity, they
    # failed at 0.37.
    red_green = ("#d62728", "#309208")
    opaque = dp.check_figure(bars_figure(alpha=1, values=red_green))
    translucent = dp.check_figure(bars_figure(alpha=0.6, values=red_green))
    assert measured_for(opaque, "colour_separability_deutan") == 0.37
    assert verdict_for(opaque, "colour_separability_deutan") == "fail"
    assert measured_for(translucent, "colour_separability_deutan") == 11.58
    assert verdict_for(translucent, "colour_separability_deutan") == "pass"
    # The good figure's colours, which pass at full opacity, fall below 3:1 as
    # fills at alpha 0.3.
    assert verdict_for(dp.check_figure(bars_figure(alpha=1)),
                       "geometry_contrast") == "pass"
    assert verdict_for(dp.check_figure(bars_figure(alpha=0.3)),
                       "geometry_contrast") == "fail"


def test_a_colours_own_alpha_is_kept_as_plotnine_draws_it():
    # plotnine keeps a colour's own alpha whether or not the layer sets one.
    # The R twin follows ggplot2, which replaces it with the layer's alpha when
    # there is one, so only the first case is shared.
    own = ("#005b964d", "#d55e004d")
    drawn = mix("#d55e00", "#ffffff", 77 / 255)
    unset = dp.check_figure(bars_figure(alpha=None, values=own))
    assert detail_for(unset, "geometry_contrast") == (
        f"Lowest-contrast colour #d55e00 at alpha 0.30, drawn as {drawn}, "
        "on #ffffff.")
    layered = dp.check_figure(bars_figure(alpha=0.6, values=own))
    assert detail_for(layered, "geometry_contrast") == (
        detail_for(unset, "geometry_contrast"))


def test_the_layer_alpha_reaches_the_parts_each_geom_applies_it_to():
    # As plotnine draws them: area geoms take it on the fill and leave the
    # outline opaque, lines take it on the colour, and points on both. A geom
    # not listed takes it on both, the cautious assumption.
    import plotnine as p9

    from depictr.audit import _alpha_targets

    fill_only = [p9.geom_col(), p9.geom_bar(), p9.geom_histogram(),
                 p9.geom_tile(), p9.geom_rect(), p9.geom_area(),
                 p9.geom_ribbon(), p9.geom_density(), p9.geom_polygon(),
                 p9.geom_violin(), p9.geom_boxplot(), p9.geom_crossbar(),
                 p9.geom_smooth()]
    colour_only = [p9.geom_path(), p9.geom_line(), p9.geom_step(),
                   p9.geom_segment(), p9.geom_errorbar(), p9.geom_linerange(),
                   p9.geom_hline(yintercept=0), p9.geom_vline(xintercept=0),
                   p9.geom_abline()]
    both = [p9.geom_point(), p9.geom_jitter(), p9.geom_pointrange(),
            p9.geom_dotplot()]
    for geom in fill_only:
        assert _alpha_targets(geom) == {"fill"}, type(geom).__name__
    for geom in colour_only:
        assert _alpha_targets(geom) == {"color"}, type(geom).__name__
    for geom in both:
        assert _alpha_targets(geom) == {"color", "fill"}, type(geom).__name__


def test_redundant_encoding_counts_the_channels_that_vary_with_colour():
    colour_only = (ggplot(CY, aes("rainfall", "yield", colour="treatment"))
                   + geom_point() + dp.scale_colour_depictr()
                   + dp.theme_depictr())
    assert measured_for(dp.check_figure(colour_only), "redundant_encoding") == 0

    both = (ggplot(CY, aes("rainfall", "yield", colour="treatment",
                           shape="treatment", linetype="treatment"))
            + geom_line() + geom_point() + dp.scale_colour_depictr()
            + dp.theme_depictr())
    report = dp.check_figure(both)
    assert measured_for(report, "redundant_encoding") == 2
    assert (report.loc[report["check"] == "redundant_encoding",
                       "detail"].item()
            == "Colour is joined by shape and linetype.")


# --- colours split across layers ---------------------------------------------


def test_a_colour_mapped_in_several_layers_one_value_each_is_an_encoding():
    # The same red and green in one layer fail deuteranopia at 0.37. Split one
    # group per layer, every colour row read "not applicable".
    one_layer = dp.check_figure(
        ggplot(CY, aes("rainfall", "yield", colour="treatment"))
        + geom_point()
        + scale_colour_manual(values=["#d62728", "#309208"])
        + dp.theme_depictr())
    split = dp.check_figure(split_figure())
    pd.testing.assert_frame_equal(encoding_rows(split),
                                  encoding_rows(one_layer))
    assert measured_for(split, "colour_separability_deutan") == 0.37
    assert verdict_for(split, "colour_separability_deutan") == "fail"
    # A layer inherits a constant the plot maps, as it inherits any mapping.
    inherited = dp.check_figure(
        ggplot(STANDARD, aes("rainfall", "yield", colour="'a'"))
        + geom_point()
        + geom_point(aes(colour="'b'"), data=ENHANCED)
        + scale_colour_manual(values=["#d62728", "#309208"])
        + dp.theme_depictr())
    pd.testing.assert_frame_equal(encoding_rows(inherited),
                                  encoding_rows(one_layer))
    # Fills split across layers form one set, measured as drawn: translucent
    # bars drawn one per layer read as the one-layer bars do.
    two = pd.DataFrame({"g": ["a", "b"], "y": [1, 2]})
    split_bars = dp.check_figure(
        ggplot(mapping=aes("g", "y"))
        + geom_col(aes(fill="'a'"), data=two.iloc[[0]], alpha=0.3)
        + geom_col(aes(fill="'b'"), data=two.iloc[[1]], alpha=0.3)
        + scale_fill_manual(values=["#005b96", "#d55e00"])
        + dp.theme_depictr())
    pd.testing.assert_frame_equal(
        encoding_rows(split_bars),
        encoding_rows(dp.check_figure(bars_figure(alpha=0.3))))


def test_shape_and_line_type_split_across_layers_count_as_redundant():
    shapes = dp.check_figure(
        ggplot(mapping=aes("rainfall", "yield"))
        + geom_point(aes(colour="'a'", shape="'a'"), data=STANDARD)
        + geom_point(aes(colour="'b'", shape="'b'"), data=ENHANCED)
        + scale_colour_manual(values=["#005b96", "#d55e00"])
        + dp.theme_depictr())
    assert measured_for(shapes, "redundant_encoding") == 1
    assert (detail_for(shapes, "redundant_encoding")
            == "Colour is joined by shape.")
    lines = dp.check_figure(
        ggplot(mapping=aes("rainfall", "yield"))
        + geom_line(aes(colour="'a'", linetype="'a'"), data=STANDARD)
        + geom_line(aes(colour="'b'", linetype="'b'"), data=ENHANCED)
        + scale_colour_manual(values=["#005b96", "#d55e00"])
        + dp.theme_depictr())
    assert (detail_for(lines, "redundant_encoding")
            == "Colour is joined by linetype.")
    # Points that vary in shape beside lines that vary in line type count both
    # channels. Each layer varies in one, and the count used to stop at one.
    points_and_lines = dp.check_figure(
        ggplot(CY, aes("rainfall", "yield", colour="treatment"))
        + geom_point(aes(shape="treatment"))
        + geom_line(aes(linetype="treatment"))
        + scale_colour_manual(values=["#005b96", "#d55e00"])
        + dp.theme_depictr())
    assert measured_for(points_and_lines, "redundant_encoding") == 2
    assert (detail_for(points_and_lines, "redundant_encoding")
            == "Colour is joined by shape and linetype.")
    # A shape set on each layer is a constant, so it is not counted.
    set_shapes = dp.check_figure(
        ggplot(mapping=aes("rainfall", "yield"))
        + geom_point(aes(colour="'a'"), data=STANDARD, shape="o")
        + geom_point(aes(colour="'b'"), data=ENHANCED, shape="^")
        + scale_colour_manual(values=["#005b96", "#d55e00"])
        + dp.theme_depictr())
    assert measured_for(set_shapes, "redundant_encoding") == 0
    assert (detail_for(set_shapes, "redundant_encoding")
            == "Colour alone distinguishes the groups.")


# --- text drawn inside the panel ---------------------------------------------


def test_text_inside_the_panel_is_held_to_the_size_floor_in_a_row_of_its_own():
    # A 5.7 pt note fails the 6 pt floor. The theme's text keeps the row it
    # always had, still 8.80 pt.
    report = dp.check_figure(annotated_figure())
    assert measured_for(report, "annotation_text_size") == 5.7
    assert verdict_for(report, "annotation_text_size") == "fail"
    assert threshold_for(report, "annotation_text_size") == 6
    assert measured_for(report, "text_size") == 8.8
    assert verdict_for(report, "text_size") == "pass"
    # The note scales with the printed width as the theme's text does.
    column = dp.check_figure(annotated_figure(), width_cm=8.89)
    assert measured_for(column, "annotation_text_size") == 2.85
    assert detail_for(column, "annotation_text_size") == (
        "Smallest in-panel text 5.70 pt, drawn at 17.78 cm and printed at "
        "8.89 cm.")
    narrow = dp.check_figure(annotated_figure(), width_cm=8.89,
                             render_width_cm=8.89)
    assert measured_for(narrow, "annotation_text_size") == 5.7
    # The smallest text across all the panels of a facetted figure: the note is
    # drawn in both, and a smaller label in the first alone.
    facetted = dp.check_figure(
        good_figure() + facet_wrap("treatment")
        + annotate("text", x=800, y=5, label="note", size=7)
        + geom_text(aes("rainfall", "yield", label="'one'"),
                    data=STANDARD.head(1), size=6.5, inherit_aes=False))
    assert measured_for(facetted, "annotation_text_size") == 6.5
    # A label's text is measured as text is: the box around it is not.
    labelled = dp.check_figure(
        good_figure() + annotate("label", x=800, y=5, label="boxed", size=5))
    assert measured_for(labelled, "annotation_text_size") == 5


def test_a_blank_label_draws_nothing_so_its_size_is_not_measured():
    # One label is empty and set smaller than the other. matplotlib draws no
    # text for it, and the R twin leaves it out of its measurement too.
    labels = pd.DataFrame({"x": [1, 2], "y": [1, 2], "label": ["", "b"],
                           "size": [4, 9]})
    report = dp.check_figure(
        ggplot(labels, aes("x", "y", label="label", size="size"))
        + geom_text() + scale_size_identity() + dp.theme_depictr())
    assert measured_for(report, "annotation_text_size") == 9


def test_a_figure_whose_only_text_is_inside_the_panel_says_so():
    # The theme draws no text, so the two rows that measure it abstain, and
    # they no longer claim that the figure draws no text at all.
    p = (ggplot(CY.head(3), aes("rainfall", "yield", label="field"))
         + geom_text(size=8) + theme_void())
    report = dp.check_figure(p)
    for check in ("text_size", "text_contrast"):
        assert verdict_for(report, check) == "not applicable"
        assert (detail_for(report, check)
                == "The figure draws no text outside the panel.")
    assert measured_for(report, "annotation_text_size") == 8


# --- what the audit declines to measure --------------------------------------


def test_a_continuous_scale_is_not_treated_as_a_set_of_category_codes():
    # A gradient's neighbouring colours are meant to be close, so measuring the
    # distance between them would only ever report that a gradient is a
    # gradient.
    heatmap = dp.correlation_heatmap(WB)
    report = dp.check_figure(heatmap)
    for check in COLOUR_CHECKS:
        assert verdict_for(report, check) == "not applicable"
        assert np.isnan(measured_for(report, check))
        # The detail says why. It used to say that nothing is distinguished
        # by colour, which is false of a heatmap.
        assert detail_for(report, check) == (
            "Colour is a continuous scale, which these checks do not audit.")
    # The heatmap's cell labels are a text layer, so they neither count as
    # encoding colours nor drag the text-size measurement down to their size.
    assert measured_for(report, "text_size") == 8.8
    # They are measured in the row for text inside the panel instead. The R
    # twin draws them at another size, so the value is read from this plot's
    # own text layer and is not pinned across the twins.
    labels = next(layer for layer in heatmap.layers
                  if isinstance(layer.geom, geom_text))
    assert (measured_for(report, "annotation_text_size")
            == round(float(labels.geom.aes_params["size"]), 2))


def test_a_continuous_scale_counts_when_the_figure_draws_from_it():
    def continuous_detail(plot):
        report = dp.check_figure(plot)
        return set(report.loc[report["check"].isin(COLOUR_CHECKS), "detail"])

    # A fill that a stat computes comes from a continuous scale, though no
    # layer maps it by name. The R suite pins the same two cases.
    assert continuous_detail(
        ggplot(CY, aes("rainfall", "yield")) + geom_bin_2d(bins=10)
        + dp.theme_depictr()) == {
        "Colour is a continuous scale, which these checks do not audit."}
    # A continuous scale added to a plot that maps nothing to it draws nothing.
    # plotnine warns that such a scale has no legend to draw, a warning this
    # case expects and does not test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unused = continuous_detail(
            ggplot(CY, aes("rainfall", "yield")) + geom_point()
            + scale_colour_gradient() + dp.theme_depictr())
    assert unused == {
        "No two encoding colours: nothing is distinguished by colour."}


def test_a_figure_with_no_colour_encoding_abstains_rather_than_passing():
    p = (ggplot(CY, aes("rainfall", "yield")) + geom_point()
         + dp.theme_depictr())
    report = dp.check_figure(p)
    abstained = report[report["check"].isin(COLOUR_CHECKS)]
    assert set(abstained["verdict"]) == {"not applicable"}
    assert abstained["measured"].isna().all()
    assert set(abstained["detail"]) == {
        "No two encoding colours: nothing is distinguished by colour."}
    # Text is still there to measure.
    assert verdict_for(report, "text_size") == "pass"


def test_colours_set_layer_by_layer_as_constants_abstain():
    def abstains(plot):
        rows = encoding_rows(dp.check_figure(plot))
        assert set(rows["verdict"]) == {"not applicable"}
        assert set(rows["detail"]) == {
            "No two encoding colours: nothing is distinguished by colour."}

    # A constant set on each layer is not a mapping. The audit cannot tell it
    # from decoration, so it leaves it out, as the docstring says.
    abstains(ggplot(mapping=aes("rainfall", "yield"))
             + geom_point(data=STANDARD, color="#d62728")
             + geom_point(data=ENHANCED, color="#309208")
             + dp.theme_depictr())
    # Nor is a colour passed through an identity scale.
    abstains(ggplot(mapping=aes("rainfall", "yield"))
             + geom_point(aes(colour="'#d62728'"), data=STANDARD)
             + geom_point(aes(colour="'#309208'"), data=ENHANCED)
             + scale_colour_identity()
             + dp.theme_depictr())
    # One mapped value is one colour, which is not a pair.
    abstains(ggplot(CY, aes("rainfall", "yield", colour="'a'"))
             + geom_point() + dp.theme_depictr())
    # A reference line does not inherit the plot's mapping, so its black does
    # not join the mapped colour.
    abstains(ggplot(CY, aes("rainfall", "yield", colour="'a'"))
             + geom_point() + geom_hline(yintercept=5) + dp.theme_depictr())


def test_a_figure_that_draws_no_text_says_so():
    p = (ggplot(CY, aes("rainfall", "yield")) + geom_point() + theme_void()
         + theme(legend_position="none"))
    report = dp.check_figure(p)
    assert verdict_for(report, "text_size") == "not applicable"
    assert verdict_for(report, "text_contrast") == "not applicable"
    assert (report.loc[report["check"] == "text_size", "detail"].item()
            == "The figure draws no text.")
    assert verdict_for(report, "annotation_text_size") == "not applicable"
    assert (detail_for(report, "annotation_text_size")
            == "The figure draws no text inside the panel.")


# --- degenerate inputs -------------------------------------------------------


def test_a_single_row_and_a_zero_variance_grouping_still_audit():
    one = CY.head(1)
    report = dp.check_figure(ggplot(one, aes("rainfall", "yield"))
                             + geom_point() + dp.theme_depictr())
    assert len(report) == 10
    assert verdict_for(report, "colour_separability") == "not applicable"

    # One group means one colour, which is not a pair, so the colour checks
    # abstain rather than declaring a lone colour infinitely safe.
    constant = CY.assign(treatment="standard")
    report = dp.check_figure(
        ggplot(constant, aes("rainfall", "yield", colour="treatment"))
        + geom_point() + dp.scale_colour_depictr() + dp.theme_depictr())
    assert verdict_for(report, "colour_separability") == "not applicable"
    assert verdict_for(report, "geometry_contrast") == "not applicable"


def test_an_empty_plot_audits_without_error():
    assert len(dp.check_figure(ggplot() + dp.theme_depictr())) == 10


def test_a_measurement_exactly_on_the_threshold_passes():
    # The boundary is inclusive: a check passes when measured >= threshold.
    p = good_figure()
    measured = measured_for(dp.check_figure(p), "text_size")
    assert verdict_for(dp.check_figure(p, min_text_pt=measured),
                       "text_size") == "pass"
    assert verdict_for(dp.check_figure(p, min_text_pt=measured + 0.01),
                       "text_size") == "fail"


# --- composites --------------------------------------------------------------


def test_a_composite_is_audited_panel_by_panel_in_reading_order():
    # Pinned so the two engines cannot drift apart unnoticed: the R suite
    # audits the same two panels and pins the same vector. Each panel reports
    # what its figure reports alone, under a leading panel column.
    report = dp.check_figure(dp.arrange_plots(good_figure(), bad_figure()))
    assert report.columns.tolist() == [
        "panel", "check", "measured", "threshold", "verdict", "detail"]
    assert len(report) == 20
    assert report["panel"].tolist() == [1] * 10 + [2] * 10
    assert report["check"].tolist() == CHECKS * 2
    np.testing.assert_array_equal(
        report["measured"],
        [111.87, 89.87, 107.07, 98.5, 16.92, 8.8, 8.45, 3.87, 1, np.nan,
         116.25, 35.04, 0.37, 119.38, 6.32, 4, 8.45, 4.01, 0, np.nan])
    assert report.index.tolist() == list(range(20))
    second = (report[report["panel"] == 2].drop(columns="panel")
              .reset_index(drop=True))
    pd.testing.assert_frame_equal(second, dp.check_figure(bad_figure()))


def test_nested_compositions_are_walked_depth_first_and_spacers_are_not_panels():
    nested = dp.check_figure((good_figure() | bad_figure()) / good_figure())
    assert nested["panel"].unique().tolist() == [1, 2, 3]
    assert nested.loc[nested["check"] == "text_size",
                      "measured"].tolist() == [8.8, 4, 8.8]

    # A spacer draws no figure of its own, so the panels on either side of it
    # are numbered 1 and 2, as in the R twin.
    spaced = dp.check_figure(good_figure() | plot_spacer() | bad_figure())
    assert spaced["panel"].unique().tolist() == [1, 2]
    assert spaced.loc[spaced["check"] == "text_size",
                      "measured"].tolist() == [8.8, 4]

    # With nothing but spacers, there is no panel to audit.
    empty = dp.check_figure(plot_spacer() | plot_spacer())
    assert empty.columns.tolist() == spaced.columns.tolist()
    assert len(empty) == 0


def test_the_package_own_composites_are_audited_panel_by_panel():
    smf = pytest.importorskip("statsmodels.formula.api")
    # "yield" is a Python keyword, so the formula needs Q("yield").
    fit = smf.ols('Q("yield") ~ rainfall + fertiliser + soil_ph', CY).fit()
    diagnostics = dp.check_figure(dp.residual_diagnostics_plot(fit))
    assert diagnostics["panel"].unique().tolist() == [1, 2, 3, 4]
    assert len(diagnostics) == 40
    report = dp.check_figure(dp.model_report(fit))
    assert report["panel"].unique().tolist() == [1, 2, 3]
    assert set(report.loc[report["check"] == "text_size", "verdict"]) == {
        "pass"}


def test_check_figure_leaves_a_composition_untouched():
    # As for a single plot, each panel is drawn from a copy, so no layer of the
    # caller's composition is built.
    composite = dp.arrange_plots(good_figure(), bad_figure())
    dp.check_figure(composite)
    assert not any(hasattr(layer, "data")
                   for panel in composite for layer in panel.layers)


# --- refusals ----------------------------------------------------------------


def test_check_figure_refuses_what_it_cannot_audit():
    message = ("`plot` must be a plot object, as returned by any depictr "
               "plotting function.")
    with pytest.raises(TypeError, match=r"must be a plot object"):
        dp.check_figure(1)
    with pytest.raises(TypeError) as excinfo:
        dp.check_figure("a plot")
    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    "argument",
    ["width_cm", "render_width_cm", "min_delta_e", "min_text_pt"])
@pytest.mark.parametrize("value", [0, -1, "wide", None, float("nan")])
def test_check_figure_refuses_non_positive_settings(argument, value):
    with pytest.raises(ValueError) as excinfo:
        dp.check_figure(good_figure(), **{argument: value})
    assert str(excinfo.value) == (
        f"`{argument}` must be a single positive number.")


# --- the audit sees what the user added --------------------------------------


def test_check_figure_reads_the_extended_plot():
    # The whole point of introspecting the build: a user's replacement scale is
    # what the figure ships with, so it is what gets audited. The dumbbell's
    # points are opaque, so the verdicts turn on the scale alone.
    base = dp.dumbbell_plot(LD, "participant", "RT", "condition")
    assert verdict_for(dp.check_figure(base),
                       "colour_separability_deutan") == "pass"
    # Replacing a scale that is already there is the ordinary way a user
    # overrides depictr's colours, and plotnine says so every time; the warning
    # is not the subject of this test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken = base + scale_colour_manual(values=["#d62728", "#309208"])
    assert verdict_for(dp.check_figure(broken),
                       "colour_separability_deutan") == "fail"


def test_check_figure_leaves_the_plot_it_was_given_untouched():
    # Drawing a plotnine plot builds its layers in place, which is why the audit
    # works on a copy. An unbuilt layer has no `data` at all, so its absence
    # afterwards is proof the caller's plot was not built behind their back.
    p = good_figure()
    assert not any(hasattr(layer, "data") for layer in p.layers)
    dp.check_figure(p)
    assert not any(hasattr(layer, "data") for layer in p.layers)
    # And the audit still reads the same figure on a second call: every check
    # passes but the one for text inside the panel, where it draws none.
    again = dp.check_figure(p)
    assert set(again.loc[again["check"] != "annotation_text_size",
                         "verdict"]) == {"pass"}
    assert verdict_for(again, "annotation_text_size") == "not applicable"
