"""Tests for check_figure(), the accessibility and honesty audit."""

import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import (
    aes,
    element_blank,
    element_rect,
    geom_line,
    geom_point,
    ggplot,
    scale_colour_manual,
    scale_fill_manual,
    theme,
    theme_void,
)

import depictr as dp
from depictr.cvd import _rgb_to_lab

CY = dp.crop_yield()
WB = dp.wellbeing_survey()
LD = dp.lexical_decision()

CHECKS = [
    "colour_separability", "colour_separability_protan",
    "colour_separability_deutan", "colour_separability_tritan",
    "greyscale_separability", "text_size", "text_contrast",
    "geometry_contrast", "redundant_encoding",
]


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


def wcag_contrast(a, b):
    """The WCAG 2.x ratio recomputed from the published definition alone."""
    import matplotlib.colors as mcolors

    def luminance(colour):
        channels = np.array(mcolors.to_rgb(colour))
        linear = np.where(channels <= 0.03928, channels / 12.92,
                          ((channels + 0.055) / 1.055) ** 2.4)
        return float(linear @ np.array([0.2126, 0.7152, 0.0722]))

    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# --- shape of the result -----------------------------------------------------


def test_report_has_the_standard_table_shape():
    report = dp.check_figure(good_figure())
    assert isinstance(report, pd.DataFrame)
    assert report.columns.tolist() == [
        "check", "measured", "threshold", "verdict", "detail"]
    assert report["check"].tolist() == CHECKS
    assert set(report["verdict"]) <= {"pass", "fail", "not applicable"}
    assert report["detail"].map(bool).all()


# --- the two verdicts the audit must be capable of ---------------------------


def test_a_deliberately_good_figure_passes_every_check():
    report = dp.check_figure(good_figure())
    assert set(report["verdict"]) == {"pass"}, report.to_string()


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
    # is constructible in both, and the R suite pins the same three vectors.
    assert dp.check_figure(good_figure())["measured"].tolist() == [
        111.87, 89.87, 107.07, 98.5, 16.92, 8.8, 8.45, 3.87, 1]
    assert dp.check_figure(bad_figure())["measured"].tolist() == [
        116.25, 35.04, 0.37, 119.38, 6.32, 4, 8.45, 4.01, 0]

    eight = pd.DataFrame({"g": list("abcdefgh"), "x": range(8), "y": range(8)})
    p = (ggplot(eight, aes("x", "y", colour="g")) + geom_point()
         + dp.scale_colour_depictr() + dp.theme_depictr())
    report = dp.check_figure(p)
    assert report["measured"].tolist() == [
        33.43, 18.15, 7.4, 16.18, 0.79, 8.8, 8.45, 1.32, 0]
    # The detail strings are part of the contract too, since they carry the
    # colours the numbers came from.
    assert report["detail"].tolist() == [
        "Closest pair #d55e00 and #e69f00 of 8 encoding colours.",
        "Closest pair #999999 and #cc79a7 of 8 encoding colours.",
        "Closest pair #999999 and #cc79a7 of 8 encoding colours.",
        "Closest pair #009e73 and #56b4e9 of 8 encoding colours.",
        "Closest pair #56b4e9 and #e69f00 in CIE lightness.",
        "Smallest text 8.80 pt, drawn at 17.78 cm and printed at 17.78 cm.",
        "Lowest-contrast text #4d4d4d on #ffffff.",
        "Lowest-contrast colour #f0e442 on #ffffff.",
        "Colour alone distinguishes the groups.",
    ]


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
            == "Lowest-contrast text #4d4d4d on #eeeeee.")
    # A blank plot background paints nothing, so white stands in for the paper.
    blanked = dp.check_figure(good_figure()
                              + theme(plot_background=element_blank()))
    assert (measured_for(blanked, "text_contrast")
            == round(wcag_contrast("#4d4d4d", "#ffffff"), 2))


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


# --- what the audit declines to measure --------------------------------------


def test_a_continuous_scale_is_not_treated_as_a_set_of_category_codes():
    # A gradient's neighbouring colours are meant to be close, so measuring the
    # distance between them would only ever report that a gradient is a
    # gradient.
    report = dp.check_figure(dp.correlation_heatmap(WB))
    for check in ("colour_separability", "greyscale_separability",
                  "geometry_contrast", "redundant_encoding"):
        assert verdict_for(report, check) == "not applicable"
        assert np.isnan(measured_for(report, check))
    # The heatmap's cell labels are a text layer, so they neither count as
    # encoding colours nor drag the text-size measurement down to their size.
    assert measured_for(report, "text_size") == 8.8


def test_a_figure_with_no_colour_encoding_abstains_rather_than_passing():
    p = (ggplot(CY, aes("rainfall", "yield")) + geom_point()
         + dp.theme_depictr())
    report = dp.check_figure(p)
    colour_checks = CHECKS[:5] + ["geometry_contrast", "redundant_encoding"]
    abstained = report[report["check"].isin(colour_checks)]
    assert set(abstained["verdict"]) == {"not applicable"}
    assert abstained["measured"].isna().all()
    assert set(abstained["detail"]) == {
        "No two encoding colours: nothing is distinguished by colour."}
    # Text is still there to measure.
    assert verdict_for(report, "text_size") == "pass"


def test_a_figure_that_draws_no_text_says_so():
    p = (ggplot(CY, aes("rainfall", "yield")) + geom_point() + theme_void()
         + theme(legend_position="none"))
    report = dp.check_figure(p)
    assert verdict_for(report, "text_size") == "not applicable"
    assert verdict_for(report, "text_contrast") == "not applicable"
    assert (report.loc[report["check"] == "text_size", "detail"].item()
            == "The figure draws no text.")


# --- degenerate inputs -------------------------------------------------------


def test_a_single_row_and_a_zero_variance_grouping_still_audit():
    one = CY.head(1)
    report = dp.check_figure(ggplot(one, aes("rainfall", "yield"))
                             + geom_point() + dp.theme_depictr())
    assert len(report) == 9
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
    assert len(dp.check_figure(ggplot() + dp.theme_depictr())) == 9


def test_a_measurement_exactly_on_the_threshold_passes():
    # The boundary is inclusive: a check passes when measured >= threshold.
    p = good_figure()
    measured = measured_for(dp.check_figure(p), "text_size")
    assert verdict_for(dp.check_figure(p, min_text_pt=measured),
                       "text_size") == "pass"
    assert verdict_for(dp.check_figure(p, min_text_pt=measured + 0.01),
                       "text_size") == "fail"


# --- refusals ----------------------------------------------------------------


def test_check_figure_refuses_what_it_cannot_audit():
    message = ("`plot` must be a plot object, as returned by any depictr "
               "plotting function.")
    with pytest.raises(TypeError, match=r"must be a plot object"):
        dp.check_figure(1)
    with pytest.raises(TypeError) as excinfo:
        dp.check_figure("a plot")
    assert str(excinfo.value) == message
    # A composite has several panels, each with its own scales, theme and text.
    composite = dp.arrange_plots(ggplot() + dp.theme_depictr(),
                                 ggplot() + dp.theme_depictr())
    with pytest.raises(TypeError) as excinfo:
        dp.check_figure(composite)
    assert str(excinfo.value) == (
        "`plot` is a multi-panel composite. Check each panel on its own.")


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
    # what the figure ships with, so it is what gets audited.
    base = dp.explore_distribution(LD, "RT", group="condition")
    assert verdict_for(dp.check_figure(base), "colour_separability") == "pass"
    # Replacing a scale that is already there is the ordinary way a user
    # overrides depictr's colours, and plotnine says so every time; the warning
    # is not the subject of this test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken = (base + scale_fill_manual(values=["#d62728", "#309208"])
                  + scale_colour_manual(values=["#d62728", "#309208"]))
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
    # And the audit still reads the same figure on a second call.
    assert set(dp.check_figure(p)["verdict"]) == {"pass"}
