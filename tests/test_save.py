"""Tests for save_plot(): files that pass journals' technical checks."""

import re
import warnings

import matplotlib
import pandas as pd
import pytest
from PIL import Image
from plotnine import (
    aes,
    geom_point,
    geom_smooth,
    geom_tile,
    ggplot,
    scale_colour_manual,
    theme_matplotlib,
    theme_void,
)

import depictr as dp
from depictr import audit

CY = dp.crop_yield()


def translucent_figure():
    """The package's own scatter: points at alpha 0.6 and a band at 0.2."""
    return dp.scatter_trend(CY, "fertiliser", "yield")


def opaque_figure():
    """Opaque points, and a smoothing layer whose alpha has no band to apply to."""
    return (ggplot(CY, aes("fertiliser", "yield"))
            + geom_point()
            + geom_smooth(method="lm", se=False)
            + dp.theme_depictr())


def declared_font_types(eps):
    """The FontType that each font embedded in a PostScript file is defined as.

    A font's first FontType entry declares its type. matplotlib before 3.10
    writes Type 42 through ttconv, which follows that entry with a fallback for
    an interpreter that cannot rasterise Type 42: guarded by
    ``42 /FontType resourcestatus``, it sets FontType 3 and draws through Apple's
    TrueType rasteriser. Every PostScript 3 interpreter supports Type 42 and
    skips the fallback, so the bytes "/FontType 3" alone do not make a Type 3
    font.
    """
    definitions = eps.split(b"definefont")[:-1]
    return [int(re.search(rb"/FontType (\d+)", font)[1]) for font in definitions]


def test_pdf_embeds_truetype_fonts(tmp_path):
    path = tmp_path / "figure.pdf"
    dp.save_plot(translucent_figure(), str(path))
    data = path.read_bytes()
    # matplotlib's default writes Type 3 fonts, which Nature refuses. Type 42
    # embeds the TrueType font program as a /FontFile2 stream.
    assert b"/FontFile2" in data
    assert b"/Type3" not in data


def test_eps_embeds_type42_fonts(tmp_path):
    path = tmp_path / "figure.eps"
    dp.save_plot(opaque_figure(), str(path))
    data = path.read_bytes()
    assert data.startswith(b"%!PS-Adobe-3.0 EPSF-3.0")
    types = declared_font_types(data)
    assert types and set(types) == {42}


def test_a_theme_with_its_own_font_type_still_embeds_truetype(tmp_path):
    # theme_matplotlib() copies the session's rcParams, Type 3 fonts included,
    # and plotnine saves inside the theme's rcParams.
    plot = opaque_figure() + theme_matplotlib()
    pdf = tmp_path / "figure.pdf"
    dp.save_plot(plot, str(pdf))
    assert b"/FontFile2" in pdf.read_bytes()
    assert b"/Type3" not in pdf.read_bytes()
    eps = tmp_path / "figure.eps"
    dp.save_plot(plot, str(eps))
    types = declared_font_types(eps.read_bytes())
    assert types and set(types) == {42}


def test_eps_warns_about_translucent_layers(tmp_path):
    path = tmp_path / "figure.eps"
    with pytest.warns(UserWarning, match="2 translucent layers opaque"):
        dp.save_plot(translucent_figure(), str(path))
    assert path.exists()


def test_opaque_eps_does_not_warn(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dp.save_plot(opaque_figure(), str(tmp_path / "figure.eps"))


def test_eps_warns_about_a_translucent_panel_in_a_composite(tmp_path):
    composite = dp.arrange_plots(opaque_figure(), translucent_figure(), ncol=2)
    with pytest.warns(UserWarning, match="translucent"):
        dp.save_plot(composite, str(tmp_path / "figure.eps"))


def test_tiff_is_lzw_compressed_rgb_at_the_requested_resolution(tmp_path):
    path = tmp_path / "figure.tiff"
    dp.save_plot(translucent_figure(), str(path))
    with Image.open(path) as image:
        # PLOS: "LZW compression is required", "RGB (8 bit/channel) or
        # grayscale only", "No alpha channels".
        assert image.mode == "RGB"
        assert image.info["compression"] == "tiff_lzw"
        assert image.info["dpi"] == pytest.approx((300, 300))
        assert image.size == (2100, 1350)
    # Uncompressed RGBA, the same figure was 10.8 MiB.
    assert path.stat().st_size < 1_000_000


def test_tiff_composites_translucent_pixels_onto_white(tmp_path):
    # A black tile at alpha 0.25 on a transparent figure prints as a light grey,
    # 0.75 * 255 = 191. Dropping the alpha channel alone would print it black.
    tile = (ggplot(pd.DataFrame({"x": [0], "y": [0]}), aes("x", "y"))
            + geom_tile(fill="black", alpha=0.25)
            + theme_void())
    path = tmp_path / "figure.tif"
    dp.save_plot(tile, str(path), width=2, height=2, dpi=72, transparent=True)
    with Image.open(path) as image:
        assert image.mode == "RGB"
        centre = image.getpixel((image.width // 2, image.height // 2))
    assert centre == pytest.approx((191, 191, 191), abs=2)


def test_pil_kwargs_leave_the_tiff_to_matplotlib(tmp_path):
    path = tmp_path / "figure.tiff"
    dp.save_plot(opaque_figure(), str(path), dpi=72,
                 pil_kwargs={"compression": "tiff_adobe_deflate"})
    with Image.open(path) as image:
        assert image.info["compression"] == "tiff_adobe_deflate"


def test_missing_directories_are_created(tmp_path):
    path = tmp_path / "figures" / "chapter-2" / "figure.png"
    returned = dp.save_plot(opaque_figure(), str(path), dpi=72)
    assert returned == str(path)
    assert path.exists()


def test_font_settings_do_not_leak_into_the_session(tmp_path):
    before = (matplotlib.rcParams["pdf.fonttype"], matplotlib.rcParams["ps.fonttype"])
    dp.save_plot(opaque_figure(), str(tmp_path / "figure.pdf"))
    after = (matplotlib.rcParams["pdf.fonttype"], matplotlib.rcParams["ps.fonttype"])
    assert after == before


@pytest.mark.parametrize("extension", ["png", "tiff"])
def test_a_composition_is_saved_at_the_size_asked_for(tmp_path, extension):
    # plotnine saves a composition at the figure size of its last plot's theme
    # and ignores the width and height it is given, so this one used to be
    # written at 6.4 by 4.8 inches whatever was asked for.
    composite = dp.arrange_plots(opaque_figure(), opaque_figure())
    path = dp.save_plot(composite, str(tmp_path / f"figure.{extension}"),
                        width=3.5, height=2, dpi=100)
    with Image.open(path) as image:
        assert image.size == (350, 200)
    # The caller's composition keeps its own theme.
    assert composite.last_plot.theme.getp("figure_size") != (3.5, 2)


# --- save_plot(check=True): the audit at the width the figure is saved at ----


# The reference figures of the check_figure() tests. The good one clears every
# check that applies, and the bad one fails deuteranopia, the text floor and
# redundant encoding. The R suite saves the same two.
def audited_good():
    return (ggplot(CY, aes("rainfall", "yield", colour="treatment",
                           shape="treatment"))
            + geom_point()
            + scale_colour_manual(values=["#005b96", "#d55e00"])
            + dp.theme_depictr())


def audited_bad():
    return (ggplot(CY, aes("rainfall", "yield", colour="treatment"))
            + geom_point()
            + scale_colour_manual(values=["#d62728", "#309208"])
            + dp.theme_depictr(base_size=5))


def test_check_audits_at_the_saved_width_given_as_both_widths(tmp_path,
                                                              monkeypatch):
    seen = {}
    real = audit.check_figure

    def spy(plot, **kwargs):
        seen["kwargs"] = kwargs
        seen["report"] = real(plot, **kwargs)
        return seen["report"]

    monkeypatch.setattr(audit, "check_figure", spy)
    path = tmp_path / "column.png"
    # Drawn at 8.9 cm and printed at 8.9 cm, the text is the size it was drawn
    # at, so the good figure fails nothing and save_plot() says nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        returned = dp.save_plot(audited_good(), str(path), width=8.9, height=6,
                                units="cm", dpi=72, check=True)
    assert returned == str(path)
    assert path.exists()
    assert seen["kwargs"] == {"width_cm": 8.9, "render_width_cm": 8.9}
    report = seen["report"]
    assert report.loc[report["check"] == "text_size", "measured"].item() == 8.8
    assert report.loc[report["check"] == "text_size", "detail"].item() == (
        "Smallest text 8.80 pt, drawn at 8.90 cm and printed at 8.90 cm.")


def test_check_warns_with_the_checks_the_figure_fails(tmp_path):
    # A row with nothing to measure, here the text inside the panel, is not a
    # failure.
    expected = (
        "The figure fails 3 checks at the 8.90 cm it is saved at: "
        "colour_separability_deutan 0.11 (threshold 5.00), text_size 4.00 "
        "(threshold 6.00) and redundant_encoding 0.00 (threshold 1.00). "
        "Pass that width to check_figure() as both width_cm and "
        "render_width_cm for the full report.")
    with pytest.warns(UserWarning) as record:
        dp.save_plot(audited_bad(), str(tmp_path / "column.png"), width=8.9,
                     height=6, units="cm", dpi=72, check=True)
    assert [str(w.message) for w in record] == [expected]
    # In a composition, each failure names its panel.
    with pytest.warns(UserWarning, match=r"fails 3 checks .*: panel 2 "
                      r"colour_separability_deutan 0\.11"):
        dp.save_plot(dp.arrange_plots(audited_good(), audited_bad()),
                     str(tmp_path / "both.png"), width=8.9, height=6,
                     units="cm", dpi=72, check=True)


def test_check_takes_the_saved_width_in_any_unit():
    from depictr.compose import _saved_width_cm

    assert _saved_width_cm(7, "in") == pytest.approx(17.78)
    assert _saved_width_cm(8.9, "cm") == pytest.approx(8.9)
    assert _saved_width_cm(89, "mm") == pytest.approx(8.9)


@pytest.mark.parametrize("arguments", [
    {"check": None}, {"check": "yes"}, {"check": 1}])
def test_check_must_be_true_or_false(tmp_path, arguments):
    path = tmp_path / "never.png"
    with pytest.raises(TypeError, match=r"^`check` must be True or False\.$"):
        dp.save_plot(audited_good(), str(path), **arguments)
    assert not path.exists()


@pytest.mark.parametrize("arguments", [
    {"width": None, "height": None}, {"units": "pt"}, {"width": -1}])
def test_check_refuses_a_width_it_cannot_read_before_writing(tmp_path,
                                                             arguments):
    path = tmp_path / "never.png"
    with pytest.raises(ValueError, match=r"^`check=True` needs the width"):
        dp.save_plot(audited_good(), str(path), check=True, **arguments)
    assert not path.exists()
    # The audit refuses what it cannot read, as check_figure() does.
    with pytest.raises(TypeError, match=r"must be a plot object"):
        dp.save_plot("a plot", str(path), check=True)
    assert not path.exists()
