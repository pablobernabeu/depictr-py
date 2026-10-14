"""simulate_figure(): a finished figure as a reader with a CVD sees it."""

import ast
import copy
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from plotnine import aes, geom_point, geom_rect, ggplot, theme, theme_void

import depictr as dp

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _block(fill):
    """A small figure whose middle is one solid rectangle of ``fill``."""
    frame = pd.DataFrame({"xmin": [0], "xmax": [1], "ymin": [0], "ymax": [1]})
    return (ggplot(frame)
            + geom_rect(aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                        fill=fill)
            + theme_void()
            + theme(figure_size=(2, 2)))


def _grouped():
    cy = dp.crop_yield()
    return (ggplot(cy, aes("rainfall", "yield", colour="treatment"))
            + geom_point()
            + dp.scale_colour_depictr()
            + dp.theme_depictr()
            + theme(figure_size=(4, 3)))


def _rasterise(plot, dpi):
    """The figure's pixels as matplotlib's Agg renderer draws them."""
    figure = copy.deepcopy(plot).draw(show=False)
    figure.set_dpi(dpi)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba()) / 255
    plt.close(figure)
    return pixels


def test_a_solid_region_takes_the_colour_simulate_cvd_gives_it():
    # The pixels go through the transform simulate_cvd() applies to a colour,
    # so the middle of a solid rectangle matches it to within the rounding of
    # an eight-bit channel.
    fill = "#e69f00"
    plot = _block(fill)
    for deficiency in dp.cvd.DEFICIENCIES:
        for severity in (1.0, 0.5, 0.25):
            seen = dp.simulate_figure(plot, deficiency, severity, dpi=40)
            height, width = seen.shape[:2]
            middle = seen[height // 2, width // 2]
            expected = mcolors.to_rgb(dp.simulate_cvd([fill], deficiency,
                                                      severity)[0])
            np.testing.assert_allclose(middle[:3], expected, atol=1 / 255,
                                       err_msg=f"{deficiency} {severity}")
            assert middle[3] == 1


def test_the_figure_is_returned_as_rgba_in_the_unit_interval():
    seen = dp.simulate_figure(_grouped(), "deutan", dpi=50)
    # Four inches by three at 50 dpi.
    assert seen.shape == (150, 200, 4)
    assert seen.dtype == np.float64
    assert seen.min() >= 0 and seen.max() <= 1
    # The orange of the second group is not left as it was drawn.
    assert not np.array_equal(seen, _rasterise(_grouped(), 50))


def test_severity_zero_is_the_figure_as_drawn():
    plot = _grouped()
    for deficiency in dp.cvd.DEFICIENCIES:
        np.testing.assert_array_equal(
            dp.simulate_figure(plot, deficiency, severity=0, dpi=45),
            _rasterise(plot, 45))


def test_the_alpha_channel_is_kept():
    from depictr.cvd import _simulate_pixels

    rng = np.random.default_rng(1)
    pixels = rng.random((4, 5, 4))
    pixels[..., 3] = np.linspace(0, 1, 20).reshape(4, 5)
    seen = _simulate_pixels(pixels, "protan", 0.7)
    np.testing.assert_array_equal(seen[..., 3], pixels[..., 3])
    assert not np.array_equal(seen[..., :3], pixels[..., :3])
    # The caller's array is not written to.
    assert np.array_equal(pixels[..., 3], np.linspace(0, 1, 20).reshape(4, 5))
    # A whole figure keeps the alpha it was drawn with.
    plot = _grouped()
    np.testing.assert_array_equal(
        dp.simulate_figure(plot, "tritan", dpi=30)[..., 3],
        _rasterise(plot, 30)[..., 3])


def test_the_callers_plot_is_left_as_it_was():
    # Drawing a plotnine plot builds its layers and attaches a figure in place,
    # so the plot is drawn from a copy.
    plot = _grouped()
    before = set(vars(plot))
    dp.simulate_figure(plot, "deutan", dpi=30)
    assert set(vars(plot)) == before


def test_a_plot_drawn_before_is_refused():
    # plotnine's copy of a plot that has been drawn shares its figure and its
    # built state. Drawing the copy changed the resolution of the caller's
    # figure and came out with the panel missing, so the plot is refused.
    message = ("`plot` has already been drawn, and plotnine does not draw a "
               "plot correctly a second time. Pass the plot before it is "
               "drawn, or build it again.")
    plot = _grouped()
    figure = plot.draw(show=False)
    dpi = figure.dpi
    with pytest.raises(ValueError) as excinfo:
        dp.simulate_figure(plot, "deutan", dpi=40)
    assert str(excinfo.value) == message
    assert figure.dpi == dpi
    # So is a composition that holds a plot drawn before.
    with pytest.raises(ValueError) as excinfo:
        dp.simulate_figure(dp.arrange_plots(_block("#56b4e9"), plot, ncol=2),
                           "deutan", dpi=40)
    assert str(excinfo.value) == message
    plt.close(figure)


def test_a_composition_is_simulated_whole():
    left = _grouped()
    right = _block("#56b4e9")
    composed = dp.arrange_plots(left, right, ncol=2)
    before = [set(vars(item)) for item in composed.items]
    seen = dp.simulate_figure(composed, "deutan", dpi=30)
    plain = dp.simulate_figure(composed, "deutan", severity=0, dpi=30)
    assert seen.shape == plain.shape
    assert seen.shape[2] == 4
    assert not np.array_equal(seen, plain)
    # The panels were drawn from copies. (plotnine's own deep copy of a
    # composition leaves a note of the copy on the original.)
    assert [set(vars(item)) for item in composed.items] == before


def test_simulate_figure_refuses_what_it_cannot_draw():
    plot = _block("#005b96")
    with pytest.raises(ValueError) as excinfo:
        dp.simulate_figure(plot, "normal")
    assert str(excinfo.value) == (
        "`deficiency` must be one of 'protan', 'deutan' or 'tritan'.")
    with pytest.raises(ValueError) as excinfo:
        dp.simulate_figure(plot, "deutan", severity=1.5)
    assert str(excinfo.value) == "`severity` must lie in [0, 1]."
    for dpi in (0, -72, float("nan"), True, "150"):
        with pytest.raises(ValueError) as excinfo:
            dp.simulate_figure(plot, "deutan", dpi=dpi)
        assert str(excinfo.value) == "`dpi` must be a positive number."
    figure = plt.figure()
    with pytest.raises(TypeError) as excinfo:
        dp.simulate_figure(figure, "deutan")
    plt.close(figure)
    assert str(excinfo.value) == (
        "`plot` must be a plotnine ggplot or composition.")


def test_simulate_figure_is_public():
    assert "simulate_figure" in dp.__all__
    assert dp.simulate_figure is dp.cvd.simulate_figure


def test_the_app_reaches_no_private_name():
    # The app used to simulate the figure itself, from three private names of
    # depictr.cvd that the package was free to change under it.
    source = APP.read_text(encoding="utf-8")
    private = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("depictr"):
            private += [alias.name for alias in node.names
                        if alias.name.startswith("_")]
            private += [part for part in node.module.split(".")
                        if part.startswith("_")]
        elif (isinstance(node, ast.Attribute) and node.attr.startswith("_")
              and isinstance(node.value, ast.Name) and node.value.id == "dp"):
            private.append(f"dp.{node.attr}")
    assert not private
    assert "dp.simulate_figure(" in source
