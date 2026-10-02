"""An accessibility and honesty audit of a finished figure.

:func:`palette_safety` can promise that the eight colours depictr ships stay
apart under colour-vision deficiency. It knows nothing about the figure in front
of you: how many of those colours it uses, what you replaced them with, how
small the text becomes once the figure is squeezed into a journal column, or
whether the only thing separating two groups is their colour.
:func:`check_figure` reads the built plot and answers those questions with
numbers.

The audit introspects a built plot rather than re-deriving what it thinks the
plot ought to contain, so it also sees whatever was added after depictr handed
the plot back: a replacement scale, a different theme, an extra layer. Drawing
the figure once supplies everything. The layers' data give the colours that
encode groups, and the matplotlib artists give the text that will actually be
drawn, at the point size and in the colour it will be drawn in.

World Wide Web Consortium. (2023). Web content accessibility guidelines (WCAG)
2.2. W3C Recommendation. https://www.w3.org/TR/WCAG22/
"""

from __future__ import annotations

import copy
from typing import NamedTuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import pandas as pd
from matplotlib.font_manager import weight_dict
from matplotlib.offsetbox import AnchoredOffsetbox

from .cvd import (
    DEFICIENCIES,
    _min_pairwise_delta_e,
    _rgb_to_lab,
    _srgb_to_linear,
    simulate_cvd,
)

# WCAG 2.2 contrast floors, used as published rather than tuned so that any
# particular figure passes: 4.5:1 for text at normal size and 3:1 for
# large-scale text (success criterion 1.4.3), and 3:1 for graphical objects
# that carry meaning (1.4.11). WCAG defines large-scale text as at least 18
# point, or 14 point in bold. They are fixed rather than arguments because they
# are somebody else's standard.
_WCAG_TEXT_CONTRAST = 4.5
_WCAG_LARGE_TEXT_CONTRAST = 3.0
_WCAG_LARGE_TEXT_PT = 18.0
_WCAG_LARGE_BOLD_TEXT_PT = 14.0
_WCAG_OBJECT_CONTRAST = 3.0

# What every colour-dependent check says when the figure encodes nothing by
# colour. Shared so the two halves of the sentence cannot drift apart.
_NO_COLOUR_DETAIL = "No two encoding colours: nothing is distinguished by colour."


def check_figure(plot, width_cm: float = 17.78, render_width_cm: float = 17.78,
                 min_delta_e: float = 5, min_text_pt: float = 6) -> pd.DataFrame:
    """Audit a finished figure for accessibility and honesty.

    Checks a figure as it will be submitted, rather than the palette it was
    built from. Every row carries the value it measured next to the threshold it
    was measured against, so a verdict can be argued with rather than merely
    accepted. A check passes when the measured value is at least the threshold.
    A check with nothing to measure, such as colour separability on a figure
    that encodes nothing by colour, reports ``NaN`` and a verdict of
    ``"not applicable"`` instead of a free pass.

    ``colour_separability`` is the smallest CIE76 colour difference (Delta-E)
    between any two of the figure's encoding colours, and the three
    ``colour_separability_*`` rows repeat that measurement after simulating each
    dichromacy at full severity with :func:`depictr.simulate_cvd`. Encoding
    colours are the distinct colour and fill values a layer uses to tell groups
    apart; a continuous colour or fill scale is a smooth ramp rather than a set
    of codes, so it is excluded.

    ``greyscale_separability`` is the smallest difference in CIE lightness
    between those same colours, which is what survives printing in black and
    white.

    ``text_size`` is the smallest point size of any text the figure draws, after
    scaling by ``width_cm / render_width_cm``: a figure drawn seven inches wide
    and printed in an 8.9 cm column has every point size halved. Text drawn
    inside the panel, by a layer such as ``geom_text`` or by an annotation, is
    deliberately left out. It sits on the marks rather than on the background,
    so there is no one background to measure its contrast against, and the two
    engines size a layer's text in different units, which would leave the check
    disagreeing with its R twin on the same figure.

    ``text_contrast`` is the WCAG 2.2 contrast ratio of the text that falls
    furthest short of its own threshold. Each text is measured against the
    background it is drawn on: a facet strip's fill for strip text, the legend
    box's fill for legend text, and the plot background for the rest. A
    translucent fill is measured as the colour it shows over what lies beneath
    it. Text that is large at its printed size, 18 pt or 14 pt in bold, is held
    to 3:1, and other text to 4.5:1, as success criterion 1.4.3 asks. The row's
    ``threshold`` is the one that applies to the text it reports, and its
    ``detail`` names that text's colour, background, printed size and weight.

    ``geometry_contrast`` is the smallest WCAG 2.2 contrast ratio of any
    encoding colour against the panel background, or against the plot
    background where the panel paints none.

    ``redundant_encoding`` counts how many of shape and line type also vary in a
    layer whose colour varies. Zero means the distinction between groups is
    carried by colour alone, the single most common way an otherwise careful
    figure becomes unreadable.

    **A limitation of the default palette.** The eight-colour qualitative
    palette clears the colour-separability checks comfortably and fails
    ``greyscale_separability``: its orange (``#e69f00``) and sky blue
    (``#56b4e9``) differ by only 0.79 in lightness, so they print as the same
    grey. The colourblind-safety guarantee depictr makes is about hue confusion,
    and it was never a claim about greyscale. A figure that may be printed in
    black and white should use fewer groups, or a sequential palette, or add a
    redundant shape or line type; the four leading colours are not safe in
    greyscale either, since the bluish green and the vermillion differ by 3.55.
    The threshold has been left where it is rather than moved to let the
    package's own defaults through.

    Parameters
    ----------
    plot : plotnine.ggplot
        A plot, as returned by any depictr plotting function, including one
        extended afterwards with ``+``. A multi-panel composite is refused: its
        panels have their own scales, themes and text, and one table of numbers
        cannot describe them all. Check each panel on its own.
    width_cm : float
        The width, in centimetres, that the figure will occupy in the finished
        document. Defaults to 17.78 cm, the seven inches
        :func:`depictr.save_plot` draws at, which means no scaling.
    render_width_cm : float
        The width, in centimetres, that the figure is drawn at. Defaults to the
        same 17.78 cm. The ratio of the two widths is the factor every point
        size is multiplied by.
    min_delta_e : float
        The smallest acceptable CIE76 colour difference, used for the colour and
        greyscale separability checks. Defaults to 5, matching
        :func:`depictr.palette_safety`.
    min_text_pt : float
        The smallest acceptable printed text size, in points. Defaults to 6, a
        common publisher floor for figure text.

    Returns
    -------
    pandas.DataFrame
        One row per check, with columns ``check``, ``measured``, ``threshold``,
        ``verdict`` (``"pass"``, ``"fail"`` or ``"not applicable"``) and
        ``detail``, a short note naming what produced the measurement.

    See Also
    --------
    depictr.palette_safety : the palette in the abstract, rather than a figure.

    Examples
    --------
    >>> import depictr as dp
    >>> from plotnine import aes, geom_point, ggplot, scale_colour_manual
    >>> cy = dp.crop_yield()
    >>> good = (ggplot(cy, aes("rainfall", "yield", colour="treatment",
    ...                        shape="treatment"))
    ...         + geom_point()
    ...         + scale_colour_manual(values=["#005b96", "#d55e00"])
    ...         + dp.theme_depictr())
    >>> report = dp.check_figure(good)
    >>> set(report["verdict"])
    {'pass'}

    The same figure destined for an 8.9 cm journal column, where the text is
    half the size it looks on screen.

    >>> shrunk = dp.check_figure(good, width_cm=8.9)
    >>> shrunk.loc[shrunk["check"] == "text_size", "verdict"].item()
    'fail'
    """
    from plotnine import ggplot
    from plotnine.composition import Compose

    # A composite has several panels, each with its own scales, theme and text.
    # One table of numbers cannot describe them all.
    if isinstance(plot, Compose):
        raise TypeError(
            "`plot` is a multi-panel composite. Check each panel on its own."
        )
    if not isinstance(plot, ggplot):
        raise TypeError(
            "`plot` must be a plot object, as returned by any depictr plotting "
            "function."
        )
    _positive_scalar(width_cm, "width_cm")
    _positive_scalar(render_width_cm, "render_width_cm")
    _positive_scalar(min_delta_e, "min_delta_e")
    _positive_scalar(min_text_pt, "min_text_pt")

    # Drawing mutates the plot it is given (it builds the layers in place), so
    # the audit works on a copy and leaves the caller's figure untouched.
    built = copy.deepcopy(plot)
    figure = built.draw(show=False)
    try:
        plot_bg = _plot_background(built, figure)
        # A blank panel, as under theme_depictr(), leaves the axes patch
        # invisible but white, so it is read only when it is drawn.
        panel_bg = (_painted(built.axs[0].patch, plot_bg) if built.axs
                    else None) or plot_bg
        colours, redundant = _figure_colour_encoding(built)
        text = _figure_text(figure, plot_bg)
    finally:
        plt.close(figure)

    rows = _separability_rows(colours, min_delta_e)
    rows.append(_text_size_row(text, width_cm, render_width_cm, min_text_pt))
    rows.append(_text_contrast_row(text, width_cm, render_width_cm))
    rows.append(_geometry_contrast_row(colours, panel_bg))
    rows.append(_redundant_encoding_row(colours, redundant))
    return pd.DataFrame(
        rows, columns=["check", "measured", "threshold", "verdict", "detail"]
    )


# --- validation -------------------------------------------------------------


def _positive_scalar(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"`{name}` must be a single positive number.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"`{name}` must be a single positive number.")


# --- colour bookkeeping -----------------------------------------------------


def _as_hex(colours) -> list[str]:
    """Normalise colours to lower-case six-digit hex, dropping the invisible.

    Anything missing, unparseable or fully transparent is removed rather than
    carried through as a colour, since it encodes nothing. The result is the
    same canonical form the R twin produces, so the two can be compared
    directly.
    """
    out = []
    for colour in np.atleast_1d(np.asarray(colours, dtype=object)):
        if colour is None or (isinstance(colour, float) and np.isnan(colour)):
            continue
        try:
            r, g, b, a = mcolors.to_rgba(colour)
        except (ValueError, TypeError):
            continue
        if a == 0:
            continue
        out.append(mcolors.to_hex((r, g, b)))
    return out


def _composite(colour, background: str) -> str:
    """A colour as drawn over a background, given its own alpha.

    Each channel, scaled to run from 0 to 1, is
    ``alpha * colour + (1 - alpha) * background``,
    then scaled to 8 bits with ``floor(255 * drawn + 0.5)``. The R twin does
    the same arithmetic in the same order, so the two agree to the last bit.
    matplotlib's ``to_hex`` rounds half to even and would disagree with R at
    some alphas.
    """
    r, g, b, a = mcolors.to_rgba(colour)
    fg = np.array([r, g, b])
    bg = np.array(mcolors.to_rgb(background))
    drawn = a * fg + (1 - a) * bg
    channels = np.floor(255 * drawn + 0.5).astype(int)
    return "#" + "".join(f"{int(c):02x}" for c in channels)


def _painted(patch, parent: str) -> str | None:
    """The colour a background patch paints over its parent, or None.

    None when the patch is missing, hidden or fully transparent, so that a
    blank element falls back to whatever lies beneath it. A translucent face is
    composited over the parent, so a half-transparent black strip on white is
    measured as the grey it shows.
    """
    if patch is None or not patch.get_visible():
        return None
    face = mcolors.to_rgba(patch.get_facecolor())
    if face[3] == 0:
        return None
    return _composite(face, parent)


def _plot_background(plot, figure) -> str:
    """The colour the figure's own background is painted.

    plotnine paints ``plot_background`` onto a rectangle of its own and leaves
    the matplotlib figure patch at its default white, whatever the theme says.
    Reading the patch alone therefore measured every label against white on a
    figure with a dark background, reporting a contrast the reader never gets
    and disagreeing with the R twin, which reads the resolved
    ``plot.background`` element. The patch stays as the fallback, for a theme
    that paints nothing, and white stands in for the paper beneath both.
    """
    painted = getattr(getattr(plot, "theme", None), "targets", None)
    painted = getattr(painted, "plot_background", None)
    return (_painted(painted, "#ffffff") or _painted(figure.patch, "#ffffff")
            or "#ffffff")


def _relative_luminance(colours) -> np.ndarray:
    """WCAG 2.2 relative luminance of one or more colours, in [0, 1].

    The linearisation is the sRGB transfer function that ``_srgb_to_linear``
    applies everywhere else in the package, with its 0.04045 breakpoint, which
    is the one the WCAG 2.2 definition of relative luminance prints. Before May
    2021 that definition printed 0.03928, and the W3C's note on the change says
    the old value had no practical effect: no 8-bit channel value lies between
    the two, so no ratio differs.
    """
    rgb = np.array([mcolors.to_rgb(c) for c in np.atleast_1d(colours)])
    return _srgb_to_linear(rgb) @ np.array([0.2126, 0.7152, 0.0722])


def _contrast_ratio(colours, backgrounds) -> np.ndarray:
    """WCAG 2.2 contrast ratio: (L1 + 0.05) / (L2 + 0.05), lighter on top.

    ``backgrounds`` is one colour for all, or one per colour.
    """
    a = _relative_luminance(colours)
    b = _relative_luminance(backgrounds)
    return (np.maximum(a, b) + 0.05) / (np.minimum(a, b) + 0.05)


def _varies(data, column: str) -> bool:
    if column not in data:
        return False
    return data[column].dropna().nunique() >= 2


def _figure_colour_encoding(plot) -> tuple[list[str], list[str]]:
    """The colours a figure uses to tell groups apart, and what varies with them.

    A colour or fill counts as encoding when a single layer draws more than one
    of them and the scale behind it is discrete. A continuous scale is a smooth
    ramp, where neighbouring colours are meant to be close, so measuring the
    distance between them would only ever say that a gradient is a gradient.

    The colours are returned sorted, so that a tie between two equally close
    pairs resolves the same way in both engines.
    """
    from plotnine import geom_label, geom_text
    from plotnine.scales.scale_continuous import scale_continuous

    continuous: set[str] = set()
    for scale in plot.scales:
        if isinstance(scale, scale_continuous):
            continuous.update(scale.aesthetics)

    colours: list[str] = []
    redundant: list[str] = []
    for layer in plot.layers:
        # A text layer's colour is chosen for legibility against whatever it
        # sits on, not to code a group, so counting it would report the distance
        # between black and white labels as though it were the distance between
        # categories.
        if isinstance(layer.geom, (geom_text, geom_label)):
            continue
        data = layer.data
        encodes = False
        for aesthetic in ("color", "fill"):
            if aesthetic in continuous or not _varies(data, aesthetic):
                continue
            hex_colours = _as_hex(data[aesthetic].dropna().unique())
            if len(set(hex_colours)) < 2:
                continue
            colours += [c for c in hex_colours if c not in colours]
            encodes = True
        if not encodes:
            continue
        also = [a for a in ("shape", "linetype") if _varies(data, a)]
        if len(also) > len(redundant):
            redundant = also
    return sorted(set(colours)), redundant


# --- text bookkeeping -------------------------------------------------------


class _Text(NamedTuple):
    """One drawn text: point size, colour, weight and its own background."""

    size: float
    colour: str
    bold: bool
    background: str


def _is_bold(weight) -> bool:
    """Whether a matplotlib font weight, named or numeric, is bold (700+)."""
    if isinstance(weight, str):
        weight = weight_dict.get(weight.lower(), 400)
    return float(weight) >= 700


def _figure_text(figure, plot_bg: str) -> list[_Text]:
    """Every piece of text the figure will actually draw.

    Reads the matplotlib text artists, which is where the resolved point size
    and colour of a label exist, and where an element the theme defines but the
    figure never draws simply does not appear.

    Text an axes owns directly is skipped, which is what leaves layer and
    annotation text out of the measurement: tick labels, axis titles and the
    legend belong to the axis, the figure or the legend rather than to
    ``Axes.texts``.

    Each text is paired with the background it is drawn on, which WCAG measures
    contrast against. plotnine's strip text carries its strip as ``.patch``,
    and legend text sits inside an anchored box whose patch is the legend
    background. Either counts only when it is drawn, and everything else is on
    the plot background.
    """
    # matplotlib's findobj lists an artist's children before the artist, so an
    # inner box is met before the box around it and the innermost one wins.
    boxed: dict[int, str] = {}
    for box in figure.findobj(lambda a: isinstance(a, AnchoredOffsetbox)):
        painted = _painted(box.patch, plot_bg)
        if painted is None:
            continue
        for text in box.findobj(mtext.Text):
            boxed.setdefault(id(text), painted)

    from_layers = {id(t) for ax in figure.axes for t in ax.texts}
    seen: set[int] = set()
    out: list[_Text] = []
    for artist in figure.findobj(mtext.Text):
        if id(artist) in from_layers or id(artist) in seen:
            continue
        seen.add(id(artist))
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        alpha = artist.get_alpha()
        if alpha is not None and alpha == 0:
            continue
        colour = _as_hex([artist.get_color()])
        size = float(artist.get_fontsize())
        if not colour or not np.isfinite(size):
            continue
        strip = getattr(artist, "patch", None)
        if isinstance(strip, mpatches.Patch):
            background = _painted(strip, plot_bg) or plot_bg
        else:
            background = boxed.get(id(artist), plot_bg)
        out.append(_Text(size, colour[0], _is_bold(artist.get_fontweight()),
                         background))
    return out


# --- rows -------------------------------------------------------------------


def _verdict(measured: float, threshold: float) -> str:
    if measured is None or (isinstance(measured, float) and np.isnan(measured)):
        return "not applicable"
    return "pass" if measured >= threshold else "fail"


def _row(check: str, measured, threshold: float, detail: str) -> dict:
    verdict = _verdict(measured, threshold)
    return {
        "check": check,
        "measured": (np.nan if verdict == "not applicable"
                     else round(float(measured), 2)),
        "threshold": float(threshold),
        "verdict": verdict,
        "detail": detail,
    }


def _min_pairwise_gap(values: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Smallest pairwise gap in one dimension, walked in the twin's order."""
    best, pair = np.inf, (0, 0)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            gap = abs(float(values[i] - values[j]))
            if gap < best:
                best, pair = gap, (i, j)
    return best, pair


def _separability_rows(colours: list[str], min_delta_e: float) -> list[dict]:
    """The four colour rows plus the greyscale row."""
    names = ["colour_separability"] + [
        f"colour_separability_{d}" for d in DEFICIENCIES
    ]
    if len(colours) < 2:
        return [_row(name, np.nan, min_delta_e, _NO_COLOUR_DETAIL)
                for name in names + ["greyscale_separability"]]

    rows = []
    for name, condition in zip(names, ["normal", *DEFICIENCIES], strict=True):
        seen = colours if condition == "normal" else simulate_cvd(colours, condition)
        distance, (i, j) = _min_pairwise_delta_e(seen)
        rows.append(_row(name, distance, min_delta_e,
                         f"Closest pair {colours[i]} and {colours[j]} of "
                         f"{len(colours)} encoding colours."))
    # Lightness is what a black-and-white printer keeps, and CIE L* is a
    # function of luminance alone, so the greyscale difference between two
    # colours is the CIE76 distance between their lightnesses.
    distance, (i, j) = _min_pairwise_gap(_rgb_to_lab(colours)[:, 0])
    rows.append(_row("greyscale_separability", distance, min_delta_e,
                     f"Closest pair {colours[i]} and {colours[j]} in CIE "
                     f"lightness."))
    return rows


def _text_size_row(text, width_cm: float, render_width_cm: float,
                   min_text_pt: float) -> dict:
    if not text:
        return _row("text_size", np.nan, min_text_pt,
                    "The figure draws no text.")
    nominal = min(t.size for t in text)
    return _row("text_size", nominal * width_cm / render_width_cm, min_text_pt,
                f"Smallest text {nominal:.2f} pt, drawn at "
                f"{render_width_cm:.2f} cm and printed at {width_cm:.2f} cm.")


def _text_contrast_row(text: list[_Text], width_cm: float,
                       render_width_cm: float) -> dict:
    """The text that falls furthest short of its own WCAG threshold.

    Each text is measured against its own background and held to 3:1 when it
    is large at its printed size (18 pt, or 14 pt in bold) and to 4.5:1
    otherwise. The row reports the text with the smallest ratio relative to its
    threshold. Ties go to the smaller printed size, then to the colour and
    background in byte order, which is the order the R twin breaks them in.
    """
    if not text:
        return _row("text_contrast", np.nan, _WCAG_TEXT_CONTRAST,
                    "The figure draws no text.")
    printed = [t.size * width_cm / render_width_cm for t in text]
    thresholds = [
        _WCAG_LARGE_TEXT_CONTRAST
        if p >= _WCAG_LARGE_TEXT_PT or (t.bold and p >= _WCAG_LARGE_BOLD_TEXT_PT)
        else _WCAG_TEXT_CONTRAST
        for t, p in zip(text, printed, strict=True)
    ]
    ratios = _contrast_ratio([t.colour for t in text],
                             [t.background for t in text])
    worst = min(range(len(text)), key=lambda i: (
        ratios[i] / thresholds[i], printed[i], text[i].colour,
        text[i].background))
    chosen = text[worst]
    weight = " bold" if chosen.bold else ""
    return _row("text_contrast", ratios[worst], thresholds[worst],
                f"Lowest-contrast text {chosen.colour} on {chosen.background} "
                f"at {printed[worst]:.2f} pt{weight}.")


def _geometry_contrast_row(colours: list[str], background: str) -> dict:
    if not colours:
        return _row("geometry_contrast", np.nan, _WCAG_OBJECT_CONTRAST,
                    _NO_COLOUR_DETAIL)
    ratios = _contrast_ratio(colours, background)
    worst = int(np.argmin(ratios))
    return _row("geometry_contrast", ratios[worst], _WCAG_OBJECT_CONTRAST,
                f"Lowest-contrast colour {colours[worst]} on {background}.")


def _redundant_encoding_row(colours: list[str], redundant: list[str]) -> dict:
    if len(colours) < 2:
        return _row("redundant_encoding", np.nan, 1, _NO_COLOUR_DETAIL)
    if redundant:
        detail = f"Colour is joined by {' and '.join(redundant)}."
    else:
        detail = "Colour alone distinguishes the groups."
    return _row("redundant_encoding", len(redundant), 1, detail)
