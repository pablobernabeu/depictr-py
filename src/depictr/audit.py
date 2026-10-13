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

    Colours and text are measured as drawn. A translucent colour is composited
    over the background it is drawn on with the alpha it is drawn with, so it
    is measured as the paler colour the reader sees. As plotnine draws them,
    bars, tiles, areas, densities, ribbons, polygons, violins, boxplots,
    crossbars and smooths apply a layer's alpha to their fill and leave the
    outline opaque. Lines, paths, segments, error bars and reference lines
    apply it to their colour, and points apply it to both. Any other geom is
    taken to apply it to both, the cautious assumption. A colour that carries
    its own alpha keeps it, as plotnine draws it. A colour drawn at several
    alphas within one layer is measured at the smallest. Where marks overlap
    one another, the colour that shows depends on what lies beneath it, and
    that is not measured.

    ``colour_separability`` is the smallest CIE76 colour difference (Delta-E)
    between two of the figure's encoding colours, and the three
    ``colour_separability_*`` rows repeat that measurement after simulating each
    dichromacy at full severity with :func:`depictr.simulate_cvd`. Encoding
    colours are the distinct colour and fill values that tell groups apart. A
    layer contributes its colours when it maps colour or fill to a discrete
    scale, even if it draws a single colour, or when it draws two or more.
    Groups drawn one per layer are therefore audited as they would be in one
    layer. A colour set as a constant, as in ``geom_point(colour="red")``, is
    left out, even when another layer sets a different one, and so is a lone
    colour passed through ``scale_colour_identity()``. The audit cannot tell
    such a colour from decoration, so a colour meant to tell groups apart
    should be mapped, as ``aes(colour="'treated'")`` maps a constant, with
    ``scale_colour_manual()`` to fix its values. A continuous colour or fill
    scale is a smooth ramp rather than a set of codes, so it is excluded. The
    outline colours, across every layer, are compared among themselves, and so
    are the fill colours, and the row reports the closest pair from either set.
    An outline is therefore never compared with its own fill, nor a group with
    itself as another layer draws it.

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
    it, and so is text drawn with an alpha. Text that is large at its printed
    size, 18 pt or 14 pt in bold, is held to 3:1, and other text to 4.5:1, as
    success criterion 1.4.3 asks. The row's ``threshold`` is the one that
    applies to the text it reports, and its ``detail`` names that text's
    colour, background, printed size and weight.

    ``geometry_contrast`` is the smallest WCAG 2.2 contrast ratio of any
    encoded group's drawn colour against the panel background, or against the
    plot background where the panel paints none. Where a layer maps its outline
    colour to the same groups as its fill, each group counts the better of the
    two, since WCAG lets a border carry a graphical object's contrast. For a
    translucent colour, the ``detail`` names the colour, its alpha and the
    colour it is drawn as.

    ``redundant_encoding`` counts how many of shape and line type also vary
    across the layers that draw the encoding colours. They are gathered as the
    colours are. A layer contributes the values it maps to a discrete scale, or
    the two or more it draws, and a shape or line type set as a constant is
    left out. Zero means the distinction between groups is carried by colour
    alone, the single most common way an otherwise careful figure becomes
    unreadable.

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
        # Marks are drawn on the panel, so a translucent one shows the panel
        # through.
        encoding = _figure_colour_encoding(built, panel_bg)
        text = _figure_text(figure, plot_bg)
    finally:
        plt.close(figure)

    rows = _separability_rows(encoding.sets, min_delta_e)
    rows.append(_text_size_row(text, width_cm, render_width_cm, min_text_pt))
    rows.append(_text_contrast_row(text, width_cm, render_width_cm))
    rows.append(_geometry_contrast_row(encoding.marks, panel_bg))
    rows.append(_redundant_encoding_row(encoding.sets, encoding.redundant))
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


def _parse_colour(colour) -> tuple[str, float] | None:
    """A colour's canonical hex and its own alpha, or None if it draws nothing.

    Anything missing or unparseable is None, and so is ``"none"``, which
    plotnine draws as nothing. The hex is lower-case and six-digit, the same
    canonical form the R twin produces, so the two can be compared directly.
    The colour's own alpha, in [0, 1], is left for the caller to apply.
    """
    if colour is None or (isinstance(colour, float) and np.isnan(colour)):
        return None
    if isinstance(colour, str) and colour.strip().lower() in ("", "none"):
        return None
    try:
        r, g, b, a = mcolors.to_rgba(colour)
    except (ValueError, TypeError):
        return None
    return mcolors.to_hex((r, g, b)), a


def _composite(colour, background: str, alpha: float = 1.0) -> str:
    """A colour as drawn over a background, given its alpha.

    Each channel, scaled to run from 0 to 1, is
    ``alpha * colour + (1 - alpha) * background``,
    then scaled to 8 bits with ``floor(255 * drawn + 0.5)``. The R twin does
    the same arithmetic in the same order, so the two agree to the last bit.
    matplotlib's ``to_hex`` rounds half to even and would disagree with R at
    some alphas, 0.3 and 0.7 among them. ``alpha`` multiplies the colour's own
    alpha.
    """
    r, g, b, own = mcolors.to_rgba(colour)
    a = own * alpha
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


def _legible_text_colour(fills) -> list[str]:
    """Black or white for text on each fill, whichever has the higher contrast.

    Against a fill of relative luminance L, black reaches (L + 0.05) / 0.05 and
    white 1.05 / (L + 0.05). The two tie at L = 0.179, where each reaches
    4.58:1, so the better of the two clears the 4.5:1 that WCAG 2.2 asks of
    text (success criterion 1.4.3) on any fill. A near-black cannot promise
    that: against fills of luminance 0.18 to 0.22, neither ``#1a1a1a`` nor
    white reaches 4.5:1. It colours the cell labels of
    :func:`depictr.confusion_matrix_plot` and
    :func:`depictr.correlation_heatmap`. The R twin's ``.legible_text_colour()``
    makes the same choice, black on a tie.
    """
    fills = list(np.atleast_1d(fills))
    black = _contrast_ratio("#000000", fills)
    white = _contrast_ratio("#ffffff", fills)
    return ["#000000" if b >= w else "#ffffff"
            for b, w in zip(black, white, strict=True)]


class _Drawn(NamedTuple):
    """A declared colour, the alpha it is drawn with and the colour it shows."""

    declared: str
    alpha: float
    drawn: str


class _Encoding(NamedTuple):
    """What a figure tells apart by colour, as drawn.

    ``sets`` holds the drawn colours of the colour and of the fill aesthetic
    across all layers, each a sorted list of ``(drawn, declared)`` pairs, and
    leaves out an aesthetic that encodes nothing. ``marks`` holds one drawn
    colour per encoded group, for the geometry check, and ``redundant`` the
    non-colour channels that vary alongside the colours.
    """

    sets: dict[str, list[tuple[str, str]]]
    marks: list[_Drawn]
    redundant: list[str]


def _alpha_targets(geom) -> set[str]:
    """Which of a layer's colour and fill take its alpha, as plotnine draws it.

    Read from the geoms' drawing code. Rectangles (bars, columns, tiles,
    histograms), ribbons (areas, densities), polygons, violins, boxplots,
    crossbars and smooths pass the layer's alpha to the fill alone and draw the
    outline, the whiskers and the median or fitted line opaque. Paths (lines,
    steps), segments, error bars, line ranges and the reference lines pass it
    to the colour. Points and point ranges pass it to both, and so does any
    geom not named here, the cautious assumption for a geom whose drawing has
    not been read. The R twin keeps the same table for ggplot2's geoms.
    """
    from plotnine import (
        geom_abline,
        geom_boxplot,
        geom_crossbar,
        geom_errorbar,
        geom_errorbarh,
        geom_hline,
        geom_linerange,
        geom_path,
        geom_polygon,
        geom_rect,
        geom_ribbon,
        geom_segment,
        geom_smooth,
        geom_violin,
        geom_vline,
    )

    if isinstance(geom, (geom_rect, geom_ribbon, geom_polygon, geom_violin,
                         geom_boxplot, geom_crossbar, geom_smooth)):
        return {"fill"}
    if isinstance(geom, (geom_path, geom_segment, geom_errorbar,
                         geom_errorbarh, geom_linerange, geom_hline,
                         geom_vline, geom_abline)):
        return {"color"}
    return {"color", "fill"}


def _keeps_own_alpha(colour) -> bool:
    """Whether plotnine draws a colour in its own alpha, whatever the layer's.

    mizani's ``to_rgba``, which plotnine's geoms call, leaves a colour that
    already carries an alpha untouched: a nine-character hex string or an RGBA
    tuple. Any other colour takes the layer's alpha.
    """
    if isinstance(colour, str):
        return colour.startswith("#") and len(colour) == 9
    return isinstance(colour, tuple) and len(colour) == 4


def _layer_mapped(layer, plot) -> set[str]:
    """The aesthetics a layer maps.

    A layer maps what its own ``aes()`` names and, unless it was made with
    ``inherit_aes=False``, what the plot's ``aes()`` names. An aesthetic also
    set to a constant is drawn as the constant, so it does not count as mapped.
    """
    mapped = set(layer.mapping)
    if layer.inherit_aes:
        mapped |= set(plot.mapping or {})
    return mapped - set(layer.geom.aes_params)


def _layer_ink(data, aesthetic: str,
               applies: bool) -> list[tuple[int, str, float]]:
    """The colour one aesthetic of a layer draws on each row, and its alpha.

    Returns ``(row, declared, alpha)`` for each row of the layer data that
    draws something, with ``declared`` in canonical hex. Where the geom does
    not pass its alpha to this aesthetic, the colour's own alpha is all there
    is. A row drawn fully transparent is left out, since it shows nothing.
    """
    values = data[aesthetic].to_numpy()
    if "alpha" in data:
        layer_alpha = pd.to_numeric(data["alpha"], errors="coerce").to_numpy()
    else:
        layer_alpha = np.ones(len(values))
    out = []
    for row, (value, layer_a) in enumerate(zip(values, layer_alpha,
                                               strict=True)):
        parsed = _parse_colour(value)
        if parsed is None:
            continue
        declared, own = parsed
        if applies and not _keeps_own_alpha(value):
            alpha = 1.0 if np.isnan(layer_a) else float(layer_a)
        else:
            alpha = own
        if alpha > 0:
            out.append((row, declared, alpha))
    return out


def _drawn_by_value(ink, background: str) -> dict[str, _Drawn]:
    """Each declared colour of a layer as drawn, at the smallest alpha it takes."""
    smallest: dict[str, float] = {}
    for _, declared, alpha in ink:
        smallest[declared] = min(alpha, smallest.get(declared, alpha))
    return {declared: _Drawn(declared, alpha,
                             _composite(declared, background, alpha))
            for declared, alpha in smallest.items()}


def _shared_groups(colour, fill) -> list[tuple[str, str]] | None:
    """The groups a layer's colour and fill share, when they mark the same ones.

    The two mark the same groups when each colour goes with exactly one fill on
    the rows that draw both, and the other way round, and every colour and fill
    the layer draws is among those pairs. Returns the ``(colour, fill)`` pairs,
    one per group, or None when the two mark different groups.
    """
    colour_of = {row: declared for row, declared, _ in colour}
    fill_of = {row: declared for row, declared, _ in fill}
    pairs = {(colour_of[row], fill_of[row])
             for row in colour_of.keys() & fill_of.keys()}
    colours = {c for c, _ in pairs}
    fills = {f for _, f in pairs}
    one_to_one = len(colours) == len(pairs) == len(fills)
    covers = (colours == set(colour_of.values())
              and fills == set(fill_of.values()))
    return sorted(pairs) if pairs and one_to_one and covers else None


def _layer_marks(ink, drawn, background: str) -> list[_Drawn]:
    """One drawn colour per group of a layer, for the geometry check.

    A group whose outline colour and fill are both mapped to it counts the one
    of the two that contrasts better with the background. WCAG's Understanding
    document for success criterion 1.4.11 lets a border supply a graphical
    object's contrast, and an area geom draws its outline opaque however pale
    its fill. Otherwise, each encoding colour stands for its own group.
    """
    if "color" in ink and "fill" in ink:
        groups = _shared_groups(ink["color"], ink["fill"])
        if groups is not None:
            marks = []
            for colour, fill in groups:
                outline, face = drawn["color"][colour], drawn["fill"][fill]
                ratios = _contrast_ratio([outline.drawn, face.drawn],
                                         background)
                marks.append(outline if ratios[0] > ratios[1] else face)
            return marks
    return [mark for by_value in drawn.values() for mark in by_value.values()]


def _figure_colour_encoding(plot, background: str) -> _Encoding:
    """The colours a figure uses to tell groups apart, and what varies with them.

    A layer carries colour or fill when it maps the aesthetic to a discrete
    scale that is not an identity scale, even if every row draws the same
    colour. It also carries one in which it draws two or more distinct colours.
    The figure encodes by the aesthetic when the layers that carry it draw two
    or more distinct colours between them, so groups drawn one per layer count
    as they would in a single layer. A constant set on a layer is not a
    mapping, and a colour passed through an identity scale is drawn as given.
    A layer that draws one such colour alone carries nothing, since the audit
    cannot tell it from decoration. A continuous scale is a smooth ramp, where
    neighbouring colours are meant to be close, so measuring the distance
    between them would only ever say that a gradient is a gradient.

    Shape and line type are gathered by the same rule from the layers that draw
    the encoding colours. Either counts as varying alongside them when those
    layers draw two or more distinct values of it between them.

    Every colour is taken as drawn over ``background``, the panel, at the alpha
    its geom applies to it. The sets are sorted, so that a tie between two
    equally close pairs resolves the same way in both engines.
    """
    from plotnine import geom_label, geom_text
    from plotnine.scales.scale_continuous import scale_continuous
    from plotnine.scales.scale_discrete import scale_discrete
    from plotnine.scales.scale_identity import MapTrainMixin

    continuous: set[str] = set()
    coded: set[str] = set()
    for scale in plot.scales:
        if isinstance(scale, scale_continuous):
            continuous.update(scale.aesthetics)
        elif (isinstance(scale, scale_discrete)
              and not isinstance(scale, MapTrainMixin)):
            # An identity scale (MapTrainMixin) draws each value as given, so a
            # value mapped through it is chosen by hand, as a constant set on a
            # layer is.
            coded.update(scale.aesthetics)

    # What each layer carries in colour and fill, gathered before it is known
    # whether the figure as a whole encodes by either.
    carriers = []
    for layer in plot.layers:
        # A text layer's colour is chosen for legibility against whatever it
        # sits on, not to code a group, so counting it would report the distance
        # between black and white labels as though it were the distance between
        # categories.
        if isinstance(layer.geom, (geom_text, geom_label)):
            continue
        data = layer.data
        mapped = _layer_mapped(layer, plot) & coded
        targets = _alpha_targets(layer.geom)
        ink = {}
        for aesthetic in ("color", "fill"):
            if aesthetic in continuous or aesthetic not in data:
                continue
            rows = _layer_ink(data, aesthetic, aesthetic in targets)
            # Distinct after normalisation: two spellings of one colour are one
            # colour, and an unmapped layer that draws only that carries
            # nothing.
            distinct = {declared for _, declared, _ in rows}
            if distinct and (aesthetic in mapped or len(distinct) >= 2):
                ink[aesthetic] = rows
        if ink:
            carriers.append((ink, data, mapped))
    encodes = {
        aesthetic for aesthetic in ("color", "fill")
        if len({declared for ink, _, _ in carriers
                for _, declared, _ in ink.get(aesthetic, [])}) >= 2
    }

    sets: dict[str, set[tuple[str, str]]] = {"color": set(), "fill": set()}
    marks: list[_Drawn] = []
    others: dict[str, set] = {"shape": set(), "linetype": set()}
    for carried, data, mapped in carriers:
        ink = {aesthetic: rows for aesthetic, rows in carried.items()
               if aesthetic in encodes}
        if not ink:
            continue
        drawn = {aesthetic: _drawn_by_value(rows, background)
                 for aesthetic, rows in ink.items()}
        for aesthetic, by_value in drawn.items():
            sets[aesthetic].update((mark.drawn, mark.declared)
                                   for mark in by_value.values())
        marks.extend(_layer_marks(ink, drawn, background))
        for aesthetic, seen in others.items():
            if aesthetic not in data:
                continue
            values = set(data[aesthetic].dropna())
            if aesthetic in mapped or len(values) >= 2:
                seen.update(values)
    return _Encoding(
        sets={aesthetic: sorted(pairs) for aesthetic, pairs in sets.items()
              if pairs},
        marks=marks,
        redundant=[aesthetic for aesthetic, seen in others.items()
                   if len(seen) >= 2],
    )


# --- text bookkeeping -------------------------------------------------------


class _Text(NamedTuple):
    """One drawn text: point size, colour, weight and its own background.

    ``colour`` is the colour drawn, after compositing, and ``declared`` and
    ``alpha`` are the colour and alpha it was drawn from.
    """

    size: float
    colour: str
    bold: bool
    background: str
    declared: str
    alpha: float


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

    A text drawn with an alpha is composited over its background, as a mark is
    over the panel. When the artist has an alpha of its own, matplotlib draws
    the text in it and ignores the colour's alpha, which counts only when the
    artist has none.
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
        parsed = _parse_colour(artist.get_color())
        size = float(artist.get_fontsize())
        if parsed is None or not np.isfinite(size):
            continue
        declared, own = parsed
        alpha = own if artist.get_alpha() is None else float(artist.get_alpha())
        if alpha <= 0:
            continue
        strip = getattr(artist, "patch", None)
        if isinstance(strip, mpatches.Patch):
            background = _painted(strip, plot_bg) or plot_bg
        else:
            background = boxed.get(id(artist), plot_bg)
        out.append(_Text(size, _composite(declared, background, alpha),
                         _is_bold(artist.get_fontweight()), background,
                         declared, alpha))
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


def _min_pairwise_gap(values: np.ndarray,
                      groups=None) -> tuple[float, tuple[int, int]]:
    """Smallest pairwise gap in one dimension, walked in the twin's order.

    A pair whose ``groups`` are equal is skipped, as in
    ``_min_pairwise_delta_e``.
    """
    best, pair = np.inf, (0, 0)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if groups is not None and groups[i] == groups[j]:
                continue
            gap = abs(float(values[i] - values[j]))
            if gap < best:
                best, pair = gap, (i, j)
    return best, pair


# How a detail names the colours of each aesthetic's set.
_SET_NOUNS = {"color": "colours", "fill": "fill colours"}


def _separability_rows(sets: dict[str, list[tuple[str, str]]],
                       min_delta_e: float) -> list[dict]:
    """The four colour rows plus the greyscale row.

    Each set holds the drawn colours of one aesthetic, and only colours
    declared differently are paired, so a group is never compared with itself.
    The rows report the closest pair over the sets, the first set winning a
    tie.
    """
    names = ["colour_separability"] + [
        f"colour_separability_{d}" for d in DEFICIENCIES
    ]
    if not sets:
        return [_row(name, np.nan, min_delta_e, _NO_COLOUR_DETAIL)
                for name in names + ["greyscale_separability"]]

    def closest_over_sets(measure):
        best = None
        for aesthetic, pairs in sets.items():
            drawn = [d for d, _ in pairs]
            declared = [g for _, g in pairs]
            distance, (i, j) = measure(drawn, declared)
            if best is None or distance < best[0]:
                best = (distance, drawn[i], drawn[j], len(pairs),
                        _SET_NOUNS[aesthetic])
        return best

    rows = []
    for name, condition in zip(names, ["normal", *DEFICIENCIES], strict=True):
        def measure(drawn, declared, condition=condition):
            seen = (drawn if condition == "normal"
                    else simulate_cvd(drawn, condition))
            return _min_pairwise_delta_e(seen, declared)

        distance, a, b, size, noun = closest_over_sets(measure)
        rows.append(_row(name, distance, min_delta_e,
                         f"Closest pair {a} and {b} of {size} encoding "
                         f"{noun}."))
    # Lightness is what a black-and-white printer keeps, and CIE L* is a
    # function of luminance alone, so the greyscale difference between two
    # colours is the CIE76 distance between their lightnesses.
    distance, a, b, _, _ = closest_over_sets(
        lambda drawn, declared: _min_pairwise_gap(_rgb_to_lab(drawn)[:, 0],
                                                  declared))
    rows.append(_row("greyscale_separability", distance, min_delta_e,
                     f"Closest pair {a} and {b} in CIE lightness."))
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
    ink = _ink_as_drawn(chosen.declared, chosen.alpha, chosen.colour)
    return _row("text_contrast", ratios[worst], thresholds[worst],
                f"Lowest-contrast text {ink} on {chosen.background} "
                f"at {printed[worst]:.2f} pt{weight}.")


def _ink_as_drawn(declared: str, alpha: float, drawn: str) -> str:
    """How a detail names a colour.

    As declared when it is opaque, and with its alpha and the colour it is
    drawn as when it is not.
    """
    if alpha >= 1:
        return drawn
    return f"{declared} at alpha {alpha:.2f}, drawn as {drawn},"


def _geometry_contrast_row(marks: list[_Drawn], background: str) -> dict:
    """The encoded group whose drawn colour contrasts least with the background.

    Ties go to the drawn colour, then the declared one, in byte order, which is
    the order the R twin breaks them in.
    """
    if not marks:
        return _row("geometry_contrast", np.nan, _WCAG_OBJECT_CONTRAST,
                    _NO_COLOUR_DETAIL)
    ratios = _contrast_ratio([mark.drawn for mark in marks], background)
    worst = min(range(len(marks)), key=lambda i: (
        ratios[i], marks[i].drawn, marks[i].declared))
    mark = marks[worst]
    ink = _ink_as_drawn(mark.declared, mark.alpha, mark.drawn)
    return _row("geometry_contrast", ratios[worst], _WCAG_OBJECT_CONTRAST,
                f"Lowest-contrast colour {ink} on {background}.")


def _redundant_encoding_row(sets: dict[str, list[tuple[str, str]]],
                            redundant: list[str]) -> dict:
    if not sets:
        return _row("redundant_encoding", np.nan, 1, _NO_COLOUR_DETAIL)
    if redundant:
        detail = f"Colour is joined by {' and '.join(redundant)}."
    else:
        detail = "Colour alone distinguishes the groups."
    return _row("redundant_encoding", len(redundant), 1, detail)
