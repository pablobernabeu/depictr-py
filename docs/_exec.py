"""Rendering helpers for the executed documentation examples.

Not part of the ``depictr`` package. The gallery, home and reference pages run
their code at build time through markdown-exec, and these helpers turn a result
into markup the page can display: ``show`` renders a plotnine figure as inline
SVG, ``table`` renders a data frame as an unstyled HTML table, and ``pre``
renders escaped text as a code block. Each returns its markup rather than
printing it, because markdown-exec gives every executed block a ``print`` of its
own that writes to the block's output buffer, and that ``print`` is not in scope
here; the blocks therefore call ``print(show(p))``.

Figures are emitted as SVG on a transparent background rather than as a raster
PNG, so that the page rather than the image decides the colour of the ink. A
PNG is composited onto matplotlib's opaque figure patch before it is written, so
every figure would arrive as a white plate on the dark slate scheme. The theme
overlay below instead paints each non-data element in one of four sentinel
colours, and ``_adapt`` swaps those for CSS keywords once the file is written:
``currentColor`` for text, ticks and axis lines, a faint ``currentColor`` for
gridlines and panel furniture, and the site's own custom properties for the
title and for the panel an inside legend sits on. One rendering therefore reads
correctly under both colour schemes, and the data colours, which carry the
meaning and are chosen to be colourblind-safe, are left exactly as the package
drew them.
"""

from __future__ import annotations

import io
import re
import warnings

import matplotlib

matplotlib.use("Agg")  # render headlessly; no GUI backend during the build

from plotnine import (
    element_line,
    element_rect,
    element_text,
    guide_colorbar,
    guides,
    theme,
)
from plotnine.composition import Compose
from plotnine.exceptions import PlotnineWarning
from plotnine.scales.scale_continuous import scale_continuous

warnings.filterwarnings("ignore", category=PlotnineWarning)

# Sentinel colours. They only have to survive the round trip through matplotlib
# and be absent from any palette the package can draw with, so they are chosen
# far from the Okabe-Ito set, from viridis and from the greys the geoms use.
_INK = "#010203"    # text, ticks and axis lines
_MUTED = "#040506"  # gridlines, strip bands and legend borders
_BRAND = "#070809"  # the plot title, which carries the brand colour
_PAPER = "#0a0b0c"  # the panel an inside legend sits on

# What the overlay recolours, named at the leaves rather than through a parent
# such as `text` or `line`. Only the colour is given: plotnine merges an element
# into the one already in place, so the weight, alignment and margins
# theme_depictr set are all kept.
_RECOLOUR = {
    "axis_text_x": element_text(color=_INK),
    "axis_text_y": element_text(color=_INK),
    "axis_title_x": element_text(color=_INK),
    "axis_title_y": element_text(color=_INK),
    "legend_text": element_text(color=_INK),
    "legend_title": element_text(color=_INK),
    "plot_title": element_text(color=_BRAND),
    "plot_subtitle": element_text(color=_INK),
    "plot_caption": element_text(color=_INK),
    "strip_text_x": element_text(color=_INK),
    "strip_text_y": element_text(color=_INK),
    "axis_ticks_major_x": element_line(color=_INK),
    "axis_ticks_major_y": element_line(color=_INK),
    "axis_line_x": element_line(color=_INK),
    "axis_line_y": element_line(color=_INK),
    "panel_grid_major_x": element_line(color=_MUTED),
    "panel_grid_major_y": element_line(color=_MUTED),
    "strip_background": element_rect(fill=_MUTED, color="none"),
    "legend_key": element_rect(fill="none", color=_MUTED),
    "legend_background": element_rect(fill=_PAPER, color=_MUTED),
}

# The themeables each of the above sits under. A property is left alone when it,
# or anything above it, was set to element_blank, because giving it a colour
# would put back something the plot deliberately removed: theme_depictr blanks
# the axis ticks, the axis lines and the legend background, ridgeline_plot and
# outlier_plot blank the y-axis text and the horizontal gridlines, and
# explore_pairs blanks the axis text on its inner panels. plotnine does not model
# this hierarchy as class inheritance, so it is written out.
_ANCESTRY = {
    "axis_text_x": ("axis_text", "text"),
    "axis_text_y": ("axis_text", "text"),
    "axis_title_x": ("axis_title", "text"),
    "axis_title_y": ("axis_title", "text"),
    "legend_text": ("text",),
    "legend_title": ("title", "text"),
    "plot_title": ("title", "text"),
    "plot_subtitle": ("title", "text"),
    "plot_caption": ("title", "text"),
    "strip_text_x": ("strip_text", "text"),
    "strip_text_y": ("strip_text", "text"),
    "axis_ticks_major_x": ("axis_ticks_major", "axis_ticks", "line"),
    "axis_ticks_major_y": ("axis_ticks_major", "axis_ticks", "line"),
    "axis_line_x": ("axis_line", "line"),
    "axis_line_y": ("axis_line", "line"),
    "panel_grid_major_x": ("panel_grid_major", "panel_grid", "line"),
    "panel_grid_major_y": ("panel_grid_major", "panel_grid", "line"),
    "strip_background": ("rect",),
    "legend_key": ("rect",),
    "legend_background": ("rect",),
}


def _blanked(p) -> set:
    """Names of the themeables a plot removed."""
    themeables = getattr(getattr(p, "theme", None), "themeables", {})
    return {name for name, themeable in themeables.items()
            if type(getattr(themeable, "theme_element", None)).__name__
            == "element_blank"}


def _adaptive(p) -> theme:
    """The recolouring overlay for one figure, minus anything it has removed."""
    gone = _blanked(p)
    return theme(**{
        name: element
        for name, element in _RECOLOUR.items()
        if name not in gone and not gone.intersection(_ANCESTRY[name])
    })


# Order matters: the two-part swaps have to run before the bare sentinel is
# taken out from under them.
_SWAPS = (
    (f"fill: {_MUTED}", "fill: currentColor; fill-opacity: .07"),
    (f"stroke: {_MUTED}", "stroke: currentColor; stroke-opacity: .22"),
    (_MUTED, "currentColor"),
    (_INK, "currentColor"),
    (_BRAND, "var(--md-typeset-a-color)"),
    (_PAPER, "var(--md-default-bg-color)"),
)

_XML_PROLOGUE = re.compile(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>", re.DOTALL)
_METADATA = re.compile(r"<metadata>.*?</metadata>", re.DOTALL)
_LOCAL_ID = re.compile(r'(id="|url\(#|href="#)([\w.:-]+)')
# matplotlib writes coordinates to six decimal places. The user unit here is a
# point, so the sixth decimal is a millionth of a point; two are kept, which is
# a hundredth of a point and well under a device pixel at any plausible zoom.
# On a scatter-heavy figure this alone takes about a quarter off the file.
_LONG_DECIMAL = re.compile(r"\d+\.\d{3,}")
# A lone ggplot honours `transparent` and writes its figure patch as `fill:
# none`; a composition ignores the argument and paints the canvas white, which
# is the white plate this whole helper exists to avoid. The figure patch is
# always matplotlib's `patch_1`, so clear it here instead.
_FIGURE_PATCH = re.compile(
    r'(<g id="[\w-]*patch_1">\s*<path d="[^"]*" style="fill: )#ffffff(")'
)
# matplotlib repeats every label as an XML comment beside the glyphs that draw
# it. Dropping them saves a little room and, more usefully, takes the only
# non-ASCII text out of the file.
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_NON_ASCII = re.compile(r"[^\x00-\x7f]")

# A scatter of n points is written as n full bezier outlines, one per marker,
# each a few hundred bytes of absolute coordinates. matplotlib only factors a
# repeated marker into a definition when every point shares one transform, which
# a plotnine layer rarely does, so the outlines are recovered here instead: a
# shape that recurs is defined once and each occurrence becomes a `<use>` at its
# own offset. The scatter-plot matrix goes from about 1.5 MB to a tenth of that.
_DEFS_BLOCK = re.compile(r"<defs>.*?</defs>", re.DOTALL)
_DRAWN_PATH = re.compile(r'<path d="([^"]*)"((?:\s+[\w:-]+="[^"]*")*)\s*/>')
_PATH_TOKEN = re.compile(r"[MLCZz]|-?\d+(?:\.\d+)?(?:e-?\d+)?")
_SIMPLE_COMMANDS = frozenset("MLCZz")
# Below this many repeats the definition costs more than the outlines it saves.
_REUSE_THRESHOLD = 4

_counter = 0


def _shorten(match: re.Match) -> str:
    return f"{float(match.group()):.2f}".rstrip("0").rstrip(".")


def _canonical(d: str) -> tuple[str, float, float] | None:
    """Split a path into a shape anchored at the origin and its offset.

    Returns ``None`` for any path this cannot handle exactly: one that uses a
    command other than moveto, lineto, curveto and closepath, or whose numbers
    do not pair up into coordinates. Every command that survives takes its
    arguments as x, y pairs, and closepath takes none, so the parity of the
    number sequence tracks the axis throughout.
    """
    tokens = _PATH_TOKEN.findall(d)
    if not tokens or tokens[0] != "M":
        return None
    numbers = []
    parts = []
    for token in tokens:
        if token[0].isalpha():
            if token not in _SIMPLE_COMMANDS:
                return None
            parts.append(token)
        else:
            parts.append(None)
            numbers.append(float(token))
    if len(numbers) < 2 or len(numbers) % 2:
        return None
    ox, oy = numbers[0], numbers[1]
    shifted = [n - (ox if i % 2 == 0 else oy) for i, n in enumerate(numbers)]
    it = iter(f"{n:.2f}".rstrip("0").rstrip(".") or "0" for n in shifted)
    shape = " ".join(part if part is not None else next(it) for part in parts)
    return shape, ox, oy


def _reuse_markers(svg: str, prefix: str) -> str:
    """Replace repeated path outlines with references to a single definition."""
    masked = [(m.start(), m.end()) for m in _DEFS_BLOCK.finditer(svg)]

    def in_defs(pos: int) -> bool:
        return any(start <= pos < end for start, end in masked)

    matches = [m for m in _DRAWN_PATH.finditer(svg) if not in_defs(m.start())]
    parsed = {m.start(): _canonical(m.group(1)) for m in matches}
    counts: dict[str, int] = {}
    for value in parsed.values():
        if value is not None:
            counts[value[0]] = counts.get(value[0], 0) + 1
    shapes = {shape: f"{prefix}s{i}"
              for i, (shape, n) in enumerate(sorted(counts.items()))
              if n >= _REUSE_THRESHOLD}
    if not shapes:
        return svg

    pieces = []
    cursor = 0
    group_attrs = None
    for m in matches:
        value = parsed[m.start()]
        if value is None or value[0] not in shapes:
            continue
        shape, ox, oy = value
        use = (f'<use xlink:href="#{shapes[shape]}" '
               f'x="{ox:g}" y="{oy:g}"/>')
        gap = svg[cursor:m.start()]
        # The attributes stay on a wrapping group rather than on the `<use>`
        # itself: a `<use>` places its content with a translate, and a clip path
        # named on the same element would be shifted along with it. Every marker
        # in a layer carries the same attributes, so consecutive ones share one
        # group and the attributes are written once instead of per point.
        if group_attrs == m.group(2) and not gap.strip():
            pieces.append(use)
        else:
            if group_attrs is not None:
                pieces.append("</g>")
            pieces.append(gap)
            pieces.append(f"<g{m.group(2)}>{use}")
            group_attrs = m.group(2)
        cursor = m.end()
    if group_attrs is not None:
        pieces.append("</g>")
    pieces.append(svg[cursor:])
    svg = "".join(pieces)

    definitions = "".join(
        f'<path id="{name}" d="{shape}"/>' for shape, name in shapes.items()
    )
    marker = "</svg>"
    return svg.replace(marker, f"<defs>{definitions}</defs>{marker}")


def _ascii(markup: str) -> str:
    """Return markup as pure ASCII.

    The build writes each page through a stream that uses the machine's default
    encoding, which on Windows is cp1252, so a stray Greek letter or square root
    sign in a figure would abort the build. Numeric character references say the
    same thing in ASCII and are understood wherever the page is read.
    """
    return _NON_ASCII.sub(lambda m: f"&#x{ord(m.group()):x};", markup)


def _adapt(svg: str, prefix: str, title: str) -> str:
    """Turn a matplotlib SVG file into an inline, theme-adaptive SVG element."""
    svg = _XML_PROLOGUE.sub("", svg)
    svg = _METADATA.sub("", svg)
    svg = _COMMENT.sub("", svg)
    for sentinel, replacement in _SWAPS:
        svg = svg.replace(sentinel, replacement)
    # Several figures share a page, and matplotlib derives its clip-path and
    # glyph ids from the content, so the same id would otherwise be defined more
    # than once in the document. Namespace every id and every reference to one.
    svg = _LOCAL_ID.sub(lambda m: m.group(1) + prefix + m.group(2), svg)
    svg = _FIGURE_PATCH.sub(r"\1none\2", svg, count=1)
    svg = _LONG_DECIMAL.sub(_shorten, svg)
    svg = _reuse_markers(svg, prefix)
    svg = svg.replace(
        "<svg ", '<svg class="dp-plot" role="img" ', 1
    ).replace(">", f"><title>{title}</title>", 1)
    return svg.strip()


def _recolour(p):
    """Add the adaptive overlay to a figure, panel by panel."""
    if isinstance(p, Compose):
        p.items = [_recolour(item) for item in p.items]
        return p
    return p + _adaptive(p) + _colourbar_guide(p)


def _colourbar_guide(p):
    """Cap the number of bands in a continuous fill legend.

    A colourbar is drawn as one gradient-filled rectangle per band, and in SVG
    each band carries its own gradient definition and pair of stops. At the
    default resolution that is several thousand elements and by far the largest
    thing in a heatmap's markup. Sixty-four bands over a bar an inch or so tall
    are finer than a device pixel, so the bar looks the same and the file is a
    tenth of the size. Returns an empty theme when the plot has no continuous
    fill, so it can be added unconditionally.
    """
    for scale in getattr(p, "scales", []):
        if "fill" in getattr(scale, "aesthetics", []) and isinstance(scale, scale_continuous):
            return guides(fill=guide_colorbar(nbin=64))
    return theme()


def show(p, width: float = 8, height: float = 5, alt: str = "depictr plot") -> str:
    """Return a plotnine figure as inline SVG.

    Parameters
    ----------
    p : plotnine.ggplot or plotnine composition
        The figure to render.
    width, height : float
        Figure size in inches, as passed to ``ggplot.save``.
    alt : str
        The figure's accessible name, written into the SVG's ``<title>``.

    Returns
    -------
    str
        The figure as an inline ``<svg>`` element, ready to be printed.
    """
    global _counter
    _counter += 1
    # Each panel is recoloured on its own terms, since a composition's panels do
    # not all remove the same things: explore_pairs blanks the axis text on its
    # inner panels and keeps it on the edges.
    p = _recolour(p)
    if isinstance(p, Compose):
        # A composition ignores the `width` and `height` given to `save` and
        # would fall back to matplotlib's default canvas, so the size goes
        # through the theme.
        p = p & theme(figure_size=(width, height))
    buf = io.StringIO()
    p.save(buf, format="svg", width=width, height=height, verbose=False,
           transparent=True)
    return _ascii('<div class="dp-figure">'
                  + _adapt(buf.getvalue(), f"d{_counter}-", alt)
                  + "</div>")


def table(df, index: bool = False) -> str:
    """Return a data frame as an HTML table the site's own styling can reach.

    ``DataFrame.to_html`` writes ``<table border="1" class="dataframe">``. The
    class means the table matches neither the brand rule nor Material's, so it
    escapes every table style on the site, keeps a browser-default 1px cell
    padding and, having no overflow rule, scrolls the whole page sideways on a
    narrow viewport. Dropping the attributes outright is independent of the
    pandas version and of ``display.html.use_mathjax``, which adds a second
    class of its own.
    """
    return _ascii(re.sub(r"<table[^>]*>", "<table>", df.to_html(index=index),
                         count=1))


def pre(text: str) -> str:
    """Return text as an escaped code block.

    The ``<code>`` child is what the site's code rules select on, so printed
    output is sized and coloured like the block that produced it rather than
    like body prose.
    """
    import html

    return _ascii("<pre><code>" + html.escape(text) + "</code></pre>")
