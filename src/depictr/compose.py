"""Saving and composition helpers."""

from __future__ import annotations

import copy
import io
import math
import numbers
import os
import warnings
from functools import reduce

# Fonts in a PDF or PostScript file embedded as TrueType (Type 42).
# matplotlib's default, Type 3, is the kind Nature's figure guide refuses.
_EMBEDDED_FONTS = {"pdf.fonttype": 42, "ps.fonttype": 42}

# Centimetres in one of each unit that plotnine's save() takes a size in.
_CM_PER_UNIT = {"in": 2.54, "cm": 1.0, "mm": 0.1}


def arrange_plots(*plots, ncol: int | None = None, nrow: int | None = None,
                  title: str | None = None, tag_levels="A",
                  tag_size: float | None = None):
    """Arrange several plots into one composed figure.

    Builds a grid by joining plots side by side (plotnine's ``|``) into rows of
    ``ncol``, then stacking the rows (``/``). The result is a plotnine
    composition, so it still has ``.draw`` and ``.save``. This is the building
    block behind the multi-panel reports (for example
    :func:`depictr.residual_diagnostics_plot`).

    Each panel is tagged A, B, C and so on by default, in reading order, as
    the R twin's patchwork annotation tags them. The tags run on through a
    composition passed as one of the plots, and a ``plot_spacer()`` takes
    none. A tag is drawn in the top-left corner of its panel at the theme's
    ``plot_tag`` size, 1.2 times the base size, as in ggplot2. In this twin
    each tag belongs to its panel's plot, so :func:`depictr.check_figure`
    measures it with the rest of the panel's text, where the R twin's audit
    leaves out the tags that patchwork draws around the panels.

    Parameters
    ----------
    *plots : plotnine.ggplot or plotnine.composition.Compose
        The plots to arrange. ``None`` entries are dropped.
    ncol, nrow : int, optional
        The grid shape. Give one and the other is derived from the count; give
        neither and a sensible default is chosen (one column for a single plot,
        two up to four plots, three beyond).
    title : str, optional
        A title. Applied only when a single plot is passed: plotnine
        compositions cannot carry a figure-level title, so for a grid the
        title is dropped with a warning, and each panel should carry its own.
    tag_levels : {"A", "a", "1", "I", "i"}, sequence of str or None
        How to tag the panels: capital or lower-case letters, Arabic numerals,
        or capital or lower-case Roman numerals, as patchwork's
        ``tag_levels`` takes them. A sequence of strings gives the tags
        themselves, one per panel in reading order. ``None`` leaves the panels
        untagged, as before this argument existed. Letters run to 26 panels.
    tag_size : float, optional
        The size of the tags in points, such as the 8 pt that Nature asks of
        panel labels. ``None``, the default, leaves it to each panel's theme.
        Ignored when ``tag_levels`` is None.

    Returns
    -------
    plotnine.ggplot or plotnine.composition.Compose
        A single plot when one is passed, otherwise a composition. The plots
        passed in are left untagged.

    Examples
    --------
    >>> import depictr as dp
    >>> ld = dp.lexical_decision()
    >>> left = dp.ecdf_plot(ld, "RT")
    >>> right = dp.explore_distribution(ld, "RT")
    >>> panel = dp.arrange_plots(left, right, ncol=2)

    Lower-case tags at 8 pt, or none at all:

    >>> panel = dp.arrange_plots(left, right, tag_levels="a", tag_size=8)
    >>> panel = dp.arrange_plots(left, right, tag_levels=None)
    """
    from plotnine import ggplot, ggtitle

    tag_style = _check_tag_levels(tag_levels)
    if tag_size is not None and (
            isinstance(tag_size, bool) or not isinstance(tag_size, numbers.Real)
            or not math.isfinite(tag_size) or tag_size <= 0):
        raise ValueError("`tag_size` must be a single positive number.")
    items = [p for p in plots if p is not None]
    if not items:
        raise ValueError("arrange_plots needs at least one plot.")
    if tag_style is not None:
        items = _tagged(items, tag_style, tag_size)
    n = len(items)
    if ncol is None and nrow is None:
        ncol = 1 if n == 1 else (2 if n <= 4 else 3)
    elif ncol is None:
        ncol = -(-n // nrow)  # ceil division
    rows = [reduce(lambda a, b: a | b, items[i:i + ncol])
            for i in range(0, n, ncol)]
    composed = reduce(lambda a, b: a / b, rows)
    # plotnine has no super-title for a composition; adding one would land on the
    # last panel. So only title a lone plot, and let grids self-title per panel.
    if title is not None:
        if isinstance(composed, ggplot):
            composed = composed + ggtitle(title)
        else:
            warnings.warn(
                "plotnine compositions cannot carry a figure-level title, so "
                "`title` is dropped for a multi-panel grid. Title the panels "
                "individually instead.",
                UserWarning,
                stacklevel=2,
            )
    return composed


# The tag styles patchwork's tag_levels takes, which arrange_plots() takes too.
_TAG_STYLES = ("A", "a", "1", "I", "i")

_TAG_LEVELS_MESSAGE = (
    "`tag_levels` must be 'A', 'a', '1', 'I' or 'i', a sequence of tags, or "
    "None.")


def _check_tag_levels(tag_levels):
    """A tag style, a tuple of custom tags, or None, from ``tag_levels``."""
    if tag_levels is None:
        return None
    if isinstance(tag_levels, str):
        if tag_levels not in _TAG_STYLES:
            raise ValueError(_TAG_LEVELS_MESSAGE)
        return tag_levels
    try:
        tags = tuple(tag_levels)
    except TypeError:
        raise ValueError(_TAG_LEVELS_MESSAGE) from None
    if not all(isinstance(tag, str) for tag in tags):
        raise ValueError(_TAG_LEVELS_MESSAGE)
    return tags


def _roman(number: int) -> str:
    """A positive whole number in capital Roman numerals."""
    numerals = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
                (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
                (5, "V"), (4, "IV"), (1, "I"))
    out = []
    for value, numeral in numerals:
        count, number = divmod(number, value)
        out.append(numeral * count)
    return "".join(out)


def _tag_sequence(style, n: int) -> list[str]:
    """The first ``n`` tags of a style, or the custom tags given."""
    if isinstance(style, tuple):
        if len(style) < n:
            raise ValueError(
                f"`tag_levels` gives {len(style)} tags for {n} panels.")
        return list(style[:n])
    if style in ("A", "a"):
        if n > 26:
            raise ValueError(
                f"`tag_levels = '{style}'` has 26 letters for {n} panels. "
                "Pass '1' or a sequence of tags.")
        start = ord(style)
        return [chr(start + i) for i in range(n)]
    if style == "1":
        return [str(i) for i in range(1, n + 1)]
    numerals = [_roman(i) for i in range(1, n + 1)]
    return numerals if style == "I" else [r.lower() for r in numerals]


def _tag_panels(plot):
    """The tag-taking panels of a plot or composition, depth-first.

    Yields each panel with the composition holding it and its index there, or
    with None for a lone plot. A ``plot_spacer()`` is left out, as patchwork
    gives a spacer no tag and :func:`depictr.check_figure` gives it no panel
    number.
    """
    from plotnine.composition import Compose, plot_spacer

    if isinstance(plot, Compose):
        for index, item in enumerate(plot):
            if isinstance(item, Compose):
                yield from _tag_panels(item)
            elif not isinstance(item, plot_spacer):
                yield plot, index, item
    elif not isinstance(plot, plot_spacer):
        yield None, None, plot


def _tagged(items, style, size):
    """Copies of the plots to arrange, each panel carrying its tag.

    A composition among them is copied and its panels tagged in place, through
    the item assignment plotnine's own ``&`` operator uses, so the tags run on
    through it in reading order. The caller's plots are left as they were.
    """
    from plotnine import element_text, labs, theme
    from plotnine.composition import Compose

    items = [copy.deepcopy(item) if isinstance(item, Compose) else item
             for item in items]
    slots = [(i, holder, index, panel)
             for i, item in enumerate(items)
             for holder, index, panel in _tag_panels(item)]
    tags = _tag_sequence(style, len(slots))
    for (i, holder, index, panel), tag in zip(slots, tags, strict=True):
        tagged = panel + labs(tag=tag)
        if size is not None:
            tagged = tagged + theme(plot_tag=element_text(size=size))
        if holder is None:
            items[i] = tagged
        else:
            holder[index] = tagged
    return items


def save_plot(plot, filename, width: float = 7, height: float = 4.5,
              dpi: int = 300, units: str = "in", check: bool = False,
              **kwargs) -> str:
    """Save a depictr/plotnine plot at publication resolution.

    Missing directories on the way to ``filename`` are created, and PDF, EPS
    and TIFF files are written to pass the technical checks journals run on
    submitted figures. With ``check=True``, the figure is also audited with
    :func:`depictr.check_figure` at the width it is saved at.

    A PDF or EPS embeds its fonts as TrueType (Type 42). matplotlib's default,
    Type 3, is the kind Nature refuses, and the setting holds for this save
    only. PostScript has no transparency, so matplotlib draws translucent
    layers opaque in an EPS, where they hide what lies beneath them, and a
    warning says how many layers that affects. A PDF keeps them as they are.

    A TIFF is written as LZW-compressed RGB at ``dpi``, with any transparent
    background composited onto white. matplotlib alone writes uncompressed
    RGBA, which PLOS refuses on both counts: the figure in the example below
    takes 10.8 MiB that way. Passing ``pil_kwargs`` leaves the TIFF to
    matplotlib as it was.

    Which formats three publishers accepted on 2 October 2026:

    | Format | PLOS | Nature (final artwork) | Elsevier |
    |:---|:---|:---|:---|
    | PDF | No | Yes, as vector art | Yes |
    | EPS | Yes, with fonts embedded | Yes, as vector art | Yes, with fonts embedded |
    | TIFF | Yes, LZW-compressed RGB or greyscale, no alpha channel, at most 10 MB | No | Yes |

    Nature also asks for fonts embedded as TrueType 2 or 42, never Type 3, and
    for line art and text that are not rasterised. A figure with translucent
    layers therefore goes to Nature as a PDF. Elsevier asks for a graph sent as
    a TIFF at 1000 dpi, so pass ``dpi=1000`` for one.

    The R twin takes the file name first, as ``save_plot(filename, plot)``. For
    an EPS, it rasterises the translucent layers and whatever they overlap,
    where matplotlib draws them opaque.

    PLOS ONE. Figures. https://journals.plos.org/plosone/s/figures

    Nature. Guide to preparing final artwork.
    https://www.nature.com/documents/nature-final-artwork.pdf

    Nature research figure guide. Preparing figures: our specifications.
    https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/

    Elsevier. Artwork overview.
    https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-overview

    Parameters
    ----------
    plot : plotnine.ggplot or plotnine.composition.Compose
        The plot to save, or a composition of several.
    filename : str
        Output path; the extension sets the format (``.png``, ``.pdf``,
        ``.eps``, ``.tiff``, ``.svg``).
    width, height : float
        Figure size, in ``units``. A composition is saved at this size too.
    dpi : int
        Dots per inch (300 suits print).
    units : str
        Size units (``"in"``, ``"cm"`` or ``"mm"``).
    check : bool
        Whether to audit the figure with :func:`depictr.check_figure` at the
        width it is saved at, given as both ``width_cm`` and
        ``render_width_cm``, so that the text sizes it reports are the printed
        ones. A warning names each check the figure fails. Defaults to False.
        The audit runs before the file is written, so a plot it cannot read is
        refused without saving. For the full report, or for other thresholds
        such as a journal's own text floor, call :func:`depictr.check_figure`
        with the saved width as both widths.
    **kwargs
        Passed to :meth:`plotnine.ggplot.save`, and from there to matplotlib's
        ``savefig()``.

    Returns
    -------
    str
        The filename written. With ``check=True``, the R twin attaches the
        audit report to it, and this twin returns the filename alone.

    Warns
    -----
    UserWarning
        When an EPS or PostScript file is saved from a plot with translucent
        layers, and, with ``check=True``, when the figure fails a check.

    Examples
    --------
    >>> import os, tempfile
    >>> import depictr as dp
    >>> ld = dp.lexical_decision()
    >>> p = dp.ecdf_plot(ld, "RT")
    >>> path = dp.save_plot(p, os.path.join(tempfile.mkdtemp(), "rt.png"),
    ...                     width=4, height=3, dpi=72)
    >>> os.path.exists(path)
    True

    A TIFF well within the 10 MB that PLOS allows, in a folder that does not
    yet exist.

    >>> tiff = os.path.join(tempfile.mkdtemp(), "figures", "rt.tiff")
    >>> path = dp.save_plot(p, tiff)
    >>> os.path.getsize(path) < 1_000_000
    True

    Audited at the 8.9 cm column it is saved for, where its text prints at the
    size it is drawn. It passes every check that applies, so nothing is said.

    >>> column = os.path.join(tempfile.mkdtemp(), "rt-column.png")
    >>> path = dp.save_plot(p, column, width=8.9, height=6, units="cm",
    ...                     dpi=72, check=True)
    """
    import matplotlib as mpl
    from plotnine import theme
    from plotnine.composition import Compose

    if not isinstance(check, bool):
        raise TypeError("`check` must be True or False.")
    # The audit runs before anything is written, so a figure it cannot read,
    # or a width it cannot convert, leaves no file behind.
    report = saved_cm = None
    if check:
        from .audit import check_figure

        saved_cm = _saved_width_cm(width, units)
        report = check_figure(plot, width_cm=saved_cm,
                              render_width_cm=saved_cm)

    # plotnine draws a composition at the figure size of its last plot's theme
    # and ignores the width and height its save() is given, so the size is
    # written into that theme. Adding to a composition copies it, which leaves
    # the caller's untouched.
    if isinstance(plot, Compose) and width is not None and height is not None:
        plot = plot + theme(figure_size=(_inches(width, units),
                                         _inches(height, units)))

    # plotnine joins `path` and the file name itself. The joined path is needed
    # here first, for the directory and for a TIFF written outside matplotlib.
    path = kwargs.pop("path", None)
    target = filename
    if path and isinstance(filename, (str, os.PathLike)):
        target = os.path.join(path, filename)
    fmt = _save_format(target, kwargs.get("format"))
    if isinstance(target, (str, os.PathLike)):
        os.makedirs(os.path.dirname(os.fspath(target)) or ".", exist_ok=True)

    with mpl.rc_context(_EMBEDDED_FONTS):
        if fmt in ("tif", "tiff") and "pil_kwargs" not in kwargs:
            _save_tiff(plot, target, width, height, dpi, units, kwargs)
        else:
            _with_embedded_fonts(plot).save(
                target, width=width, height=height, dpi=dpi, units=units,
                verbose=False, **kwargs)

    if fmt in ("eps", "ps"):
        count = _translucent_layers(plot)
        if count:
            kind = "EPS" if fmt == "eps" else "PostScript"
            layers = "layer" if count == 1 else "layers"
            warnings.warn(
                f"{kind} has no transparency, so matplotlib draws {count} "
                f"translucent {layers} opaque. Save as PDF, or as TIFF at "
                "300-600 dpi, to keep them.",
                UserWarning,
                stacklevel=2,
            )
    if report is not None:
        failed = _failed_checks(report)
        if failed:
            noun = "check" if len(failed) == 1 else "checks"
            warnings.warn(
                f"The figure fails {len(failed)} {noun} at the "
                f"{saved_cm:.2f} cm it is saved at: {_and_list(failed)}. "
                "Pass that width to check_figure() as both width_cm and "
                "render_width_cm for the full report.",
                UserWarning,
                stacklevel=2,
            )
    return filename


def _saved_width_cm(width, units) -> float:
    """The width a figure is saved at, in centimetres.

    Raises ValueError when it is not a single positive number in one of the
    units plotnine's save() takes.
    """
    if (isinstance(width, bool) or not isinstance(width, numbers.Real)
            or not math.isfinite(width) or width <= 0
            or not isinstance(units, str) or units not in _CM_PER_UNIT):
        raise ValueError(
            "`check=True` needs the width the figure is saved at: a single "
            'positive number in "in", "cm" or "mm".')
    return float(width) * _CM_PER_UNIT[units]


def _inches(value, units) -> float:
    """A size in inches, from any unit plotnine's save() takes."""
    if units not in _CM_PER_UNIT:
        raise ValueError(f"Unknown units '{units}'")
    return value * _CM_PER_UNIT[units] / 2.54


def _failed_checks(report) -> list[str]:
    """Each check a report fails, with what it measured against its threshold.

    In a composition's report, each names the panel it is in. A check with
    nothing to measure is not a failure, so it is not named.
    """
    failed = report[report["verdict"] == "fail"]
    return [
        (f"panel {row['panel']} " if "panel" in row else "")
        + f"{row['check']} {row['measured']:.2f} "
        f"(threshold {row['threshold']:.2f})"
        for row in failed.to_dict("records")
    ]


def _and_list(items: list[str]) -> str:
    """Join items as "a, b and c"."""
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _with_embedded_fonts(plot):
    """The plot to save, with a theme that cannot undo the TrueType setting.

    plotnine saves a plot inside its theme's rcParams, which win over the
    surrounding rc_context. A theme that copies matplotlib's own, as
    theme_matplotlib() does, carries the Type 3 default with it, so the plot is
    saved from a copy whose theme carries the embedded font types. A
    composition is saved outside its panels' rcParams and needs no copy.
    """
    from plotnine import ggplot

    if not isinstance(plot, ggplot):
        return plot
    rc = getattr(plot.theme, "_rcParams", None)
    if not isinstance(rc, dict) or all(
        rc.get(key, value) == value for key, value in _EMBEDDED_FONTS.items()
    ):
        return plot
    plot = copy.deepcopy(plot)
    plot.theme._rcParams.update(_EMBEDDED_FONTS)
    return plot


def _save_format(filename, fmt) -> str:
    """The format a save writes: ``format`` when given, else the extension."""
    if fmt:
        return str(fmt).lower().lstrip(".")
    if isinstance(filename, (str, os.PathLike)):
        return os.path.splitext(os.fspath(filename))[1].lower().lstrip(".")
    return ""


def _save_tiff(plot, target, width, height, dpi, units, kwargs) -> None:
    """Write a TIFF as PLOS asks for one: LZW-compressed RGB with no alpha.

    matplotlib renders into an RGBA buffer. Converting that straight to RGB
    would drop the alpha channel and leave transparent pixels black, so the
    buffer is first composited onto white, as a transparent background prints.
    """
    from PIL import Image

    options = {key: value for key, value in kwargs.items() if key != "format"}
    buffer = io.BytesIO()
    plot.save(buffer, format="tiff", width=width, height=height, dpi=dpi,
              units=units, verbose=False, **options)
    buffer.seek(0)
    with Image.open(buffer) as image:
        rgba = image.convert("RGBA")
        resolution = (dpi, dpi) if dpi else image.info.get("dpi")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, rgba).convert("RGB")
    extra = {"dpi": resolution} if resolution else {}
    rgb.save(target, format="TIFF", compression="tiff_lzw", **extra)


def _translucent_layers(plot) -> int:
    """How many layers draw with an alpha below 1, over a composition's panels.

    The layers' alpha is known only once the plot is built, so a copy is drawn,
    since drawing builds the layers in place. Its warnings are muffled because
    the save has already shown them.
    """
    import matplotlib.pyplot as plt
    from plotnine import ggplot

    if not isinstance(plot, ggplot):
        return sum(_translucent_layers(item) for item in getattr(plot, "items", ()))
    built = copy.deepcopy(plot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure = built.draw(show=False)
    plt.close(figure)
    return sum(_layer_is_translucent(layer) for layer in built.layers)


def _layer_is_translucent(layer) -> bool:
    """Whether a built layer draws anything with an alpha strictly between 0 and 1.

    A smoothing layer applies its alpha only to its confidence band, so without
    the band, as under ``se=False``, its alpha describes nothing drawn.
    """
    import matplotlib.colors as mcolors
    import pandas as pd
    from plotnine import geom_smooth

    data = layer.data
    check_alpha = "alpha" in data
    if isinstance(layer.geom, geom_smooth):
        band = {"ymin", "ymax"}.issubset(data.columns)
        check_alpha = check_alpha and bool(layer.geom.params.get("se", True)) and band
    if check_alpha:
        alpha = pd.to_numeric(data["alpha"], errors="coerce")
        if ((alpha > 0) & (alpha < 1)).any():
            return True
    for aesthetic in ("color", "fill"):
        if aesthetic not in data:
            continue
        for colour in data[aesthetic].dropna().unique():
            try:
                opacity = mcolors.to_rgba(colour)[3]
            except (ValueError, TypeError):
                continue
            if 0 < opacity < 1:
                return True
    return False
