"""Saving and composition helpers."""

from __future__ import annotations

import copy
import io
import os
import warnings
from functools import reduce

# Fonts in a PDF or PostScript file embedded as TrueType (Type 42).
# matplotlib's default, Type 3, is the kind Nature's figure guide refuses.
_EMBEDDED_FONTS = {"pdf.fonttype": 42, "ps.fonttype": 42}


def arrange_plots(*plots, ncol: int | None = None, nrow: int | None = None,
                  title: str | None = None):
    """Arrange several plots into one composed figure.

    Builds a grid by joining plots side by side (plotnine's ``|``) into rows of
    ``ncol``, then stacking the rows (``/``). The result is a plotnine
    composition, so it still has ``.draw`` and ``.save``. This is the building
    block behind the multi-panel reports (for example
    :func:`depictr.diagnostics.residual_diagnostics_plot`).

    Parameters
    ----------
    *plots : plotnine.ggplot
        The plots to arrange. ``None`` entries are dropped.
    ncol, nrow : int, optional
        The grid shape. Give one and the other is derived from the count; give
        neither and a sensible default is chosen (one column for a single plot,
        two up to four plots, three beyond).
    title : str, optional
        A title. Applied only when a single plot is passed: plotnine
        compositions cannot carry a figure-level title, so for a grid the
        title is dropped with a warning, and each panel should carry its own.

    Returns
    -------
    plotnine.ggplot or plotnine.composition.Compose
        A single plot when one is passed, otherwise a composition.

    Examples
    --------
    >>> import depictr as dp
    >>> ld = dp.lexical_decision()
    >>> left = dp.ecdf_plot(ld, "RT")
    >>> right = dp.explore_distribution(ld, "RT")
    >>> panel = dp.arrange_plots(left, right, ncol=2)
    """
    from plotnine import ggplot, ggtitle

    items = [p for p in plots if p is not None]
    if not items:
        raise ValueError("arrange_plots needs at least one plot.")
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


def save_plot(plot, filename, width: float = 7, height: float = 4.5,
              dpi: int = 300, units: str = "in", **kwargs) -> str:
    """Save a depictr/plotnine plot at publication resolution.

    Missing directories on the way to ``filename`` are created, and PDF, EPS
    and TIFF files are written to pass the technical checks journals run on
    submitted figures.

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
    plot : plotnine.ggplot
        The plot to save.
    filename : str
        Output path; the extension sets the format (``.png``, ``.pdf``,
        ``.eps``, ``.tiff``, ``.svg``).
    width, height : float
        Figure size, in ``units``.
    dpi : int
        Dots per inch (300 suits print).
    units : str
        Size units (``"in"``, ``"cm"`` or ``"mm"``).
    **kwargs
        Passed to :meth:`plotnine.ggplot.save`, and from there to matplotlib's
        ``savefig()``.

    Returns
    -------
    str
        The filename written.

    Warns
    -----
    UserWarning
        When an EPS or PostScript file is saved from a plot with translucent
        layers.

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
    """
    import matplotlib as mpl

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
    return filename


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
