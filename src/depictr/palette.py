"""Colourblind-aware palettes shared by every depictr plot.

The qualitative palette is based on the Okabe-Ito set (Okabe & Ito, 2008), a
categorical scheme that stays distinguishable under the common forms of
colour-vision deficiency. It is led by the depictr brand blue. The sequential
and diverging palettes are perceptually ordered single-hue and red-blue ramps.
These are the defaults behind :func:`depictr.theme.scale_colour_depictr` and the
whole package, so every figure shares one accessible colour language.

The guarantee is a property of the eight built-in colours, not of any number of
colours: past eight, :func:`depictr_palette` has to interpolate between them, and
the interpolated colours sit close enough together to fail
:func:`depictr.cvd.palette_safety` (nine colours come within a CIEDE2000
difference of 4.96 and ten within 1.87, against a threshold of 5). Rather than
let a package built on accessibility make a claim it stops honouring, that case
warns and says what to do instead.

Okabe, M., & Ito, K. (2008). Color universal design (CUD): How to make figures
and presentations that are friendly to colorblind people.
https://jfly.uni-koeln.de/color/
"""

from __future__ import annotations

import math
import numbers
import warnings

import matplotlib.colors as mcolors
import numpy as np

# The qualitative palette, based on the Okabe-Ito colours: the depictr brand blue
# (#005b96) leads in place of their lighter blue so the brand reads first, and a
# grey takes the place of their black.
OKABE_ITO: list[str] = [
    "#005b96",  # blue (brand)
    "#e69f00",  # orange
    "#009e73",  # bluish green
    "#d55e00",  # vermillion (accent)
    "#cc79a7",  # reddish purple
    "#56b4e9",  # sky blue
    "#f0e442",  # yellow
    "#999999",  # grey
]

# The contrast-first order of the same colours. The only three of the eight
# that clear every colour row of check_figure() together lead, then the bluish
# green, which keeps 3:1 against white, then the rest in their default order.
# The R twin's depictr_palette(order = "contrast") gives the same order.
CONTRAST_FIRST: list[str] = [OKABE_ITO[i] for i in (0, 3, 4, 2, 1, 5, 6, 7)]
# How many colours of that order keep 3:1 against white.
_CONTRAST_KEPT = 4

# Anchors for the perceptual ramps, interpolated on demand to any length.
_SEQUENTIAL_ANCHORS = ["#e6eff5", "#4a8fc0", "#005b96", "#08315a"]
_DIVERGING_ANCHORS = ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#005b96"]
# The sequential anchors without the palest, for lines. #e6eff5 suits a fill,
# but at 1.16:1 against white a line drawn in it is close to invisible.
# #4a8fc0 is 3.51:1 and every colour after it darker, so each line clears the
# 3:1 that WCAG 2.2 asks of a graphical object (success criterion 1.4.11).
_CYCLE_ANCHORS = _SEQUENTIAL_ANCHORS[1:]

#: The depictr brand colour (the lead of the qualitative palette).
BRAND = "#005b96"
#: The accent colour, used for a single highlighted series or annotation: the
#: Okabe-Ito vermillion, as in the R twin. At 3.87:1 against white it clears
#: the 3:1 that WCAG 2.2 asks of a graphical object (success criterion 1.4.11).
ACCENT = "#d55e00"
#: The reference grey of the residual smoothers: R's grey60, the default of
#: the R twin's ``depictr_reference()``.
REFERENCE = "#999999"
#: Default colour for missing (``NA``) levels: the hex equivalent of R's
#: grey80, since matplotlib does not know R colour names.
NA_VALUE = "#cccccc"


def depictr_brand() -> str:
    """Return the depictr brand colour as a hex string.

    Examples
    --------
    >>> import depictr as dp
    >>> dp.depictr_brand()
    '#005b96'
    """
    return BRAND


def depictr_accent() -> str:
    """Return the depictr accent colour as a hex string.

    The accent marks a single element against the brand blue, such as the
    flagged points of [`depictr.outlier_plot`][] and the missing cells of
    [`depictr.missingness_map`][]. It is the Okabe-Ito vermillion, as in the
    R twin, which stays distinguishable from the brand blue under the common
    colour-vision deficiencies and reaches 3.87:1 against white.

    Examples
    --------
    >>> import depictr as dp
    >>> dp.depictr_accent()
    '#d55e00'
    """
    return ACCENT


def _ramp(anchors: list[str], n: int) -> list[str]:
    """Interpolate ``n`` evenly spaced colours along a list of anchor colours."""
    cmap = mcolors.LinearSegmentedColormap.from_list("depictr", anchors)
    xs = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
    return [mcolors.to_hex(cmap(x)) for x in xs]


def _cycle_palette(n: int) -> list[str]:
    """Colours for the ``n`` cycle lines of a seasonal plot, earliest first.

    The R twin's internal ``cycle_palette()`` interpolates the same anchors.
    """
    return _ramp(_CYCLE_ANCHORS, n)


def depictr_palette(n: int | None = None, kind: str = "qualitative",
                    order: str = "default") -> list[str]:
    """Return a depictr palette.

    The qualitative palette is based on the Okabe-Ito set and keeps six of its
    eight colours. The depictr brand blue (``#005b96``) takes the place of its
    blue (``#0072b2``) and leads the palette, and a grey (``#999999``) takes the
    place of its black.

    In the default order the second colour is the orange (``#e69f00``), at
    2.25:1 against white. Any figure on white whose colour tells two or more
    groups apart therefore fails the ``geometry_contrast`` row of
    [`depictr.check_figure`][]. No more than three of the eight colours pass
    every colour row of that audit together.

    ``order="contrast"`` puts those three first: the brand blue, the vermillion
    (``#d55e00``) and the reddish purple (``#cc79a7``), then the bluish green
    (``#009e73``) and the other four in their default order. Up to three groups
    then pass every colour row, contrast against white included. They stay at
    least 14.01 apart in CIEDE2000 under every vision type, 6.86 apart in
    lightness and 3.06:1 against white. Four groups keep 3:1 against white,
    but the bluish green and the reddish purple are 3.31 apart in lightness, which
    fails ``greyscale_separability``. Under this order, the second group is
    drawn in the vermillion, which is also [`depictr.depictr_accent`][].
    The default order is unchanged, and the R twin gives the same order.

    The R twin can make the order the default for every plot through its
    options. Python has none, so add
    ``dp.scale_colour_depictr(order="contrast")``, or the fill scale, to each
    plot that should use it. plotnine then says that it replaces the plot's
    colour scale.

    Parameters
    ----------
    n : int, optional
        Number of colours to return, a single number of at least 1. A
        fraction is truncated, as in the R twin, so 2.7 gives two colours. For
        the qualitative palette ``None`` (the default) returns all eight
        colours, and an ``n`` larger than the available colours is
        interpolated. The sequential and diverging palettes are ramps and
        accept any ``n`` (defaulting to 7).
    kind : {"qualitative", "sequential", "diverging"}
        The palette family.
    order : {"default", "contrast"}
        The order of the qualitative palette. ``"contrast"`` gives the
        contrast-first order described above. The ramps have an order of their
        own and refuse ``"contrast"``.

    Returns
    -------
    list of str
        Hex colour codes.

    Raises
    ------
    ValueError
        For an ``n`` below 1 or not a number, an unknown ``kind`` or
        ``order``, or ``order="contrast"`` with a ramp.

    Warns
    -----
    UserWarning
        When ``n`` exceeds the eight built-in colours and the qualitative
        palette is therefore interpolated. The colourblind-safety guarantee
        covers the eight base colours only; an interpolated palette fails
        [`depictr.cvd.palette_safety`][]. Facet the groups, or use the
        sequential ramp, when there are more than eight of them. Also when
        ``order="contrast"`` and ``n`` exceeds four, since the fifth colour,
        the orange, falls below 3:1 against white.

    Examples
    --------
    >>> import depictr as dp
    >>> dp.depictr_palette(3)
    ['#005b96', '#e69f00', '#009e73']
    >>> dp.depictr_palette(3, order="contrast")
    ['#005b96', '#d55e00', '#cc79a7']
    """
    if n is not None:
        # Unchecked, depictr_palette(-1) would give seven colours and (0) none,
        # and an untruncated fraction would fail inside the slicing or the
        # ramp. The R twin's check and message, and, as there, a fraction is
        # truncated.
        if (isinstance(n, bool) or not isinstance(n, numbers.Real)
                or not math.isfinite(n) or n < 1):
            raise ValueError("`n` must be a single positive number.")
        n = int(n)
    if order not in ("default", "contrast"):
        raise ValueError("`order` must be 'default' or 'contrast'.")
    if kind in ("sequential", "diverging") and order != "default":
        raise ValueError(
            f"`order` applies to the qualitative palette, not to the {kind} "
            f"ramp."
        )
    if kind == "sequential":
        return _ramp(_SEQUENTIAL_ANCHORS, n or 7)
    if kind == "diverging":
        return _ramp(_DIVERGING_ANCHORS, n or 7)
    if kind != "qualitative":
        raise ValueError(
            "`kind` must be 'qualitative', 'sequential' or 'diverging'."
        )
    base = CONTRAST_FIRST if order == "contrast" else OKABE_ITO
    if n is None:
        return list(base)
    if order == "contrast" and n > _CONTRAST_KEPT:
        warnings.warn(
            "Past four colours the contrast-first order no longer keeps every "
            "colour at 3:1 against white.",
            stacklevel=2,
        )
    if n <= len(base):
        return base[:n]
    # More categories than base colours: interpolate through the set so groups
    # stay as distinct as the space allows. "As the space allows" is no longer
    # far enough apart to be colourblind-safe, and the package says so rather
    # than handing back colours under a guarantee it cannot keep.
    warnings.warn(
        f"{n} colours interpolated through the {len(OKABE_ITO)}-colour "
        f"Okabe-Ito set: the colour-vision-deficiency guarantee holds only up "
        f"to {len(OKABE_ITO)} categories. Use a sequential palette, or facet "
        f"the groups, if the distinction must survive colour-vision "
        f"deficiency.",
        stacklevel=2,
    )
    return _ramp(base, n)
