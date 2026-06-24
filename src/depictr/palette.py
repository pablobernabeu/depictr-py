"""Colourblind-aware palettes shared by every depictr plot.

The qualitative palette is the Okabe-Ito set (Okabe & Ito, 2008), a categorical
scheme that stays distinguishable under the common forms of colour-vision
deficiency, led here by the depictr brand blue. The sequential and diverging
palettes are perceptually ordered single-hue and red-blue ramps. These are the
defaults behind :func:`depictr.theme.scale_colour_depictr` and the whole package,
so every figure shares one accessible colour language.

Okabe, M., & Ito, K. (2008). Color universal design (CUD): how to make figures
and presentations that are friendly to colorblind people.
https://jfly.uni-koeln.de/color/
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np

# The qualitative palette: the Okabe-Ito colours, led by the depictr brand blue
# (#005b96) in place of the lighter standard blue so the brand reads first.
OKABE_ITO: list[str] = [
    "#005b96",  # blue (brand)
    "#e69f00",  # orange (accent)
    "#009e73",  # bluish green
    "#d55e00",  # vermillion
    "#cc79a7",  # reddish purple
    "#56b4e9",  # sky blue
    "#f0e442",  # yellow
    "#999999",  # grey
]

# Anchors for the perceptual ramps, interpolated on demand to any length.
_SEQUENTIAL_ANCHORS = ["#e6eff5", "#4a8fc0", "#005b96", "#08315a"]
_DIVERGING_ANCHORS = ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#005b96"]

#: The depictr brand colour (the lead of the qualitative palette).
BRAND = "#005b96"
#: The accent colour, used for a single highlighted series or annotation.
ACCENT = "#e69f00"
#: Default colour for missing (``NA``) levels.
NA_VALUE = "grey80"


def depictr_brand() -> str:
    """Return the depictr brand colour as a hex string."""
    return BRAND


def depictr_accent() -> str:
    """Return the depictr accent colour as a hex string."""
    return ACCENT


def _ramp(anchors: list[str], n: int) -> list[str]:
    """Interpolate ``n`` evenly spaced colours along a list of anchor colours."""
    cmap = mcolors.LinearSegmentedColormap.from_list("depictr", anchors)
    xs = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
    return [mcolors.to_hex(cmap(x)) for x in xs]


def depictr_palette(n: int | None = None, kind: str = "qualitative") -> list[str]:
    """Return a depictr palette.

    Parameters
    ----------
    n : int, optional
        Number of colours to return. For the qualitative palette ``None`` (the
        default) returns the full Okabe-Ito set, and an ``n`` larger than the
        available colours is interpolated. The sequential and diverging palettes
        are ramps and accept any ``n`` (defaulting to 7).
    kind : {"qualitative", "sequential", "diverging"}
        The palette family.

    Returns
    -------
    list of str
        Hex colour codes.
    """
    if kind == "sequential":
        return _ramp(_SEQUENTIAL_ANCHORS, n or 7)
    if kind == "diverging":
        return _ramp(_DIVERGING_ANCHORS, n or 7)
    if kind != "qualitative":
        raise ValueError(
            "`kind` must be 'qualitative', 'sequential' or 'diverging'."
        )
    if n is None:
        return list(OKABE_ITO)
    if n <= len(OKABE_ITO):
        return OKABE_ITO[:n]
    # More categories than base colours: interpolate through the set so groups
    # stay as distinct as the space allows.
    return _ramp(OKABE_ITO, n)
