"""The shared depictr theme and colour/fill scales for plotnine.

Every plotting function builds on :func:`theme_depictr` and the depictr scales,
so figures across the whole workflow share one consistent, accessible look
without the caller setting anything. The theme is a light refinement of
plotnine's ``theme_minimal``: subtle gridlines, comfortable margins, and centred,
bold titles (including legend titles).
"""

from __future__ import annotations

import copy

import numpy as np
from plotnine import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
)

from .palette import (
    _CONTRAST_KEPT,
    NA_VALUE,
    OKABE_ITO,
    depictr_brand,
    depictr_palette,
)


def theme_depictr(base_size: float = 11, grid: str = "xy") -> theme:
    """Return the depictr plotnine theme.

    Parameters
    ----------
    base_size : float
        Base font size in points.
    grid : {"xy", "x", "y", "none"}
        Which major gridlines to keep.

    Returns
    -------
    plotnine.theme
        A theme object to add to any plot with ``+``.

    Examples
    --------
    >>> import depictr as dp
    >>> ld = dp.lexical_decision()
    >>> p = dp.ecdf_plot(ld, "RT") + dp.theme_depictr(base_size=13, grid="y")
    """
    if grid not in {"xy", "x", "y", "none"}:
        raise ValueError("`grid` must be 'xy', 'x', 'y' or 'none'.")

    th = theme_minimal(base_size=base_size) + theme(
        plot_title=element_text(
            color=depictr_brand(), weight="bold", ha="center",
            size=base_size * 1.15, margin={"b": base_size * 0.5},
        ),
        plot_subtitle=element_text(color="#4d4d4d", ha="center",
                                   margin={"b": base_size * 0.4}),
        panel_grid_minor=element_blank(),
        panel_grid_major=element_line(color="#e6e6e6", size=0.4),
        # Bold, centred legend titles, matching the depictr R defaults. The
        # bottom margin lifts the title clear of the first key so it reads as a
        # heading rather than crowding the labels.
        legend_title=element_text(weight="bold", ha="center", margin={"b": 6}),
        # A white key with a thin border so a light-coloured swatch (e.g. a pale
        # grey "Present" tile) stays delineated wherever a legend appears.
        legend_key=element_rect(fill="#ffffff", color="#cccccc", size=0.3),
        legend_key_spacing_y=4,
        strip_background=element_rect(fill="#f5f5f5", color="none"),
        strip_text=element_text(size=base_size),
    )

    if grid in {"x", "none"}:
        th += theme(panel_grid_major_y=element_blank())
    if grid in {"y", "none"}:
        th += theme(panel_grid_major_x=element_blank())
    return th


def _sized_to_the_data(scale, n, order="default"):
    """Let a depictr manual scale size the palette to the levels it is given.

    A manual scale holds a fixed list of values, so plotnine draws every level
    past the eighth in the missing-value colour. With ``n`` unset, the palette
    here is asked for as many colours as the plot has levels, as in the R twin's
    ``function(k) depictr_palette(n %||% k)``. A ninth group is then
    interpolated, and :func:`depictr.palette.depictr_palette` warns, as it
    does for a fifth group in the contrast-first order. Up to eight levels,
    four in that order, and whenever ``n`` is given, plotnine's own palette is
    kept, so those figures are drawn exactly as before.
    """
    if n is not None:
        return scale
    manual = scale.palette
    kept = _CONTRAST_KEPT if order == "contrast" else len(OKABE_ITO)

    def palette(k):
        if k > kept:
            return depictr_palette(k, order=order)
        return manual(k)

    scale.palette = palette
    return scale


def _scale_fills(scale, values) -> list[str]:
    """The colours a continuous fill scale paints ``values`` in.

    A copy of the scale is trained on the values and maps them, as plotnine
    does when it draws the figure. Each colour therefore comes from the scale's
    own limits, rescaling, out-of-bounds rule and palette (mizani's
    ``gradient_n_pal`` for ``scale_fill_gradientn``), with its ``na_value``
    where a value cannot be mapped. A label colour chosen against these is
    chosen against the tile the reader sees.
    """
    probe = copy.deepcopy(scale)
    x = np.asarray(values, dtype=float)
    probe.train(x)
    return [str(c) for c in probe.map(x)]


def scale_colour_depictr(n: int | None = None, order: str = "default",
                         **kwargs):
    """A discrete colour scale drawn from :func:`depictr.palette.depictr_palette`.

    Parameters
    ----------
    n : int, optional
        Number of colours to draw. By default, the scale takes as many colours
        as the plot has groups: the Okabe-Ito colours in order up to eight,
        and past eight a palette interpolated through them. Interpolation loses
        the colourblind-safety guarantee, which covers the eight base colours
        only, so :func:`depictr.palette.depictr_palette` warns past eight; facet
        the groups, or use the sequential ramp, when there are more. Pass ``n``
        to fix the palette's length whatever the number of groups.
    order : {"default", "contrast"}
        ``"contrast"`` draws the groups in the contrast-first order of
        :func:`depictr.palette.depictr_palette`, with which two or three
        groups pass every colour row of :func:`depictr.check_figure`. More than
        four groups warn when the plot is drawn. Added to one of depictr's own
        plots, the scale replaces the plot's colour scale.
    **kwargs
        Passed to :func:`plotnine.scale_color_manual` (for example ``name``).

    Examples
    --------
    >>> import depictr as dp
    >>> from plotnine import aes, geom_point, ggplot
    >>> cy = dp.crop_yield()
    >>> p = (ggplot(cy, aes("rainfall", "yield", colour="treatment"))
    ...      + geom_point() + dp.scale_colour_depictr())

    The vermillion in the orange's place, at 3.87:1 against white:

    >>> p = (ggplot(cy, aes("rainfall", "yield", colour="treatment",
    ...                     shape="treatment"))
    ...      + geom_point() + dp.scale_colour_depictr(order="contrast"))
    """
    kwargs.setdefault("na_value", NA_VALUE)
    return _sized_to_the_data(
        scale_color_manual(values=depictr_palette(n, order=order), **kwargs),
        n, order)


def scale_fill_depictr(n: int | None = None, order: str = "default", **kwargs):
    """A discrete fill scale drawn from :func:`depictr.palette.depictr_palette`.

    See :func:`scale_colour_depictr` for the parameters.

    Examples
    --------
    >>> import depictr as dp
    >>> from plotnine import aes, geom_bar, ggplot
    >>> wb = dp.wellbeing_survey()
    >>> p = (ggplot(wb, aes("education", fill="region"))
    ...      + geom_bar() + dp.scale_fill_depictr())
    """
    kwargs.setdefault("na_value", NA_VALUE)
    return _sized_to_the_data(
        scale_fill_manual(values=depictr_palette(n, order=order), **kwargs),
        n, order)


# American-spelling alias, matching the convention plotnine and matplotlib use.
scale_color_depictr = scale_colour_depictr


# Anchor positions for an inside legend, by corner.
_LEGEND_CORNERS = {
    "top right": ((0.98, 0.98), (1, 1)),
    "top left": ((0.02, 0.98), (0, 1)),
    "bottom right": ((0.98, 0.03), (1, 0)),
    "bottom left": ((0.02, 0.03), (0, 0)),
}


def legend_inside(corner="top right"):
    """Theme fragment that moves the legend into the plotting area.

    Places the legend in a corner the plot's geometry usually leaves empty, on
    a solid white panel with a light border, so the figure needs no separate
    legend column.
    The functions that expose a ``legend_inside`` argument add this for you.

    Parameters
    ----------
    corner : {"top right", "top left", "bottom right", "bottom left"} or tuple
        Which corner to use, or an explicit ``((x, y), (jx, jy))`` pair of
        position and justification in axis fractions.

    Returns
    -------
    plotnine.theme

    Examples
    --------
    >>> import depictr as dp
    >>> ld = dp.lexical_decision()
    >>> p = dp.ecdf_plot(ld, "RT", group="condition") + dp.legend_inside("bottom right")
    """
    if isinstance(corner, str):
        if corner not in _LEGEND_CORNERS:
            raise ValueError(f"`corner` must be one of {list(_LEGEND_CORNERS)}.")
        position, justification = _LEGEND_CORNERS[corner]
    else:
        position, justification = corner
    return theme(
        legend_position=position,
        legend_justification=justification,
        # A solid white panel (not translucent) so the legend stays legible even
        # over a filled plot area, with a light border and delineated keys.
        legend_background=element_rect(fill="#ffffff", color="#cccccc", size=0.5),
        legend_key=element_rect(fill="#ffffff", color="#cccccc", size=0.3),
    )
