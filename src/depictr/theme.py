"""The shared depictr theme and colour/fill scales for plotnine.

Every plotting function builds on :func:`theme_depictr` and the depictr scales,
so figures across the whole workflow share one clean, accessible look without the
caller setting anything. The theme is a light refinement of plotnine's
``theme_minimal``: subtle gridlines, comfortable margins, and centred, bold
titles (including legend titles).
"""

from __future__ import annotations

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

from .palette import NA_VALUE, depictr_brand, depictr_palette


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


def scale_colour_depictr(n: int | None = None, **kwargs):
    """A discrete colour scale drawn from :func:`depictr.palette.depictr_palette`.

    Parameters
    ----------
    n : int, optional
        Number of colours to draw. Defaults to the full qualitative palette;
        pass ``n`` when a plot has more groups than the eight base colours so the
        palette is interpolated to fit.
    **kwargs
        Passed to :func:`plotnine.scale_color_manual` (for example ``name``).
    """
    kwargs.setdefault("na_value", NA_VALUE)
    return scale_color_manual(values=depictr_palette(n), **kwargs)


def scale_fill_depictr(n: int | None = None, **kwargs):
    """A discrete fill scale drawn from :func:`depictr.palette.depictr_palette`.

    See :func:`scale_colour_depictr` for the parameters.
    """
    kwargs.setdefault("na_value", NA_VALUE)
    return scale_fill_manual(values=depictr_palette(n), **kwargs)


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

    Places the legend in a corner the plot's geometry usually leaves empty, over
    a semi-transparent background, so the figure needs no separate legend column.
    The functions that expose a ``legend_inside`` argument add this for you.

    Parameters
    ----------
    corner : {"top right", "top left", "bottom right", "bottom left"} or tuple
        Which corner to use, or an explicit ``((x, y), (jx, jy))`` pair of
        position and justification in axis fractions.

    Returns
    -------
    plotnine.theme
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
