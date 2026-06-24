"""Saving and composition helpers."""

from __future__ import annotations


def save_plot(plot, filename, width: float = 7, height: float = 4.5,
              dpi: int = 300, units: str = "in", **kwargs) -> str:
    """Save a depictr/plotnine plot at publication resolution.

    Parameters
    ----------
    plot : plotnine.ggplot
        The plot to save.
    filename : str
        Output path; the extension sets the format (``.png``, ``.pdf``, ``.svg``).
    width, height : float
        Figure size, in ``units``.
    dpi : int
        Dots per inch (300 suits print).
    units : str
        Size units (``"in"``, ``"cm"`` or ``"mm"``).
    **kwargs
        Passed to :meth:`plotnine.ggplot.save`.

    Returns
    -------
    str
        The filename written.
    """
    plot.save(filename, width=width, height=height, dpi=dpi, units=units,
              verbose=False, **kwargs)
    return filename
