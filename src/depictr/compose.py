"""Saving and composition helpers."""

from __future__ import annotations

import warnings
from functools import reduce


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
    """
    plot.save(filename, width=width, height=height, dpi=dpi, units=units,
              verbose=False, **kwargs)
    return filename
