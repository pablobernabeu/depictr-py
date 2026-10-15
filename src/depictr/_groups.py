"""One rule for the grouping variable of every grouped plot.

The grouped plots used to treat a grouping column in as many ways as there were
functions. A numeric code raised in some and was silently ungrouped in others.
A missing group was drawn as a grey group of its own or vanished without a
word. An unused category kept its slot in the palette, so the levels after it
took other colours than in the R twin. :func:`_prepare_group` applies the R
twin's ``prepare_group()`` rule in every one of them, so the same data give the
same groups and colours in every function and in both packages.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_numeric_dtype
from plotnine import scale_color_manual, scale_fill_manual

from .palette import NA_VALUE, depictr_palette

#: A numeric group with more distinct values than this is taken for a
#: continuous variable, the threshold the R twin's explore_categorical()
#: applies to its ``x``.
_MAX_NUMERIC_GROUPS = 20


def _whole_numbers(values):
    """Whether every value is a whole number that a 64-bit integer holds exactly."""
    v = values.to_numpy(dtype=float)
    return bool(np.isfinite(v).all() and (v == np.round(v)).all()
                and (np.abs(v) < 2.0**53).all())


def _prepare_group(data, group):
    """Prepare the grouping column of a grouped plot.

    The group becomes a categorical, so a numeric code is a group like any
    other. A numeric column with more than 20 distinct values raises, though,
    since one group per value of a continuous variable is never what was
    meant. Rows whose group is missing belong to no group, and they are dropped
    with a warning in :func:`depictr.survival_plot`'s words. Categories left
    without rows are removed, so they take no slot in the legend or the
    palette. Each remaining level is given its colour by name, so it keeps that
    colour in every layer. Codes stored as floats, as a missing value leaves
    them, are labelled as whole numbers, as R labels them.

    Call it after the rows missing a plotted value have been set aside, so that
    the levels are those of the rows drawn.

    Parameters
    ----------
    data : pandas.DataFrame
        The rows to plot, which are left unmodified.
    group : str
        Name of the grouping column.

    Returns
    -------
    tuple
        A new frame of the rows with a group, whose group column is now a
        categorical of the levels in use, followed by those levels in order
        and a dict giving each level its colour from
        :func:`depictr.depictr_palette`.
    """
    if group not in data.columns:
        raise KeyError(f"{group!r} is not a column of `data`.")
    values = data[group]
    if is_numeric_dtype(values) and not is_bool_dtype(values):
        n_distinct = values.dropna().nunique()
        if n_distinct > _MAX_NUMERIC_GROUPS:
            raise ValueError(
                f"`group` ({group}) is numeric with {n_distinct} distinct "
                "values; it looks continuous, not categorical. Bin it, or "
                "convert it to a categorical first."
            )
    missing = values.isna().to_numpy()
    if len(values) and missing.all():
        raise ValueError("`group` is missing for every observation.")
    if missing.any():
        # stacklevel 3 points past the plotting function to the caller's line.
        warnings.warn(
            f"{int(missing.sum())} observation(s) with a missing group were "
            f"dropped.",
            UserWarning,
            stacklevel=3,
        )
        data = data.loc[~missing]
        values = data[group]
    if isinstance(values.dtype, pd.CategoricalDtype):
        codes = values.cat.remove_unused_categories()
    else:
        if is_float_dtype(values) and _whole_numbers(values):
            values = values.astype("int64")
        codes = values.astype("category")
    data = data.copy()
    data[group] = codes
    levels = list(codes.cat.categories)
    # No rows leave no levels, and depictr_palette() refuses n = 0.
    colours = depictr_palette(len(levels)) if levels else []
    palette = dict(zip(levels, colours, strict=True))
    return data, levels, palette


def _group_scales(palette, aesthetics=("color", "fill")):
    """Manual scales that give each level the colour ``palette`` names for it.

    A colour looked up by level does not depend on which levels a layer holds,
    so a level is drawn in the same colour in every layer of the figure.
    """
    make = {"color": scale_color_manual, "fill": scale_fill_manual}
    return [make[a](values=palette, na_value=NA_VALUE) for a in aesthetics]
