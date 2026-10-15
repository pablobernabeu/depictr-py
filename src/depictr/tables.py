"""Descriptive summary tables ("Table 1").

The descriptive table that opens many empirical papers: numeric variables as
mean (SD), categorical variables as count (percent), optionally split into one
column per level of a grouping variable. Unlike the rest of depictr this returns
a pandas DataFrame rather than a plot, ready to pass to a formatter such as
``DataFrame.to_markdown`` or ``to_latex``.
"""

from __future__ import annotations

import warnings

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

# The units a duration is reported in, largest first, named as R names the
# units of a difftime so that the twins label a row alike.
_DURATION_UNITS = (("days", 86400.0), ("hours", 3600.0), ("mins", 60.0),
                   ("secs", 1.0))


def summary_table(data, vars=None, group=None, digits=1, missing=True,
                  max_levels=20):
    """Build a "Table 1" descriptive summary.

    The first row always reports the sample size (``N``) overall and per group.
    Numeric variables are summarised as mean (SD); categorical variables get one
    row per level with count (percent). A ``Missing, n (%)`` row follows any
    variable that has missing values, unless ``missing`` is ``False``.

    A datetime column gets a single ``Range`` row, written as ``<earliest> to
    <latest>`` in each column. The cells give the date alone when every time in
    the column falls at midnight, and the time to the minute, or to the second
    when any time has seconds. A timedelta column is summarised as mean (SD) in
    the largest of days, hours, minutes and seconds that its median duration
    reaches, and the statistic names the unit, as in ``Mean (SD), days``. The R
    twin reports a ``difftime`` in the units it carries, which a timedelta
    lacks.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    vars : list of str, optional
        Columns to summarise. If ``None``, every column except ``group`` is
        used, with high-cardinality identifier-like columns skipped (see
        ``max_levels``).
    group : str, optional
        A grouping column; one summary column is produced per level, alongside an
        overall column. Categorical levels keep their declared category order and
        everything else is sorted, so the table does not depend on the row order.
        Records whose group value is missing get a ``Missing`` column, so the
        group sizes always sum to the overall ``N``.
    digits : int
        Decimal places for the numeric summaries.
    missing : bool
        Add a ``Missing, n (%)`` row for variables that contain missing values.
    max_levels : int
        When ``vars`` is ``None``, a categorical or text column whose
        distinct-value count is at least ``max_levels`` and exceeds half the
        number of rows is treated as an identifier and skipped with a
        ``UserWarning`` that names it, so it does not explode into one row per
        value. Pass such a column in ``vars`` to include it.

    Returns
    -------
    pandas.DataFrame
        Columns ``variable``, ``statistic``, ``Overall``, one column per group
        level, and a trailing ``Missing`` column when the group column has any
        missing values. The first row reports ``N``. The variable name is blanked
        on its repeated rows for readability.

    Examples
    --------
    >>> import depictr as dp
    >>> wb = dp.wellbeing_survey()
    >>> tab = dp.summary_table(wb, vars=["life_satisfaction", "stress"], group="region")
    >>> tab.columns.tolist()[0]
    'variable'
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    if group is not None and group not in data.columns:
        raise KeyError(f"group column {group!r} not found.")

    n = len(data)
    if vars is None:
        vars = [c for c in data.columns if c != group]
        # Drop identifier-like columns: a column summarised level by level with
        # nearly as many distinct values as rows would expand into hundreds of
        # useless rows. Dates and durations take one row, so they stay.
        kept, skipped = [], []
        for v in vars:
            col = data[v]
            if not _is_continuous(col):
                n_lev = col.dropna().nunique()
                if n_lev >= max_levels and n_lev > n / 2:
                    skipped.append(v)
                    continue
            kept.append(v)
        vars = kept
        if skipped:
            # The R twin's message, in a warning. The columns used to vanish
            # without a word.
            warnings.warn(
                "summary_table(): skipping high-cardinality column(s): "
                f"{', '.join(str(v) for v in skipped)}. "
                "Pass them via `vars` to include them.",
                UserWarning,
                stacklevel=2,
            )

    missing_cols = [v for v in vars if v not in data.columns]
    if missing_cols:
        raise KeyError(f"column(s) not found: {missing_cols}")

    # One sub-frame per output column: the whole data plus, if grouping, a split.
    columns = {"Overall": data}
    if group is not None:
        # _levels rather than first-appearance order, the same helper the level
        # rows use: a categorical keeps its declared category order and anything
        # else is sorted, so shuffling the rows cannot reorder the table.
        for lvl in _levels(data[group]):
            columns[str(lvl)] = data[data[group] == lvl]
        # Records with no group value need a column of their own. Without one
        # they fall out of every group while Overall still counts them, so the
        # group sizes stop summing to N with nothing to say why.
        if data[group].isna().any():
            na_name = "Missing"
            while na_name in columns:  # a level may itself be called "Missing"
                na_name += "_"
            columns[na_name] = data[data[group].isna()]
    col_names = list(columns.keys())

    def num_cell(v):
        v = pd.Series(v).dropna()
        if len(v) == 0:
            return "--"
        if len(v) == 1:
            # SD is undefined for a single observation; report the mean alone.
            return f"{v.mean():.{digits}f} (n=1)"
        return f"{v.mean():.{digits}f} ({v.std(ddof=1):.{digits}f})"

    def cat_cell(col, level):
        x = col.dropna()
        tot = len(x)
        if tot == 0:
            return "--"
        k = int((x == level).sum())
        return f"{k} ({100 * k / tot:.0f}%)"

    def miss_cell(col):
        tot = len(col)
        if tot == 0:
            return "--"
        k = int(col.isna().sum())
        return f"{k} ({100 * k / tot:.0f}%)"

    rows = []

    # Sample size first.
    row = {"variable": "N", "statistic": ""}
    for name in col_names:
        row[name] = str(len(columns[name]))
    rows.append(row)

    for v in vars:
        col = data[v]
        if is_datetime64_any_dtype(col):
            # A date is not a level: expanded as one, it gave a row per day.
            fmt = _date_cell_format(col)
            row = {"variable": v, "statistic": "Range"}
            for name in col_names:
                row[name] = _date_range_cell(columns[name][v], fmt)
            rows.append(row)
        elif is_timedelta64_dtype(col):
            unit, size = _duration_unit(col)
            row = {"variable": v, "statistic": f"Mean (SD), {unit}"}
            for name in col_names:
                row[name] = num_cell(columns[name][v].dt.total_seconds() / size)
            rows.append(row)
        elif is_numeric_dtype(col):
            row = {"variable": v, "statistic": "Mean (SD)"}
            for name in col_names:
                row[name] = num_cell(columns[name][v])
            rows.append(row)
        else:
            for level in _levels(col):
                row = {"variable": v, "statistic": str(level)}
                for name in col_names:
                    row[name] = cat_cell(columns[name][v], level)
                rows.append(row)
        if missing and col.isna().any():
            row = {"variable": v, "statistic": "Missing, n (%)"}
            for name in col_names:
                row[name] = miss_cell(columns[name][v])
            rows.append(row)

    out = pd.DataFrame(rows, columns=["variable", "statistic"] + col_names)
    # Blank the repeated variable name so each variable reads as one block.
    out.loc[out["variable"].duplicated(), "variable"] = ""
    return out


def _is_continuous(col):
    """Whether a column holds numbers, dates or durations.

    Such a column takes a single row of the summary table, where a category
    takes one per level, and ``explore_bivariate()`` places it on a continuous
    axis.
    """
    return (is_numeric_dtype(col) or is_datetime64_any_dtype(col)
            or is_timedelta64_dtype(col))


def _date_cell_format(col):
    """Return the format of the cells of a datetime column.

    It is chosen once for the whole column, so that every cell of the row has
    the same precision: the date alone when every time falls at midnight,
    minutes when no time has seconds and seconds otherwise. The R twin applies
    the same rule. The clock fields are read in the column's own time zone, the
    one the cells are printed in.
    """
    x = col.dropna()
    no_seconds = ((x.dt.second == 0) & (x.dt.microsecond == 0)
                  & (x.dt.nanosecond == 0))
    if (no_seconds & (x.dt.hour == 0) & (x.dt.minute == 0)).all():
        return "%Y-%m-%d"
    if no_seconds.all():
        return "%Y-%m-%d %H:%M"
    return "%Y-%m-%d %H:%M:%S"


def _date_range_cell(col, fmt):
    """One cell of a date row: the earliest and latest values present, or
    ``--`` when there are none."""
    x = col.dropna()
    if len(x) == 0:
        return "--"
    return f"{x.min().strftime(fmt)} to {x.max().strftime(fmt)}"


def _duration_unit(col):
    """Return the name and length in seconds of the unit a duration is reported
    in: the largest unit that the median absolute duration reaches."""
    seconds = col.dropna().dt.total_seconds().abs()
    median = seconds.median() if len(seconds) else 0.0
    for name, size in _DURATION_UNITS:
        if median >= size:
            return name, size
    return _DURATION_UNITS[-1]


def _levels(col):
    """Return a column's category levels in a stable, sensible order.

    Categoricals keep their declared category order; everything else is sorted
    so the table is reproducible.
    """
    if isinstance(col.dtype, pd.CategoricalDtype):
        return list(col.cat.categories)
    return sorted(col.dropna().unique(), key=str)
