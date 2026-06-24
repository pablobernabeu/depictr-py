"""Descriptive summary tables ("Table 1").

The descriptive table that opens many empirical papers: numeric variables as
mean (SD), categorical variables as count (percent), optionally split into one
column per level of a grouping variable. Unlike the rest of depictr this returns
a pandas DataFrame rather than a plot, ready to pass to a formatter such as
``DataFrame.to_markdown`` or ``to_latex``.
"""

from __future__ import annotations

import pandas as pd


def summary_table(data, vars=None, group=None, digits=1, missing=True,
                  max_levels=20):
    """Build a "Table 1" descriptive summary.

    The first row always reports the sample size (``N``) overall and per group.
    Numeric variables are summarised as mean (SD); categorical variables get one
    row per level with count (percent). A ``Missing, n (%)`` row follows any
    variable that has missing values, unless ``missing`` is ``False``.

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
        overall column.
    digits : int
        Decimal places for the numeric summaries.
    missing : bool
        Add a ``Missing, n (%)`` row for variables that contain missing values.
    max_levels : int
        When ``vars`` is ``None``, a non-numeric column whose distinct-value
        count is at least ``max_levels`` and exceeds half the number of rows is
        treated as an identifier and skipped, so it does not explode into one row
        per value.

    Returns
    -------
    pandas.DataFrame
        Columns ``variable``, ``statistic``, ``Overall`` and one column per group
        level. The first row reports ``N``. The variable name is blanked on its
        repeated rows for readability.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    if group is not None and group not in data.columns:
        raise KeyError(f"group column {group!r} not found.")

    n = len(data)
    if vars is None:
        vars = [c for c in data.columns if c != group]
        # Drop identifier-like columns: a non-numeric column with nearly as many
        # distinct values as rows would expand into hundreds of useless rows.
        kept = []
        for v in vars:
            col = data[v]
            if not pd.api.types.is_numeric_dtype(col):
                n_lev = col.dropna().nunique()
                if n_lev >= max_levels and n_lev > n / 2:
                    continue
            kept.append(v)
        vars = kept

    missing_cols = [v for v in vars if v not in data.columns]
    if missing_cols:
        raise KeyError(f"column(s) not found: {missing_cols}")

    # One sub-frame per output column: the whole data plus, if grouping, a split.
    columns = {"Overall": data}
    if group is not None:
        for lvl in pd.unique(data[group].dropna()):
            columns[str(lvl)] = data[data[group] == lvl]
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
        if pd.api.types.is_numeric_dtype(col):
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


def _levels(col):
    """Return a column's category levels in a stable, sensible order.

    Ordered categoricals keep their declared order; everything else is sorted so
    the table is reproducible.
    """
    if isinstance(col.dtype, pd.CategoricalDtype):
        return list(col.cat.categories)
    return sorted(col.dropna().unique(), key=str)
