"""Time-series plots, themed consistently.

The correlation structure and the seasonal decomposition come from
``statsmodels`` (the de-facto standard for these); the figures are re-drawn under
the depictr theme so an ACF stem plot or a decomposition sits in the same visual
language as the rest of a report. The plain series and seasonal subseries plots
need only the depictr stack.

Install the optional dependency with ``pip install depictr[models]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    facet_wrap,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_segment,
    ggplot,
    labs,
    scale_x_continuous,
)

from .palette import ACCENT, BRAND
from .theme import scale_colour_depictr, theme_depictr


def _require_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The time-series plots need statsmodels. Install it with "
            "`pip install depictr[models]`."
        ) from exc
    import statsmodels  # noqa: F401

    return statsmodels


def _as_series(x):
    """Coerce the input to a pandas Series, keeping any index it arrives with.

    A 1-D array becomes a Series over a plain integer index; a Series is passed
    through. The index is used only as the x-axis for the series and seasonal
    plots, so a datetime, period or integer index all work.
    """
    if isinstance(x, pd.Series):
        return x.dropna()
    arr = np.asarray(x, dtype=float).ravel()
    return pd.Series(arr).dropna()


def _infer_period(x, period):
    """Return the seasonal period, inferred from a datetime/period index if absent.

    A monthly index gives 12, a quarterly index 4. Anything else needs an
    explicit ``period``.
    """
    if period is not None:
        return int(period)
    idx = getattr(x, "index", None)
    freq = getattr(idx, "freqstr", None) or getattr(idx, "inferred_freq", None)
    if freq:
        head = freq.upper()[0]
        mapping = {"M": 12, "Q": 4, "A": 1, "Y": 1, "D": 7}
        if head in mapping:
            return mapping[head]
    raise ValueError(
        "Could not infer the seasonal period from the index; pass `period` "
        "(for example period=12 for monthly data)."
    )


def acf_plot(x, kind="acf", lags=None, title=None):
    """Autocorrelation or partial autocorrelation as a stem plot.

    The correlations come from statsmodels (``acf``/``pacf``). The approximate
    95% significance band, plus or minus 1.96 / sqrt(n), is drawn as a shaded
    ribbon: stems reaching past it are the candidate lags.

    Parameters
    ----------
    x : pandas.Series or array-like
        The series. Any index is ignored here; only the values are used.
    kind : {"acf", "pacf"}
        Autocorrelation or partial autocorrelation.
    lags : int, optional
        Number of lags to show. Defaults to ``min(10 * log10(n), n - 1)`` for the
        ACF and a smaller cap for the PACF, which is undefined beyond ``n // 2``.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
    """
    _require_statsmodels()
    from statsmodels.tsa.stattools import acf, pacf

    if kind not in {"acf", "pacf"}:
        raise ValueError("`kind` must be 'acf' or 'pacf'.")
    series = _as_series(x)
    n = len(series)
    if lags is None:
        lags = int(min(10 * np.log10(n), n - 1))
        if kind == "pacf":
            lags = int(min(lags, n // 2 - 1))
    lags = max(int(lags), 1)

    if kind == "acf":
        values = acf(series.to_numpy(), nlags=lags, fft=True)
        y_lab = "Autocorrelation"
    else:
        values = pacf(series.to_numpy(), nlags=lags)
        y_lab = "Partial autocorrelation"

    # Drop lag 0 (always 1) so the band and the scale are not dominated by it.
    df = pd.DataFrame({"lag": np.arange(len(values)), "value": values})
    df = df[df["lag"] > 0]
    band = 1.96 / np.sqrt(n)

    return (
        ggplot(df, aes("lag", "value"))
        + geom_ribbon(aes(ymin=-band, ymax=band), fill=BRAND, alpha=0.12)
        + geom_hline(yintercept=0, color="#9e9e9e", size=0.4)
        + geom_segment(aes(xend="lag", yend=0), color=BRAND, size=0.7)
        + geom_point(color=BRAND, size=2)
        + labs(x="Lag", y=y_lab, title=title)
        + theme_depictr()
    )


def decompose_plot(x, period=None, model="additive", title=None):
    """Seasonal decomposition as a stacked, facetted figure.

    statsmodels' ``seasonal_decompose`` splits the series into observed, trend,
    seasonal and residual components. The four are drawn in one figure, stacked
    in a single column with a free y-scale per component (``facet_wrap`` on the
    component), so each is legible at its own magnitude.

    Parameters
    ----------
    x : pandas.Series or array-like
        The series. A datetime or period index sets the x-axis; otherwise the
        observation number is used.
    period : int, optional
        Seasonal period. Inferred from a monthly/quarterly index when omitted.
    model : {"additive", "multiplicative"}
        The decomposition model.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
    """
    _require_statsmodels()
    from statsmodels.tsa.seasonal import seasonal_decompose

    if model not in {"additive", "multiplicative"}:
        raise ValueError("`model` must be 'additive' or 'multiplicative'.")
    series = _as_series(x)
    period = _infer_period(series, period)

    result = seasonal_decompose(series.to_numpy(), model=model, period=period)
    pieces = {
        "Observed": result.observed,
        "Trend": result.trend,
        "Seasonal": result.seasonal,
        "Residual": result.resid,
    }
    # A shared x-axis: the original index if it plots, else the position.
    idx = series.index
    x_vals = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else idx
    if not np.issubdtype(np.asarray(x_vals).dtype, np.datetime64):
        x_vals = np.arange(len(series))

    order = list(pieces)
    long = pd.concat(
        [pd.DataFrame({"x": x_vals, "value": comp, "component": name})
         for name, comp in pieces.items()],
        ignore_index=True,
    )
    long["component"] = pd.Categorical(long["component"], categories=order,
                                       ordered=True)

    return (
        ggplot(long, aes("x", "value"))
        + geom_line(color=BRAND, size=0.7)
        + facet_wrap("component", ncol=1, scales="free_y")
        + labs(x=None, y=None, title=title)
        + theme_depictr()
    )


def seasonal_plot(x, period=None, title=None):
    """Seasonal subseries plot: value by position in the period, one line per cycle.

    Each completed cycle (for example each year of monthly data) is one line,
    drawn across the within-period position (month 1 to 12). Overlaying the cycles
    makes the repeating shape and any drift between cycles easy to read.

    Parameters
    ----------
    x : pandas.Series or array-like
        The series.
    period : int, optional
        Seasonal period. Inferred from a monthly/quarterly index when omitted.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
    """
    series = _as_series(x)
    period = _infer_period(series, period)

    values = series.to_numpy()
    n = len(values)
    position = (np.arange(n) % period) + 1
    cycle = np.arange(n) // period
    df = pd.DataFrame({
        "position": position,
        "value": values,
        "cycle": pd.Categorical(cycle.astype(str)),
    })

    n_cycles = df["cycle"].nunique()
    return (
        ggplot(df, aes("position", "value", color="cycle", group="cycle"))
        + geom_line(size=0.7)
        + geom_point(size=1.5)
        + scale_x_continuous(breaks=list(range(1, period + 1)))
        + scale_colour_depictr(n=n_cycles, name="Cycle")
        + labs(x="Position in period", y="Value", title=title)
        + theme_depictr()
    )


def timeseries_plot(x, rolling=None, title=None):
    """The series as a line, optionally with a rolling-mean overlay.

    Parameters
    ----------
    x : pandas.Series or array-like
        The series. A datetime or period index sets the x-axis; otherwise the
        observation number is used.
    rolling : int, optional
        Window length for a centred rolling mean, drawn over the series in the
        accent colour. Omit for the raw line only.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
    """
    series = _as_series(x)
    idx = series.index
    x_vals = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else idx
    if not np.issubdtype(np.asarray(x_vals).dtype, np.datetime64):
        x_vals = np.arange(len(series))
    df = pd.DataFrame({"x": x_vals, "value": series.to_numpy()})

    p = ggplot(df, aes("x", "value")) + geom_line(color=BRAND, size=0.7)
    if rolling:
        df = df.assign(roll=df["value"].rolling(int(rolling), center=True).mean())
        p = p + geom_line(aes(y="roll"), data=df, color=ACCENT, size=1.0)
    return p + labs(x=None, y="Value", title=title) + theme_depictr()
