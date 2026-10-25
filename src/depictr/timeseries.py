"""Time-series plots, themed consistently.

The correlation structure and the seasonal decomposition come from
``statsmodels`` (the de-facto standard for these); the figures are re-drawn under
the depictr theme so an ACF stem plot or a decomposition sits in the same visual
language as the rest of a report. The plain series and the seasonal plot (one
line per cycle) need only the depictr stack.

``statsmodels`` arrives with the core install (plotnine itself requires it);
the ``depictr[models]`` extra remains to pin the tested floor.
"""

from __future__ import annotations

import math
import warnings

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
    guide_legend,
    labs,
    scale_color_manual,
    scale_x_continuous,
)

from .palette import ACCENT, BRAND, _cycle_palette
from .theme import theme_depictr


def _require_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The time-series plots need statsmodels. Install it with "
            '`pip install "depictr[models]"`.'
        ) from exc


_SEVERAL_COLUMNS = "`x` has {} columns; pass one series at a time."


def _as_series(x):
    """Coerce the input to a Series of floats, keeping its index and its gaps.

    A Series keeps its index, and a 1-D array becomes a Series over a plain
    integer index. The index is used only as the x-axis of the series plot, so a
    datetime, period or integer index all work. A frame or 2-D array with one
    column is read as that column, so a frame keeps its index. Several columns
    are refused: read row by row, as a flattened array is, they would be
    interleaved into one sequence.

    The values are read as floats, with NaN for every missing value. plotnine
    reads the values of an object column, such as Python floats mixed with
    None, as categories, one group each. Missing values are kept in place.
    Dropping one would close up the gap and move every later observation one
    step along.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(_SEVERAL_COLUMNS.format(x.shape[1]))
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        return pd.Series(x.to_numpy(dtype=float, na_value=np.nan), index=x.index)
    arr = np.asarray(x, dtype=float)
    n_cols = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
    if n_cols != 1:
        raise ValueError(_SEVERAL_COLUMNS.format(n_cols))
    return pd.Series(arr.reshape(-1))


def _as_complete_series(x):
    """Coerce to a Series, trim missing ends, and refuse internal missing values.

    The ACF and the decomposition read position in the sequence as time, and so
    does the seasonal plot of a series without a calendar index. Dropping an
    internal missing value closes up the gap: the ACF correlates values across
    it, the decomposition misaligns, and every later observation lands on the
    wrong within-period position. Whether to drop or impute is the analyst's
    call, not something to make silently on their behalf. Missing values at
    either end only shorten the series, so they are trimmed.
    """
    series = _as_series(x)
    present = series.notna().to_numpy()
    if not present.any():
        return series.dropna()  # empty; each caller reports this case itself
    first, last = np.flatnonzero(present)[[0, -1]]
    series = series.iloc[first:last + 1]
    if series.isna().any():
        raise ValueError(
            "`x` must not contain internal missing values; drop or impute "
            "them before plotting."
        )
    return series


# The index frequencies that imply a seasonal period, matched whole and with
# their case. pandas names minutes "min" and milliseconds "ms", so a match on
# the first letter would read both as monthly. Months and quarters are named by
# their start or end ("MS", "ME", "QS-OCT"), and a PeriodIndex names them "M"
# and "Q-DEC".
_SEASONAL_PERIODS = {
    "M": 12, "ME": 12, "MS": 12,
    "Q": 4, "QE": 4, "QS": 4,
    "D": 7,
}


def _index_period(x):
    """The seasonal period that the frequency of ``x``'s index implies, or None."""
    idx = getattr(x, "index", None)
    freq = getattr(idx, "freqstr", None) or getattr(idx, "inferred_freq", None)
    if not freq:
        return None
    # The anchor after the hyphen, as in "Q-DEC", names the month that ends or
    # starts the year and has no bearing on the length of the cycle.
    return _SEASONAL_PERIODS.get(freq.split("-")[0])


def _infer_period(x, period):
    """Return the seasonal period, inferred from a datetime/period index if absent.

    A monthly index gives 12, a quarterly index 4 and a daily index 7 (the
    weekly cycle, the only seasonality a bare daily index implies). Anything
    else needs an explicit ``period``. Annual data is deliberately absent: it
    has no within-year position to plot against, and a period of 1 is not a
    seasonal period, so it falls through to the error below rather than
    producing a figure with one observation per cycle.
    """
    if period is not None:
        return int(period)
    inferred = _index_period(x)
    if inferred is not None:
        return inferred
    raise ValueError(
        "Could not infer the seasonal period from the index; pass `period` "
        "(for example period=12 for monthly data)."
    )


# R's ts() takes a frequency within getOption("ts.eps") of a whole number for
# that number. A period computed in floating point, such as 0.1 * 3 * 40, is
# therefore read as the whole number it was meant to be.
_TS_EPS = 1e-5


def _whole_period(period):
    """A seasonal period as whole seasons, rounded down with a warning if need be.

    A period such as 365.25 / 7, the weeks in a year, has no whole number of
    seasons, so it is rounded down and a ``UserWarning`` says so. The R twin
    rounds such a frequency down with a message, as R's ``stl()`` does. A
    period below 2 leaves no seasons to compare and is refused, as the R twin
    refuses such a frequency.
    """
    if period is None:
        return None
    value = float(period)
    nearest = round(value)
    if abs(value - nearest) < _TS_EPS:
        value = float(nearest)
    if value < 2:
        raise ValueError("A seasonal plot needs a seasonal series (period >= 2).")
    whole = math.floor(value)
    if whole != value:
        warnings.warn(
            f"Period {value:.7g} is not a whole number; using {whole} seasons "
            f"per cycle.",
            UserWarning,
            stacklevel=3,
        )
    return whole


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
           "Nov", "Dec"]
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _seasons(series, period):
    """Each observation's season and cycle, and the names of the seasons.

    Seasons run from 1 to ``period`` and cycles from 1. A calendar index sets
    the season when its frequency implies ``period``, as the R twin's
    ``stats::cycle()`` reads it from a ``ts``. The season is the month for 12,
    the quarter for 4 and the weekday for 7, Monday first. A series starting in
    April then starts at season 4, and its Aprils line up with every other
    April. The quarter is the one pandas reports, so a ``PeriodIndex`` with a
    fiscal year numbers its quarters from that year's start. The seasons are
    then named by month, quarter or weekday, in English as R's ``month.abb``
    names them.

    Otherwise, as for a bare array, the seasons are counted from the first
    observation and left unnamed (``None``).

    A cycle starts wherever the season fails to advance, as in the R twin, so
    the cycles of a calendar break at the turn of the year (or of the week)
    and the first and last of them can be short.
    """
    idx = series.index
    calendar = isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex))
    if calendar and _index_period(series) == period:
        if period == 12:
            season, names = idx.month, _MONTHS
        elif period == 4:
            season, names = idx.quarter, [f"Q{q}" for q in range(1, 5)]
        else:
            season, names = idx.dayofweek + 1, _WEEKDAYS
        season = np.asarray(season, dtype=int)
        cycle = np.cumsum(np.r_[True, np.diff(season) <= 0])
        return season, cycle, names
    n = len(series)
    return np.arange(n) % period + 1, np.arange(n) // period + 1, None


def acf_plot(x, kind="acf", lags=None, title=None):
    """Autocorrelation or partial autocorrelation as a stem plot.

    The correlations come from statsmodels (``acf``/``pacf``). The approximate
    95% significance band, plus or minus 1.96 / sqrt(n), is drawn as a shaded
    ribbon: stems reaching past it are the candidate lags.

    The partial autocorrelations are the Yule-Walker estimates from the sample
    autocorrelations, without the small-sample adjustment
    (``pacf(method="ywm")``). They are the values R's ``pacf()`` gives, so the
    twins draw the same stems. statsmodels' default, the adjusted estimator,
    differs from R by up to 0.18 on R's ``AirPassengers``, and statsmodels'
    documentation reports that it performed consistently worse than the other
    estimators in simulations.

    Parameters
    ----------
    x : pandas.Series or array-like
        The series. Any index is ignored here; only the values are used.
        Missing values at either end are trimmed; an internal missing value is
        an error, since dropping it would correlate values across the
        closed-up gap. The R twin keeps such a value in place and computes the
        correlations from the pairs of observations both present, with a
        warning. A frame or 2-D array with one column is read as that column,
        and one with several columns is refused.
    kind : {"acf", "pacf"}
        Autocorrelation or partial autocorrelation.
    lags : int, optional
        Number of lags to show. Defaults to ``min(10 * log10(n), n - 1)`` for the
        ACF and a smaller cap for the PACF, which is undefined beyond ``n // 2``.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(120)
    >>> series = pd.Series(50 + 0.3 * t + rng.normal(0, 3, 120))
    >>> p = dp.acf_plot(series)
    """
    _require_statsmodels()
    from statsmodels.tsa.stattools import acf, pacf

    if kind not in {"acf", "pacf"}:
        raise ValueError("`kind` must be 'acf' or 'pacf'.")
    series = _as_complete_series(x)
    n = len(series)
    if n == 0:
        # The default lag count below takes log10(n), which is -inf at zero and
        # then an OverflowError from int(), naming neither the argument nor the
        # problem. The sibling plots turn an all-missing series away by name, so
        # this one does too.
        raise ValueError("`x` has no non-missing values to plot.")
    if lags is None:
        lags = int(min(10 * np.log10(n), n - 1))
        if kind == "pacf":
            lags = int(min(lags, n // 2 - 1))
    lags = max(int(lags), 1)

    if kind == "acf":
        values = acf(series.to_numpy(), nlags=lags, fft=True)
        y_lab = "Autocorrelation"
    else:
        values = pacf(series.to_numpy(), nlags=lags, method="ywm")
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
        observation number is used. Missing values at either end are trimmed;
        an internal missing value is an error, since dropping it would shift
        every later observation and misalign the components. A frame or 2-D
        array with one column is read as that column, and one with several
        columns is refused.
    period : int, optional
        Seasonal period. Inferred from a monthly, quarterly or daily index when
        omitted (12, 4 and 7 respectively; a daily index is assumed weekly).
    model : {"additive", "multiplicative"}
        The decomposition model.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(120)
    >>> series = pd.Series(
    ...     50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 120),
    ...     index=pd.period_range("2016-01", periods=120, freq="M"),
    ... )
    >>> p = dp.decompose_plot(series, period=12)
    """
    _require_statsmodels()
    from statsmodels.tsa.seasonal import seasonal_decompose

    if model not in {"additive", "multiplicative"}:
        raise ValueError("`model` must be 'additive' or 'multiplicative'.")
    series = _as_complete_series(x)
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
        + labs(x="", y="", title=title)
        + theme_depictr()
    )


def seasonal_plot(x, period=None, title=None):
    """Seasonal plot (one line per cycle): value by position in the period.

    Each cycle (for example each year of monthly data) is one line across the
    positions in the period. A cycle that the series starts or ends partway
    through is drawn as the short line it is. Overlaying the cycles makes the
    repeating shape and any drift between cycles easy to read.

    A monthly, quarterly or daily index carries the calendar. When its
    frequency implies the period, the position is read from it: the month for
    12, labelled Jan to Dec, the quarter for 4, labelled Q1 to Q4, and the
    weekday for 7, labelled Mon to Sun. A series that starts in April then
    draws April at position 4, and each cycle runs from January to December,
    as in the R twin. A quarterly ``PeriodIndex`` with a fiscal year numbers
    its quarters from that year's start, as pandas does. Without such an
    index, as for a bare array, the positions are counted from the first
    observation and labelled by number.

    The lines run from a mid blue for the first cycle to a dark blue for the
    last, so they read in time order. Even the palest has a contrast of 3.51:1
    against white. Past about six cycles, neighbouring cycles take colours too
    close to tell apart, and the colours then show the order of the cycles more
    than which is which. When individual cycles must be told apart, plot fewer
    cycles, or use the R twin's ``style = "subseries"``. It gives each season a
    panel, in which every cycle keeps its own place on the axis.

    Parameters
    ----------
    x : pandas.Series or array-like
        The series. Missing values at either end are trimmed; an internal
        missing value is an error, since dropping it would assign every later
        observation to the wrong position in the period. A frame or 2-D array
        with one column is read as that column, and one with several columns is
        refused.
    period : int, optional
        Seasonal period. Inferred from a monthly, quarterly or daily index when
        omitted (12, 4 and 7 respectively; a daily index is assumed weekly). A
        period that is not a whole number, such as 365.25 / 7 for weekly data,
        is rounded down to whole seasons with a ``UserWarning``, as the R twin
        rounds a frequency down. A period below 2 is refused, as in the R twin.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(120)
    >>> series = pd.Series(
    ...     50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 120),
    ...     index=pd.period_range("2016-01", periods=120, freq="M"),
    ... )
    >>> p = dp.seasonal_plot(series, period=12)
    """
    series = _as_complete_series(x)
    period = _infer_period(series, _whole_period(period))

    values = series.to_numpy()
    if len(values) == 0:
        raise ValueError("`x` has no non-missing values to plot.")
    position, cycle, names = _seasons(series, period)
    n_cycles = int(cycle.max())
    df = pd.DataFrame({
        "position": position,
        "value": values,
        # Ordered on the cycle number, not on its string form: a plain
        # categorical over strings sorts lexicographically, which puts cycle 10
        # between 1 and 2 and hands it the third colour of a ramp meant to run
        # in time order.
        "cycle": pd.Categorical(
            cycle.astype(str),
            categories=[str(c) for c in range(1, n_cycles + 1)],
            ordered=True,
        ),
    })
    breaks = list(range(1, period + 1))
    x_scale = (scale_x_continuous(breaks=breaks) if names is None
               else scale_x_continuous(breaks=breaks, labels=names))

    return (
        ggplot(df, aes("position", "value", color="cycle", group="cycle"))
        + geom_line(size=0.7)
        + geom_point(size=1.5)
        + x_scale
        # The cycles are a sequence, not unrelated categories, so they take a
        # single-hue ramp. It reads in time order and, unlike the eight colours
        # of the qualitative palette, extends to any number of cycles, although
        # past about six its neighbouring colours blur. It leaves out the
        # palest colour of the sequential palette, which is close to invisible
        # as a line. The legend is reversed so the darkest, most recent cycle
        # sits at the top of the key, matching the R twin.
        + scale_color_manual(values=_cycle_palette(n_cycles),
                             name="Cycle", guide=guide_legend(reverse=True))
        + labs(x="Position in period", y="Value", title=title)
        + theme_depictr()
    )


def timeseries_plot(x, rolling=None, title=None):
    """The series as a line, optionally with a rolling-mean overlay.

    Parameters
    ----------
    x : pandas.Series or array-like
        The series. A datetime or period index sets the x-axis; otherwise the
        observation number is used. A missing value keeps its place on the axis
        and leaves a break in the line. A frame or 2-D array with one column is
        read as that column, and one with several columns is refused. The R
        twin draws one line per column of a multivariate ``ts``.
    rolling : int, optional
        Window length for a centred moving average, drawn over the series in
        the accent colour. An odd window averages each observation with the
        same number of neighbours on either side. An even window, such as 12
        for monthly data, gives the 2×12 moving average, a 12-point average
        followed by a 2-point one (Hyndman & Athanasopoulos, 2021, Section
        3.3), as in the R twin. The two observations at the ends of the 13 it
        spans count half, which centres the average on its observation, as
        equal weights over an even window cannot. A window
        that runs off either end of the series, or reaches a missing value, is
        missing too, so the overlay breaks where the line does. Omit for the
        raw line only.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    References
    ----------
    Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and
    practice (3rd ed.). OTexts. https://otexts.com/fpp3/

    Examples
    --------
    >>> import depictr as dp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(120)
    >>> series = pd.Series(50 + 0.3 * t + rng.normal(0, 3, 120))
    >>> p = dp.timeseries_plot(series, rolling=12)
    """
    series = _as_series(x)
    idx = series.index
    x_vals = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else idx
    if not np.issubdtype(np.asarray(x_vals).dtype, np.datetime64):
        x_vals = np.arange(len(series))
    df = pd.DataFrame({"x": x_vals, "value": series.to_numpy()})

    # plotnine keeps a missing value inside a line as a break in it, which is
    # how a gap should look. na_rm silences the warning about the missing values
    # it drops at either end, which every rolling overlay has.
    p = ggplot(df, aes("x", "value")) + geom_line(color=BRAND, size=0.7, na_rm=True)
    if rolling:
        # Taken over the series with its gaps, so a window that reaches one is
        # missing as well.
        df = df.assign(roll=_moving_average(df["value"], int(rolling)))
        p = p + geom_line(aes(y="roll"), data=df, color=ACCENT, size=1.0, na_rm=True)
    return p + labs(x="", y="Value", title=title) + theme_depictr()


def _moving_average(values, window):
    """Centred moving average over ``window`` observations, as the R twin's.

    Equal weights over an odd window centre the average on an observation.
    Over an even window they centre it between two observations: pandas'
    ``rolling(4, center=True)`` puts the mean of observations 1 to 4 at
    observation 3. An even window m therefore gives the 2 x m average, an
    m-point average followed by a 2-point one (Hyndman & Athanasopoulos, 2021,
    Section 3.3). Its weights, ``[0.5, 1, ..., 1, 0.5] / m``, span m + 1
    observations. It is the trend that ``seasonal_decompose()`` estimates for
    a period of m. A window that runs off either end of the series, or reaches
    a missing value, gives NaN, as R's ``filter()`` gives NA.
    """
    values = np.asarray(values, dtype=float)
    if window < 2 or window % 2:
        # An odd window is centred already. A negative one takes the same path,
        # so pandas refuses it with its own error.
        return pd.Series(values).rolling(window, center=True).mean().to_numpy()
    out = np.full(values.shape, np.nan)
    # np.convolve() swaps its arguments when the weights are the longer of the
    # two, so a series too short for one full window keeps every value missing.
    if len(values) > window:
        weights = np.r_[0.5, np.ones(window - 1), 0.5] / window
        half = window // 2
        # Each full window holds one value, centred on the observation half a
        # window in from its start. np.convolve() works window by window, so a
        # missing value spoils only the windows that reach it.
        out[half:-half] = np.convolve(values, weights, mode="valid")
    return out
