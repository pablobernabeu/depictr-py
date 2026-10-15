"""Smoke tests for the time-series plots: each builds a figure without error."""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.lines import Line2D
from plotnine import ggplot
from plotnine.exceptions import PlotnineWarning

import depictr.timeseries as ts


def _monthly_series(n=120, seed=0):
    """A monthly series with a trend, a 12-period seasonal term and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)
    idx = pd.period_range("2010-01", periods=n, freq="M")
    return pd.Series(values, index=idx)


S = _monthly_series()
V = S.to_numpy()


def _builds(p):
    assert isinstance(p, ggplot)
    p.draw(show=False)
    return True


def test_the_gallery_describes_the_autocorrelations_it_draws():
    # docs/gallery/multivariate-and-time-series.md described spikes at the
    # multiples of twelve in both functions. The default lags reach one
    # multiple, where the autocorrelation peaks on a slow decay, and the
    # partial autocorrelation there lies inside the band. The page now says
    # so, and this holds it to the series the page draws.
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(0)
    t = np.arange(120)
    series = pd.Series(
        50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 120),
        index=pd.period_range("2016-01", periods=120, freq="M"),
    )
    acf = ts.acf_plot(series).data.set_index("lag")["value"]
    assert [lag for lag in acf.index if lag % 12 == 0] == [12]
    assert acf[6] < min(acf[5], acf[7])
    assert acf[12] > max(acf[11], acf[13])
    pacf = ts.acf_plot(series, kind="pacf").data.set_index("lag")["value"]
    assert pacf[1] == pacf.abs().max()
    assert abs(pacf[12]) < 1.96 / np.sqrt(len(series)) / 2


def test_acf_plot_kinds():
    pytest.importorskip("statsmodels")
    assert _builds(ts.acf_plot(S, kind="acf"))
    assert _builds(ts.acf_plot(S, kind="pacf"))
    assert _builds(ts.acf_plot(V, kind="acf", lags=20))


def test_acf_plot_bad_kind():
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError):
        ts.acf_plot(S, kind="nonsense")


# ---- the partial autocorrelation: parity with the R twin -------------------
# AirPassengers from R's datasets package: the monthly totals of international
# airline passengers, in thousands, from January 1949 to December 1960, one
# year per row. R_PACF_AIR holds the partial autocorrelations at lags 1 to 21
# that the R twin's acf_plot(type = "partial") draws for it, from R's pacf().
# Both are printed by dev/goldens/timeseries.R from the R twin at the commit of
# change set CS40, whose tests pin the same values.
AIR_PASSENGERS = np.array([
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
], dtype=float)
R_PACF_AIR = [
    0.948047340752492, -0.229421874117177, 0.038147780503991,
    0.093785438151741, 0.073606697894160, 0.007727602590501,
    0.125597130176117, 0.089951343164267, 0.232488542208562,
    0.166051259804356, 0.171274419743056, -0.135431102278015,
    -0.539691241907120, -0.026610140069523, 0.090764981938823,
    0.024955559165498, 0.032515570006784, 0.073432640501125,
    0.048442114810552, -0.045541680123359, 0.045752935204032,
]


def test_air_passengers_literal_is_rs_series():
    # A typing slip in the literal would move every value pinned against it.
    assert AIR_PASSENGERS.size == 144
    assert AIR_PASSENGERS.sum() == 40363


def test_pacf_gives_the_r_twins_values():
    # statsmodels' default estimator, the adjusted Yule-Walker, differed from
    # R's pacf() by up to 0.18 on this series (-0.718 against -0.540 at lag
    # 13) and put lags 15, 18 and 20 outside the band where R does not.
    pytest.importorskip("statsmodels")
    p = ts.acf_plot(pd.Series(AIR_PASSENGERS), kind="pacf")
    assert p.data["lag"].tolist() == list(range(1, 22))
    np.testing.assert_allclose(p.data["value"], R_PACF_AIR, rtol=0, atol=1e-10)


def test_pacf_at_lag_one_is_the_autocorrelation():
    # The partial autocorrelation at lag 1 is the autocorrelation at lag 1. The
    # adjusted estimator multiplied it by n / (n - 1), 120 / 119 for this
    # series.
    pytest.importorskip("statsmodels")
    pacf_1 = ts.acf_plot(S, kind="pacf").data["value"].iloc[0]
    acf_1 = ts.acf_plot(S, kind="acf").data["value"].iloc[0]
    assert pacf_1 == pytest.approx(acf_1, rel=1e-12)


def test_decompose_plot():
    pytest.importorskip("statsmodels")
    assert _builds(ts.decompose_plot(S, model="additive"))
    assert _builds(ts.decompose_plot(V, period=12, model="additive"))


def test_decompose_plot_needs_period_for_bare_array():
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError):
        ts.decompose_plot(V)  # no index, no period to infer from


def test_seasonal_plot():
    assert _builds(ts.seasonal_plot(S))
    assert _builds(ts.seasonal_plot(V, period=12))


def test_seasonal_cycles_are_ordered_numerically():
    # A plain categorical over strings sorts "10" between "1" and "2", which
    # both scrambles the legend and misassigns the colour ramp. The cycles are
    # numbered from 1, as in the R twin, where they were numbered from 0.
    p = ts.seasonal_plot(_monthly_series(144), period=12)
    categories = list(p.data["cycle"].cat.categories)
    assert p.data["cycle"].cat.ordered
    assert categories == [str(c) for c in range(1, 13)]


# The colour rows of check_figure() that a ramp of cycle colours can pass.
# redundant_encoding is left out: colour alone tells the cycles apart.
CYCLE_COLOUR_ROWS = [
    "colour_separability", "colour_separability_protan",
    "colour_separability_deutan", "colour_separability_tritan",
    "greyscale_separability", "geometry_contrast",
]


def test_seasonal_cycle_lines_clear_3_to_1_against_white():
    # The lines took the sequential ramp, which starts at #e6eff5, 1.16:1
    # against white, so with twelve cycles four of the lines fell below the
    # 3:1 that WCAG 2.2 asks of a graphical object. The ramp now runs from
    # #4a8fc0, at 3.51:1, through the brand blue to #08315a, as in the R twin.
    # Twelve cycles is past the eight qualitative colours, and the ramp takes
    # them without the warning an interpolated qualitative palette gives.
    import depictr as dp

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = ts.seasonal_plot(_monthly_series(144), period=12)
    scale = next(s for s in p.scales if "color" in s.aesthetics)
    ramp = list(scale.palette(12))
    assert (ramp[0], ramp[-1]) == ("#4a8fc0", "#08315a")
    # Fills keep the sequential palette, pale end and all.
    assert dp.depictr_palette(3, kind="sequential")[0] == "#e6eff5"
    for n_cycles in (3, 12):
        report = dp.check_figure(ts.seasonal_plot(_monthly_series(12 * n_cycles)))
        row = report.set_index("check").loc["geometry_contrast"]
        assert (row["measured"], row["verdict"]) == (3.51, "pass")


def test_seasonal_cycle_colours_pass_every_colour_check_up_to_six_cycles():
    # Past about six cycles, a single-hue ramp shows the order of the cycles
    # more than which is which. Measured with CIEDE2000, six cycles clear 5
    # by the least margin under deuteranopia, at 5.24 (the R twin's ramp,
    # rounded one hex unit away, measures 4.93 there). Every colour row fails
    # from seven cycles and greyscale separability from eight, so none is
    # asserted there.
    import depictr as dp

    for n_cycles in range(2, 7):
        report = dp.check_figure(ts.seasonal_plot(_monthly_series(12 * n_cycles)))
        verdicts = report.set_index("check").loc[CYCLE_COLOUR_ROWS, "verdict"]
        assert (verdicts == "pass").all(), (n_cycles, verdicts.to_dict())


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"]


def _x_scale(p):
    """The breaks and labels given to a plot's x scale."""
    scale = next(s for s in p.scales if "x" in s.aesthetics)
    return list(scale.breaks), scale.labels


def test_seasonal_plot_takes_the_month_from_the_index():
    # The position was counted from the first observation, so a series
    # starting in April drew April at position 1. The R twin reads the month
    # from the ts, as stats::cycle() does, and puts April at 4. An index whose
    # frequency pandas infers counts as well as one that states it.
    starts = [
        pd.period_range("2016-04", periods=24, freq="M"),
        pd.date_range("2016-04-01", periods=24, freq="MS"),
        pd.date_range("2016-04-30", periods=24, freq="ME"),
        pd.DatetimeIndex(list(pd.date_range("2016-04-01", periods=24, freq="MS"))),
    ]
    for idx in starts:
        p = ts.seasonal_plot(pd.Series(np.arange(24.0), index=idx))
        assert p.data["position"].tolist() == (
            list(range(4, 13)) + list(range(1, 13)) + [1, 2, 3])
        assert p.data["cycle"].astype(int).tolist() == [1] * 9 + [2] * 12 + [3] * 3
        assert _x_scale(p) == (list(range(1, 13)), MONTHS)


def test_seasonal_plot_breaks_cycles_at_the_turn_of_the_year():
    # A series from March 2016 that peaks every July. The peak was drawn at
    # position 5, and December 2016 and January 2017 shared a cycle. As in the
    # R twin, the peak sits at Jul and each cycle runs from January to
    # December, the first and last of them short.
    idx = pd.period_range("2016-03", periods=36, freq="M")
    series = pd.Series(np.where(idx.month == 7, 10.0, 1.0), index=idx)
    data = ts.seasonal_plot(series).data
    peak = data.loc[data["value"] == 10.0, "position"].unique().tolist()
    assert [MONTHS[p - 1] for p in peak] == ["Jul"]
    assert data["cycle"].astype(int).tolist() == (
        [1] * 10 + [2] * 12 + [3] * 12 + [4] * 2)
    assert list(data["cycle"].cat.categories) == ["1", "2", "3", "4"]


def test_seasonal_plot_takes_quarters_and_weekdays_from_the_index():
    # A quarterly series from the third quarter starts at Q3. A PeriodIndex
    # with a fiscal year numbers its quarters from that year's start, as
    # Period.quarter does: 2016Q3 of a year ending in March is October to
    # December 2015. A daily index sets the weekday, Monday first, which a
    # DatetimeIndex knows and the R twin's plain ts does not.
    quarterly = pd.Series(np.arange(8.0),
                          index=pd.period_range("2015Q3", periods=8, freq="Q"))
    p = ts.seasonal_plot(quarterly)
    assert p.data["position"].tolist() == [3, 4, 1, 2, 3, 4, 1, 2]
    assert p.data["cycle"].astype(int).tolist() == [1, 1, 2, 2, 2, 2, 3, 3]
    assert _x_scale(p) == ([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
    fiscal = pd.Series(np.arange(6.0),
                       index=pd.period_range("2016Q3", periods=6, freq="Q-MAR"))
    assert ts.seasonal_plot(fiscal).data["position"].tolist() == [3, 4, 1, 2, 3, 4]
    # 6 January 2016 was a Wednesday.
    daily = pd.Series(np.arange(10.0),
                      index=pd.date_range("2016-01-06", periods=10, freq="D"))
    p = ts.seasonal_plot(daily)
    assert p.data["position"].tolist() == [3, 4, 5, 6, 7, 1, 2, 3, 4, 5]
    assert _x_scale(p) == (list(range(1, 8)),
                           ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])


def test_seasonal_plot_counts_from_the_first_observation_without_a_calendar():
    # A bare array has no calendar to read, and nor has a monthly index read
    # with a period its frequency does not imply. Both count the positions from
    # the first observation and label them by number.
    p = ts.seasonal_plot(np.arange(30.0), period=12)
    assert p.data["position"].tolist() == list(range(1, 13)) * 2 + list(range(1, 7))
    assert p.data["cycle"].astype(int).tolist() == [1] * 12 + [2] * 12 + [3] * 6
    assert _x_scale(p) == (list(range(1, 13)), True)
    monthly = pd.Series(np.arange(12.0),
                        index=pd.period_range("2016-04", periods=12, freq="M"))
    p = ts.seasonal_plot(monthly, period=6)
    assert p.data["position"].tolist() == list(range(1, 7)) * 2
    assert _x_scale(p) == (list(range(1, 7)), True)


def test_minutes_and_milliseconds_are_not_read_as_months():
    # The period was read from the first letter of the index frequency, upper
    # cased, and pandas names minutes "min" and milliseconds "ms", so both were
    # taken for monthly data with a period of 12. Hours, business days and
    # two-monthly data imply no period either.
    for freq in ("min", "ms", "h", "B", "2ME"):
        s = pd.Series(np.arange(48.0),
                      index=pd.date_range("2016-01-01", periods=48, freq=freq))
        with pytest.raises(ValueError, match="Could not infer the seasonal period"):
            ts.seasonal_plot(s)
        # Given a period, the positions count from the first observation. Read
        # from the calendar, every minute would be in January.
        positions = ts.seasonal_plot(s, period=12).data["position"].tolist()
        assert positions == list(range(1, 13)) * 4
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match="Could not infer the seasonal period"):
        ts.decompose_plot(s)


def test_month_and_quarter_aliases_still_infer_their_period():
    # pandas names a month by its end or start and a quarter by its end or
    # start with the month that ends or starts the year, as "QS-OCT".
    for freq, period in (("ME", 12), ("MS", 12), ("QE", 4), ("QS", 4),
                         ("QS-OCT", 4), ("QE-MAR", 4), ("D", 7)):
        s = pd.Series(np.arange(30.0),
                      index=pd.date_range("2016-01-01", periods=30, freq=freq))
        assert ts._infer_period(s, None) == period, freq


def test_a_period_that_is_not_whole_is_rounded_down_with_a_warning():
    # int() truncated such a period without a word, and the R twin refused it
    # with a message about season labels that no labels could satisfy. Both
    # twins now use the whole seasons and say so.
    message = "Period 52.17857 is not a whole number; using 52 seasons per cycle."
    with pytest.warns(UserWarning, match=re.escape(message)):
        p = ts.seasonal_plot(np.arange(130.0), period=365.25 / 7)
    assert p.data["position"].tolist() == (
        list(range(1, 53)) * 2 + list(range(1, 27)))
    assert _builds(p)
    # A period a rounding error away from a whole number is that number, as
    # R's ts() takes a frequency within getOption("ts.eps") of one, and a
    # calendar index still sets the seasons. The R suite pins the same April
    # start for a frequency of 12.0000001.
    april = pd.Series(np.arange(24.0),
                      index=pd.period_range("2016-04", periods=24, freq="M"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = ts.seasonal_plot(np.arange(36.0), period=0.1 * 3 * 40)
        q = ts.seasonal_plot(april, period=12 + 1e-7)
    assert p.data["position"].max() == 12
    assert q.data["position"].tolist() == (
        list(range(4, 13)) + list(range(1, 13)) + [1, 2, 3])


def test_a_period_below_two_is_refused():
    # The R twin refuses a frequency below 2. Here a period of 1, or one of 1.5
    # truncated to 1, drew every observation at one position, each in a cycle
    # of its own.
    for period in (1, 1.5, 0, -12):
        with pytest.raises(ValueError, match=re.escape("(period >= 2)")):
            ts.seasonal_plot(np.arange(24.0), period=period)


def test_annual_index_is_not_a_seasonal_period():
    # An annual index once mapped to period 1, which is not a seasonal period:
    # seasonal_plot drew one point per "cycle" and neither function complained.
    annual = pd.Series(
        np.arange(20.0), index=pd.period_range("2000", periods=20, freq="Y")
    )
    with pytest.raises(ValueError, match="Could not infer the seasonal period"):
        ts.seasonal_plot(annual)
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match="Could not infer the seasonal period"):
        ts.decompose_plot(annual)


def test_seasonal_plot_rejects_an_empty_series():
    # The cycle count now comes from cycle.max(), so an empty series has to be
    # turned away by name rather than by a numpy reduction error.
    with pytest.raises(ValueError, match="no non-missing values"):
        ts.seasonal_plot(np.array([]), period=12)


def test_internal_missing_values_are_refused():
    # Dropping an internal NaN closed up the gap silently: the ACF correlated
    # values across it, the decomposition misaligned, and every post-gap
    # observation landed on the wrong within-period position.
    s = _monthly_series()
    s.iloc[40] = np.nan
    with pytest.raises(ValueError, match="internal missing values"):
        ts.seasonal_plot(s)
    with pytest.raises(ValueError, match="internal missing values"):
        ts.seasonal_plot(s.to_numpy(), period=12)
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match="internal missing values"):
        ts.acf_plot(s)
    with pytest.raises(ValueError, match="internal missing values"):
        ts.decompose_plot(s)


def test_missing_values_at_the_ends_are_trimmed():
    # Trimming the ends only shortens the series; no observation moves relative
    # to another, so it stays legitimate where an internal drop is not.
    s = _monthly_series()
    s.iloc[[0, -1]] = np.nan
    assert _builds(ts.seasonal_plot(s))
    pytest.importorskip("statsmodels")
    assert _builds(ts.acf_plot(s))
    assert _builds(ts.decompose_plot(s))


def test_all_missing_series_keeps_its_existing_error():
    # An all-missing series trims to nothing, so it reaches the same named
    # refusal an empty series does rather than the internal-missing error.
    with pytest.raises(ValueError, match="no non-missing values"):
        ts.seasonal_plot(pd.Series([np.nan] * 24), period=12)
    # acf_plot took log10 of a zero length and raised an OverflowError from the
    # int() around it, so it is held to the same named refusal.
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match="no non-missing values"):
        ts.acf_plot(pd.Series([np.nan] * 24))
    with pytest.raises(ValueError, match="no non-missing values"):
        ts.acf_plot(np.array([]))


def test_daily_index_infers_a_weekly_period():
    daily = pd.Series(
        np.arange(40.0),
        index=pd.period_range("2000-01-01", periods=40, freq="D"),
    )
    assert ts._infer_period(daily, None) == 7


def test_timeseries_plot_raw_and_rolling():
    assert _builds(ts.timeseries_plot(S))
    assert _builds(ts.timeseries_plot(V, rolling=12))


def _draw(p):
    """Draw `p` once, returning its lines as drawn and plotnine's warnings.

    Each line is an (n, 2) array of x and y, in layer order. plotnine hands
    matplotlib one Line2D per path and keeps an interior missing value as a NaN
    vertex, where matplotlib leaves a gap. A NaN row inside an array is
    therefore a break in the drawn line. A plot is drawn only once here, since
    drawing it again adds its layers to the same figure a second time.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = p.draw(show=False)
    try:
        lines = [a.get_xydata() for a in fig.axes[0].get_children()
                 if isinstance(a, Line2D)]
    finally:
        plt.close(fig)
    notes = [str(w.message) for w in caught if issubclass(w.category, PlotnineWarning)]
    return lines, notes


def test_timeseries_plot_keeps_a_gap_in_place():
    # Dropping the missing value closed up the gap: 6.0 was drawn at x = 5,
    # every later point one step to the left, and the line ran from 4 to 6.
    values = np.arange(20.0)
    values[5] = np.nan
    p = ts.timeseries_plot(pd.Series(values))
    assert p.data["x"].tolist() == list(range(20))
    assert p.data.loc[p.data["x"] == 6, "value"].item() == 6.0
    (line,), _ = _draw(p)
    np.testing.assert_array_equal(line[:, 0], np.arange(20.0))
    assert np.isnan(line[5, 1])
    np.testing.assert_array_equal(np.delete(line[:, 1], 5), np.delete(values, 5))


def test_timeseries_plot_rolling_mean_is_missing_across_a_gap():
    # The centred mean was taken over the closed-up series, so the windows next
    # to the gap averaged values from both sides of it (4.33 and 5.67 here).
    values = np.arange(20.0)
    values[5] = np.nan
    (_, roll), _ = _draw(ts.timeseries_plot(pd.Series(values), rolling=3))
    # The first and last windows run off the series and are not drawn.
    np.testing.assert_array_equal(roll[:, 0], np.arange(1.0, 19.0))
    # The windows centred on 4, 5 and 6 reach the gap, so the overlay breaks
    # there. Elsewhere, the centred mean of a straight line is the line itself.
    gap = np.isin(roll[:, 0], [4, 5, 6])
    assert np.isnan(roll[gap, 1]).all()
    np.testing.assert_allclose(roll[~gap, 1], roll[~gap, 0])


def test_timeseries_plot_draws_an_object_column_with_its_gap():
    # Python floats and None arrive with the object dtype. plotnine read such
    # values as categories, one group each, so no line was drawn at all, with
    # or without the None.
    s = pd.Series([1.0, None, 3.0, 4.0], dtype=object)
    (line,), _ = _draw(ts.timeseries_plot(s))
    np.testing.assert_array_equal(line[:, 0], [0, 1, 2, 3])
    assert np.isnan(line[1, 1])


def test_timeseries_plot_keeps_a_missing_month_on_the_date_axis():
    # With June missing, the line ran straight from May to July.
    idx = pd.period_range("2020-01", periods=12, freq="M")
    values = np.arange(1.0, 13.0)
    values[5] = np.nan
    p = ts.timeseries_plot(pd.Series(values, index=idx))
    assert list(p.data["x"]) == list(idx.to_timestamp())
    (line,), _ = _draw(p)
    assert len(line) == 12
    assert np.isnan(line[5, 1])


def test_timeseries_plot_counts_positions_from_the_first_entry():
    # Missing values at either end keep their positions too, and the ends that
    # cannot be drawn are left out without a warning.
    values = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, np.nan])
    p = ts.timeseries_plot(pd.Series(values), rolling=3)
    assert p.data["x"].tolist() == list(range(7))
    (line, roll), notes = _draw(p)
    assert notes == []
    np.testing.assert_array_equal(line, [[2, 1], [3, 2], [4, 3], [5, 4]])
    np.testing.assert_array_equal(roll, [[3, 2], [4, 3]])


def test_timeseries_plot_rolling_overlay_does_not_warn_on_complete_data():
    # The undrawable ends of the overlay raised "geom_path: Removed 11 rows
    # containing missing values" for every complete series.
    _, notes = _draw(ts.timeseries_plot(V, rolling=12))
    assert notes == []


def _overlay(p):
    """The values of the rolling-mean overlay of a timeseries_plot() figure."""
    return p.layers[1].geom.data["roll"].to_numpy()


def test_an_even_window_is_the_centred_2_by_m_average():
    # pandas' rolling(4, center=True) put the mean of observations 1 to 4 at
    # observation 3, half a step after its centre, and the R twin put it half a
    # step before. A centred moving average of even order averages two
    # adjacent windows, which gives the observations at either end half weight
    # (Hyndman & Athanasopoulos, 2021, Section 3.3). The R twin's tests pin the
    # same values, and dev/goldens/timeseries.R prints them from it.
    roll = _overlay(ts.timeseries_plot(pd.Series(np.arange(1.0, 11.0)), rolling=4))
    np.testing.assert_allclose(roll, [np.nan, np.nan, 3, 4, 5, 6, 7, 8, np.nan, np.nan])
    # An odd window was already centred on its observation and is unchanged.
    roll = _overlay(ts.timeseries_plot(V, rolling=5))
    np.testing.assert_array_equal(roll, pd.Series(V).rolling(5, center=True).mean())


def test_a_twelve_month_average_is_the_classical_trend():
    # The 2 x 12 average is the trend of a classical decomposition of monthly
    # data, which statsmodels computes on its own.
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.seasonal import seasonal_decompose

    roll = _overlay(ts.timeseries_plot(S, rolling=12))
    np.testing.assert_allclose(roll, seasonal_decompose(V, period=12).trend, rtol=1e-10)


def test_an_even_window_breaks_at_a_gap():
    # The 2 x 4 average spans five observations, so the windows centred on 3
    # to 7 reach the missing value at 5. Elsewhere, the centred average of a
    # straight line is the line itself.
    values = np.arange(20.0)
    values[5] = np.nan
    roll = _overlay(ts.timeseries_plot(pd.Series(values), rolling=4))
    missing = np.isnan(roll)
    assert np.flatnonzero(missing).tolist() == [0, 1, 3, 4, 5, 6, 7, 18, 19]
    np.testing.assert_allclose(roll[~missing], np.flatnonzero(~missing))


def test_an_even_window_longer_than_the_series_draws_nothing():
    # A 2 x 4 average needs five observations. On four, every window runs off
    # the series, as an odd window longer than the series does.
    roll = _overlay(ts.timeseries_plot(pd.Series([1.0, 2.0, 3.0, 4.0]), rolling=4))
    assert len(roll) == 4
    assert np.isnan(roll).all()


def test_several_columns_are_refused():
    # A frame with two columns was read row by row into one series (1, 100, 2,
    # 200 and so on), and timeseries_plot() dropped its index as well.
    frame = pd.DataFrame({"a": V, "b": V + 100}, index=S.index)
    two = re.escape("`x` has 2 columns; pass one series at a time.")
    three = re.escape("`x` has 3 columns; pass one series at a time.")
    array = np.column_stack([V, V, V])
    with pytest.raises(ValueError, match=two):
        ts.timeseries_plot(frame)
    with pytest.raises(ValueError, match=three):
        ts.timeseries_plot(array)
    with pytest.raises(ValueError, match=two):
        ts.seasonal_plot(frame)
    with pytest.raises(ValueError, match=three):
        ts.seasonal_plot(array, period=12)
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match=two):
        ts.acf_plot(frame)
    with pytest.raises(ValueError, match=three):
        ts.acf_plot(array, kind="pacf")
    with pytest.raises(ValueError, match=two):
        ts.decompose_plot(frame)
    with pytest.raises(ValueError, match=three):
        ts.decompose_plot(array, period=12)


def test_a_single_column_is_the_series():
    # A one-column frame keeps its index, so the period and the date axis come
    # from it as they do from the Series.
    frame = S.to_frame("sales")
    pd.testing.assert_frame_equal(ts.timeseries_plot(frame).data,
                                  ts.timeseries_plot(S).data)
    pd.testing.assert_frame_equal(ts.seasonal_plot(frame).data,
                                  ts.seasonal_plot(S).data)
    pd.testing.assert_frame_equal(ts.timeseries_plot(V.reshape(-1, 1)).data,
                                  ts.timeseries_plot(V).data)
    pytest.importorskip("statsmodels")
    pd.testing.assert_frame_equal(ts.acf_plot(frame).data, ts.acf_plot(S).data)
    pd.testing.assert_frame_equal(ts.decompose_plot(frame).data,
                                  ts.decompose_plot(S).data)


def test_object_values_are_read_as_numbers():
    # The seasonal plot took the values of an object column as they came, and
    # plotnine drew them as categories, one evenly spaced tick per value.
    obj = S.astype(object)
    expected = ts.seasonal_plot(S).data
    pd.testing.assert_frame_equal(ts.seasonal_plot(obj).data, expected)
    pd.testing.assert_frame_equal(ts.seasonal_plot(obj.to_frame("sales")).data,
                                  expected)
