"""depictr gallery: a Streamlit shop window for the package.

Pick a bundled dataset (or upload a CSV), choose a plot, and see the depictr
figure together with the exact Python call that made it. A colourblind-vision
toggle re-renders the whole figure as each deficiency would be seen, so a reader
can judge the accessibility of the default palette on the figure itself.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import keyword
import unicodedata

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

import depictr as dp

st.set_page_config(page_title="depictr gallery", layout="wide")


def show(plot, code: str, deficiency: str):
    """Render a depictr plot under the chosen vision, beside its source call."""
    # Normal vision is any deficiency at severity zero, which gives the
    # figure's pixels as drawn.
    if deficiency == "normal":
        image = dp.simulate_figure(plot, "deutan", severity=0)
    else:
        image = dp.simulate_figure(plot, deficiency)
    left, right = st.columns([3, 2])
    with left:
        st.image(image, width="stretch")
    with right:
        st.caption("The exact call")
        st.code(code, language="python")


class _Source:
    """An argument whose value is passed to the plot and whose code is shown.

    The data, its columns, a series and a fitted model are shown as the code
    that makes them, such as ``dp.crop_yield()['yield']``, where any other
    argument is shown as its repr.
    """

    def __init__(self, value, code: str):
        self.value, self.code = value, code


def _value(arg):
    return arg.value if isinstance(arg, _Source) else arg


def _code(arg):
    return arg.code if isinstance(arg, _Source) else repr(arg)


def draw(fn: str, *args, setup: str = "", **kwargs):
    """Call ``dp.<fn>`` and show its figure beside the call that made it.

    The call is written from the arguments the function receives, so the two
    cannot drift apart. Each call used to be typed out beside its plot, and
    several no longer matched it. The estimation call left out its seed and
    the PCA call its grouping, and the survival call carried a literal
    ``group=...``. ``setup`` holds any lines the call needs first, such as a
    model fit.
    """
    plot = getattr(dp, fn)(*map(_value, args),
                           **{name: _value(arg) for name, arg in kwargs.items()})
    call = ", ".join([*map(_code, args),
                      *(f"{name}={_code(arg)}" for name, arg in kwargs.items())])
    show(plot, f"{setup}dp.{fn}({call})", deficiency)


def _read_upload(upload):
    """An uploaded CSV as a data frame, with the encoding that read it.

    Excel on Windows saves a CSV in the system's code page, cp1252 in Western
    Europe, where an accented letter is a byte that is not valid UTF-8. UTF-8
    is tried first, then cp1252, then latin-1, which gives every byte a
    character and so cannot fail on the encoding. The bytes are parsed from a
    fresh buffer each time, since a failed attempt leaves the upload's read
    position wherever pandas stopped.
    """
    raw = upload.getvalue()
    for encoding in ("utf-8", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding), encoding
        except UnicodeDecodeError:
            pass
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1"), "latin-1"


class _NothingToOffer(Exception):
    """A choice the pane needs has no column of this data to offer."""


def _choose(label, options, why, **kwargs):
    """A selectbox the pane cannot draw without.

    Given no options, ``st.selectbox`` returns None, and the plot fails on
    ``data[None]`` with the bare message "None". With no options, this raises
    instead, and the handler at the foot of the file shows ``why``, what the
    pane needs, in place of a figure.
    """
    options = list(options)
    if not options:
        raise _NothingToOffer(why)
    return st.selectbox(label, options, **kwargs)


NEEDS_NUMERIC = "This plot needs a numeric column, and this dataset has none."
NEEDS_GROUP = ("This plot needs a categorical column with 2 to 12 distinct "
               "values, and this dataset has none.")


# ---- sidebar: data and vision -------------------------------------------------
st.sidebar.title("depictr")
st.sidebar.caption("Unified, colourblind-safe statistical visualisation")

source = st.sidebar.radio("Data", ["Bundled dataset", "Upload a CSV"])
read_error = None
if source == "Upload a CSV":
    upload = st.sidebar.file_uploader("CSV file", type="csv")
    data, data_call = None, "pd.read_csv(your_file)"
    if upload is not None:
        # The read sits outside the handler at the foot of the file. A file
        # pandas cannot parse, such as an empty or ragged one, is caught here
        # instead and reported below the title.
        try:
            data, encoding = _read_upload(upload)
        except Exception as exc:
            read_error = f"Could not read {upload.name} as a CSV: {exc}"
        else:
            # pandas reads UTF-8 by default, so only another encoding needs
            # naming for the displayed call to read the file the same way.
            if encoding != "utf-8":
                data_call = f"pd.read_csv(your_file, encoding={encoding!r})"
else:
    name = st.sidebar.selectbox("Dataset", list(dp.DATASETS))
    data = dp.DATASETS[name]()
    data_call = f"dp.{name}()"

deficiency = st.sidebar.selectbox(
    "Simulate vision", ["normal", "deutan", "protan", "tritan"],
    help="Re-renders the figure as each colour-vision deficiency would see it.",
)

legend_in = st.sidebar.checkbox(
    "Legend inside plot", value=False,
    help="For the plots that support it, tucks the legend into the panel, "
         "so the figure needs no right-hand margin.",
)
report = dp.palette_safety()
st.sidebar.metric("Palette min ΔE (worst case)", report["min_delta_e"],
                  help=f"Safe: {report['safe']} (worst under {report['worst_condition']})")

st.title("depictr gallery")
st.write(
    "One consistent, colourblind-safe theme across the whole statistical "
    "workflow. Every plot below is a depictr one-liner returning a plotnine "
    "object you can extend with `+`."
)

if read_error is not None:
    st.error(read_error)
    st.stop()
if data is None:
    st.info("Upload a CSV to begin.")
    st.stop()

st.dataframe(data.head(), width="stretch")
num_cols = list(data.select_dtypes("number").columns)
# A grouping column is offered only when it has 2 to 12 distinct values. An
# identifier (participant, item, field, id, patient) has dozens or hundreds,
# which would draw a legend nobody can read, and one value is no grouping.
cat_cols = [c for c in data.columns
            if c not in num_cols and 2 <= data[c].nunique() <= 12]
binary_cols = [c for c in data.columns if data[c].nunique() == 2]
# The ROC/PR/gain/lift/calibration/threshold family does arithmetic on y_true
# (a cumulative sum, for one), so it needs a column coded 0/1. Any other
# two-level column counts no positives and meets the single-class refusal,
# whose wording describes the wrong problem. confusion_matrix_plot carries no
# such constraint and keeps the wider list. When no column is coded 0/1, the
# pane says so through _choose(). The wider list used to be offered in that
# case, since an empty selectbox left the family unreachable with no word of
# why, and the visitor met the misleading refusal.
binary01_cols = [c for c in binary_cols
                 if set(pd.unique(data[c].dropna())) <= {0, 1}]
DATA = _Source(data, data_call)


def _column(name):
    """A column of the data, shown as the code that selects it."""
    return _Source(data[name], f"{data_call}[{name!r}]")


def _opt(label, options):
    """A '(none)' + options selectbox returning None for '(none)'."""
    choice = st.selectbox(label, ["(none)"] + list(options))
    return None if choice == "(none)" else choice


def _term(name):
    """A column name as a formula term, quoted with Q() only where it must be.

    patsy reads a bare name as Python, so a keyword (crop_yield's ``yield``)
    or a name that is not an identifier (one with a space) must be quoted. A
    name that NFKC normalisation changes must be quoted too, since Python
    normalises identifiers when it compiles them, and a bare "ﬁeld" would look
    up a column "field". Any other name stays bare, keeping the model's term
    names and the call free of quoting. A JSON string is also a valid Python
    string literal, escapes included, so ``json.dumps`` quotes any name safely.
    """
    if (name.isidentifier() and not keyword.iskeyword(name)
            and unicodedata.normalize("NFKC", name) == name):
        return name
    return f"Q({json.dumps(name, ensure_ascii=False)})"


def _fit_ols(outcome, predictors):
    """A quick OLS fit, with the two lines of code that make it."""
    import statsmodels.formula.api as smf

    formula = f"{_term(outcome)} ~ {' + '.join(map(_term, predictors))}"
    setup = ("import statsmodels.formula.api as smf\n"
             f"model = smf.ols({formula!r}, {data_call}).fit()\n")
    return _Source(smf.ols(formula, data).fit(), "model"), setup


FAMILIES = {
    "Exploratory": ["Distribution", "Categorical", "ECDF", "Ridgeline",
                    "Raincloud", "Dumbbell", "Outliers", "Group comparison",
                    "Scatter + trend", "Correlation heatmap", "Pairs matrix",
                    "Missingness map"],
    "Estimation": ["Estimation plot"],
    "Multivariate": ["PCA biplot", "Scree", "Clusters", "Dendrogram",
                     "Silhouette"],
    "Classification": ["ROC curve", "PR curve", "Gains", "Lift", "Calibration",
                       "Confusion matrix", "Threshold"],
    "Survival": ["Kaplan-Meier"],
    "Time series": ["Series", "ACF / PACF",
                    "Seasonal plot (one line per cycle)", "Decomposition"],
    "Models (fits an OLS)": ["Coefficients", "Residual diagnostics",
                             "Effect of a predictor"],
}
family = st.selectbox("Family", list(FAMILIES))
stage = st.selectbox("Plot", FAMILIES[family])

try:
    if stage == "Distribution":
        x = _choose("Numeric variable", num_cols, NEEDS_NUMERIC)
        g = _opt("Group (optional)", cat_cols)
        draw("explore_distribution", DATA, x, group=g, kind="both",
             legend_inside=legend_in)
    elif stage == "Categorical":
        x = _choose("Category", cat_cols, NEEDS_GROUP)
        g = _opt("Group (optional)", cat_cols)
        draw("explore_categorical", DATA, x, group=g)
    elif stage == "ECDF":
        x = _choose("Numeric variable", num_cols, NEEDS_NUMERIC)
        g = _opt("Group (optional)", cat_cols)
        draw("ecdf_plot", DATA, x, group=g, legend_inside=legend_in)
    elif stage == "Ridgeline":
        x = _choose("Numeric variable", num_cols, NEEDS_NUMERIC)
        g = _choose("Group", cat_cols, NEEDS_GROUP)
        draw("ridgeline_plot", DATA, x, g)
    elif stage == "Raincloud":
        x = _choose("Numeric variable", num_cols, NEEDS_NUMERIC)
        g = _opt("Group (optional)", cat_cols)
        draw("raincloud_plot", DATA, x, group=g)
    elif stage == "Dumbbell":
        cat = _choose("Category", cat_cols, NEEDS_GROUP)
        val = _choose("Value", num_cols, NEEDS_NUMERIC)
        # dumbbell_plot needs a group with exactly two levels, categorical or
        # numeric, in a column apart from the category and the value, or its
        # pivot collides. The pane offers only such columns. It used to fall
        # back to any other category, or to the category itself, and so opened
        # on a column the plot refuses.
        g = _choose("Two-level group", [c for c in binary_cols if c not in (cat, val)],
                    "This plot needs a two-level column other than its category "
                    "and value, and this dataset has none.")
        draw("dumbbell_plot", DATA, cat, val, g, legend_inside=legend_in)
    elif stage == "Outliers":
        x = _choose("Numeric variable", num_cols, NEEDS_NUMERIC)
        draw("outlier_plot", DATA, x)
    elif stage == "Group comparison":
        x = _choose("Numeric variable", num_cols, NEEDS_NUMERIC)
        g = _choose("Group", cat_cols, NEEDS_GROUP)
        draw("group_comparison_plot", DATA, x, g)
    elif stage == "Scatter + trend":
        x = _choose("x", num_cols, NEEDS_NUMERIC)
        y = _choose("y", [c for c in num_cols if c != x],
                    "This plot needs a second numeric column, and this dataset "
                    "has none.")
        g = _opt("Group (optional)", cat_cols)
        draw("scatter_trend", DATA, x, y, group=g)
    elif stage == "Correlation heatmap":
        draw("correlation_heatmap", DATA)
    elif stage == "Pairs matrix":
        cols = st.multiselect("Numeric columns (2-5)", num_cols, default=num_cols[:4])
        draw("explore_pairs", DATA, cols=cols or None)
    elif stage == "Missingness map":
        draw("missingness_map", DATA, legend_inside=legend_in)
    elif stage == "Estimation plot":
        y = _choose("Outcome", num_cols, NEEDS_NUMERIC)
        g = _choose("Group", cat_cols, NEEDS_GROUP)
        two = st.checkbox("Two-panel (Cumming)", value=True)
        # The seed fixes the bootstrap and the jitter, so the call shown
        # redraws this figure.
        draw("estimation_plot", DATA, y, g, two_panel=two, seed=1)
    elif stage == "PCA biplot":
        g = _opt("Colour by (optional)", cat_cols)
        draw("pca_plot", DATA, group=g)
    elif stage in {"Scree", "Clusters", "Dendrogram", "Silhouette"}:
        fn = {"Scree": "scree_plot", "Clusters": "cluster_plot",
              "Dendrogram": "dendrogram_plot", "Silhouette": "silhouette_plot"}[stage]
        draw(fn, DATA)
    elif stage in {"ROC curve", "PR curve", "Gains", "Lift", "Calibration",
                   "Threshold"}:
        target = _choose("Binary outcome (0/1)", binary01_cols,
                         "This plot needs an outcome column coded 0/1, and this "
                         "dataset has none.")
        score = _choose("Score / probability", num_cols, NEEDS_NUMERIC)
        fn = {"ROC curve": "roc_curve_plot", "PR curve": "pr_curve_plot",
              "Gains": "gain_plot", "Lift": "lift_plot",
              "Calibration": "calibration_plot", "Threshold": "threshold_plot"}[stage]
        draw(fn, _column(target), _column(score))
    elif stage == "Confusion matrix":
        needs = "This plot needs a two-level column, and this dataset has none."
        target = _choose("Binary outcome", binary_cols, needs)
        pred = _choose("Predicted label", binary_cols, needs)
        draw("confusion_matrix_plot", _column(target), _column(pred), normalise="true")
    elif stage == "Kaplan-Meier":
        # survival_plot reads the event under the 0/1 coding (0 = censored) or
        # the survival::Surv() 1/2 coding (1 = censored) and refuses any other,
        # so only such columns are offered. There is no fallback to the other
        # two-level columns, since each would only meet the refusal. The
        # default is a column named event or status, since the first numeric
        # column is as likely to be the follow-up time.
        event_cols = [c for c in binary_cols
                      if set(pd.unique(data[c].dropna())) in ({0, 1}, {1, 2})]
        if not event_cols:
            st.info("The Kaplan-Meier plot needs an event column coded 0/1 "
                    "(0 = censored) or 1/2 (1 = censored), and this dataset "
                    "has none.")
        else:
            time = _choose("Time", num_cols, NEEDS_NUMERIC)
            names = [str(c).lower() for c in event_cols]
            default = next((names.index(n) for n in ("event", "status")
                            if n in names), 0)
            event = st.selectbox(
                "Event (0/1 or 1/2 coding)", event_cols, index=default,
                help="0 = censored and 1 = event, or the survival::Surv() "
                     "coding, 1 = censored and 2 = event.")
            g = _opt("Group (optional)", cat_cols)
            rt = st.checkbox("Number-at-risk table", value=True)
            group = {} if g is None else {"group": _column(g)}
            draw("survival_plot", _column(time), _column(event), **group,
                 risk_table=rt, legend_inside=legend_in, logrank=True)
    elif family == "Time series":
        # None of the datasets offered here is temporal (monthly_sales, which
        # holds two series in long format, is not among them), so this family
        # carries its own demo: the trend-plus-seasonality series from the docstring
        # examples in depictr.timeseries, seeded so the gallery is stable. Any
        # numeric column can stand in, read in row order.
        DEMO = "Demo monthly series (trend + seasonality)"
        pick = st.selectbox(
            "Series", [DEMO] + num_cols,
            help="Data columns are read in row order. The correlation, "
                 "seasonal and decomposition plots need a column with no "
                 "internal gaps, since closing a gap up would shift every "
                 "later observation.")
        if pick == DEMO:
            t = np.arange(120)
            series = pd.Series(
                50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12)
                + np.random.default_rng(0).normal(0, 3, 120),
                index=pd.period_range("2016-01", periods=120, freq="M"),
            )
            series = _Source(series, "series")
            # The mirror must show how the demo series was built, or the
            # displayed call would not be reproducible.
            code_prefix = (
                "t = np.arange(120)\n"
                "series = pd.Series(\n"
                "    50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12)\n"
                "    + np.random.default_rng(0).normal(0, 3, 120),\n"
                "    index=pd.period_range('2016-01', periods=120, freq='M'))\n"
            )
        else:
            series = _column(pick)
            code_prefix = ""
        if stage == "Series":
            roll = st.checkbox(
                "Rolling-mean overlay (12-point window)", value=True,
                help="A window of one seasonal period smooths the season "
                     "away, leaving the trend.")
            window = 12 if roll else None
            draw("timeseries_plot", series, rolling=window, setup=code_prefix)
        elif stage == "ACF / PACF":
            kind = st.selectbox("Kind", ["acf", "pacf"])
            draw("acf_plot", series, kind=kind, setup=code_prefix)
        else:
            # seasonal_plot and decompose_plot need the cycle length. The demo
            # series' monthly index would let it be inferred, but a plain data
            # column has no such index, so the widget is always shown.
            period = int(st.number_input("Seasonal period", min_value=2,
                                         value=12))
            fn = {"Seasonal plot (one line per cycle)": "seasonal_plot",
                  "Decomposition": "decompose_plot"}[stage]
            draw(fn, series, period=period, setup=code_prefix)
    elif family.startswith("Models"):
        outcome = _choose("Outcome (numeric)", num_cols, NEEDS_NUMERIC)
        # Identifiers are left out, as for the grouping choices: one dummy per
        # field or respondent would saturate the model.
        candidates = [c for c in data.columns
                      if c != outcome and (c in num_cols or c in cat_cols)]
        predictors = st.multiselect("Predictors", candidates,
                                    default=candidates[:2])
        if predictors:
            # Each model pane shows the import and the fit before its call, so
            # the snippet runs as it stands.
            model, fit_code = _fit_ols(outcome, predictors)
            if stage == "Coefficients":
                draw("coefficient_plot", model, setup=fit_code)
            elif stage == "Residual diagnostics":
                draw("residual_diagnostics_plot", model, setup=fit_code)
            elif stage == "Effect of a predictor":
                var = _choose("Predictor to vary",
                              [p for p in predictors if p in num_cols],
                              "This plot varies a numeric predictor, and none of "
                              "the predictors chosen is numeric.")
                draw("effects_plot", model, var, setup=fit_code)
        else:
            st.info("Choose at least one predictor.")
except _NothingToOffer as why:
    st.info(str(why))
except Exception as exc:  # surface a friendly message rather than a stack trace
    st.error(f"Could not draw that plot for this data: {exc}")
