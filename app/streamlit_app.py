"""depictr gallery: a Streamlit shop window for the package.

Pick a bundled dataset (or upload a CSV), choose a plot, and see the depictr
figure together with the exact Python call that made it. A colourblind-vision
toggle re-renders the whole figure as each deficiency would be seen, so the
accessibility of the default palette is visible, not just claimed.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

import depictr as dp
from depictr.cvd import _MACHADO_1, _linear_to_srgb, _srgb_to_linear

st.set_page_config(page_title="depictr gallery", layout="wide")


def simulate_image(fig, deficiency: str):
    """Return the figure's pixels as seen under a colour-vision deficiency."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3:]
    if deficiency == "normal":
        return np.concatenate([rgb, alpha], axis=-1)
    linear = _srgb_to_linear(rgb)
    simulated = _linear_to_srgb(linear @ _MACHADO_1[deficiency].T)
    return np.concatenate([np.clip(simulated, 0, 1), alpha], axis=-1)


def show(plot, code: str, deficiency: str):
    """Render a depictr plot under the chosen vision, beside its source call."""
    fig = plot.draw(show=False)
    left, right = st.columns([3, 2])
    with left:
        st.image(simulate_image(fig, deficiency), use_container_width=True)
    with right:
        st.caption("The exact call")
        st.code(code, language="python")


# ---- sidebar: data and vision -------------------------------------------------
st.sidebar.title("depictr")
st.sidebar.caption("Unified, colourblind-safe statistical visualisation")

source = st.sidebar.radio("Data", ["Bundled dataset", "Upload a CSV"])
if source == "Upload a CSV":
    upload = st.sidebar.file_uploader("CSV file", type="csv")
    data = pd.read_csv(upload) if upload else None
    data_call = "pd.read_csv(your_file)"
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
    help="For the plots that support it, tucks the legend into the panel "
         "instead of a right-hand margin.",
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

if data is None:
    st.info("Upload a CSV to begin.")
    st.stop()

st.dataframe(data.head(), use_container_width=True)
num_cols = list(data.select_dtypes("number").columns)
cat_cols = [c for c in data.columns if c not in num_cols]
binary_cols = [c for c in data.columns if data[c].nunique() == 2]
# The ROC/PR/gain/lift/calibration/threshold family does arithmetic on y_true
# (e.g. a cumulative sum), so it needs a genuine 0/1 column, not just any
# two-level one -- confusion_matrix_plot has no such constraint.
binary01_cols = [c for c in binary_cols
                 if set(pd.unique(data[c].dropna())) <= {0, 1}] or binary_cols


def _opt(label, options):
    """A '(none)' + options selectbox returning None for '(none)'."""
    choice = st.selectbox(label, ["(none)"] + list(options))
    return None if choice == "(none)" else choice


def _fit_ols(outcome, predictors):
    """Fit a quick OLS; Q() quotes names that are keywords or contain spaces."""
    import statsmodels.formula.api as smf

    rhs = " + ".join(f'Q("{p}")' for p in predictors)
    return smf.ols(f'Q("{outcome}") ~ {rhs}', data).fit()


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
    "Models (fits an OLS)": ["Coefficients", "Residual diagnostics",
                             "Effect of a predictor"],
}
family = st.selectbox("Family", list(FAMILIES))
stage = st.selectbox("Plot", FAMILIES[family])

try:
    if stage == "Distribution":
        x = st.selectbox("Numeric variable", num_cols)
        g = _opt("Group (optional)", cat_cols)
        show(dp.explore_distribution(data, x, group=g, kind="both",
                                     legend_inside=legend_in),
             f"dp.explore_distribution({data_call}, {x!r}, group={g!r}, "
             f"kind='both', legend_inside={legend_in})",
             deficiency)
    elif stage == "Categorical":
        x = st.selectbox("Category", cat_cols)
        g = _opt("Group (optional)", cat_cols)
        show(dp.explore_categorical(data, x, group=g),
             f"dp.explore_categorical({data_call}, {x!r}, group={g!r})", deficiency)
    elif stage == "ECDF":
        x = st.selectbox("Numeric variable", num_cols)
        g = _opt("Group (optional)", cat_cols)
        show(dp.ecdf_plot(data, x, group=g, legend_inside=legend_in),
             f"dp.ecdf_plot({data_call}, {x!r}, group={g!r}, "
             f"legend_inside={legend_in})", deficiency)
    elif stage == "Ridgeline":
        x = st.selectbox("Numeric variable", num_cols)
        g = st.selectbox("Group", cat_cols)
        show(dp.ridgeline_plot(data, x, g),
             f"dp.ridgeline_plot({data_call}, {x!r}, {g!r})", deficiency)
    elif stage == "Raincloud":
        x = st.selectbox("Numeric variable", num_cols)
        g = _opt("Group (optional)", cat_cols)
        show(dp.raincloud_plot(data, x, group=g),
             f"dp.raincloud_plot({data_call}, {x!r}, group={g!r})", deficiency)
    elif stage == "Dumbbell":
        cat = st.selectbox("Category", cat_cols)
        val = st.selectbox("Value", num_cols)
        # The group must differ from the category, or dumbbell_plot's pivot
        # collides on one column. Prefer another two-level column (categorical
        # or numeric -- dumbbell_plot only needs exactly two levels); fall back
        # to any other category, widening to cat_cols only as a last resort.
        g_options = ([c for c in binary_cols if c != cat]
                     or [c for c in cat_cols if c != cat] or cat_cols)
        g = st.selectbox("Two-level group", g_options)
        show(dp.dumbbell_plot(data, cat, val, g, legend_inside=legend_in),
             f"dp.dumbbell_plot({data_call}, {cat!r}, {val!r}, {g!r}, "
             f"legend_inside={legend_in})", deficiency)
    elif stage == "Outliers":
        x = st.selectbox("Numeric variable", num_cols)
        show(dp.outlier_plot(data, x),
             f"dp.outlier_plot({data_call}, {x!r})", deficiency)
    elif stage == "Group comparison":
        x = st.selectbox("Numeric variable", num_cols)
        g = st.selectbox("Group", cat_cols)
        show(dp.group_comparison_plot(data, x, g),
             f"dp.group_comparison_plot({data_call}, {x!r}, {g!r})", deficiency)
    elif stage == "Scatter + trend":
        x = st.selectbox("x", num_cols)
        y = st.selectbox("y", [c for c in num_cols if c != x])
        g = _opt("Group (optional)", cat_cols)
        show(dp.scatter_trend(data, x, y, group=g),
             f"dp.scatter_trend({data_call}, {x!r}, {y!r}, group={g!r})", deficiency)
    elif stage == "Correlation heatmap":
        show(dp.correlation_heatmap(data),
             f"dp.correlation_heatmap({data_call})", deficiency)
    elif stage == "Pairs matrix":
        cols = st.multiselect("Numeric columns (2-5)", num_cols, default=num_cols[:4])
        show(dp.explore_pairs(data, cols=cols or None),
             f"dp.explore_pairs({data_call}, cols={cols!r})", deficiency)
    elif stage == "Missingness map":
        show(dp.missingness_map(data, legend_inside=legend_in),
             f"dp.missingness_map({data_call}, legend_inside={legend_in})",
             deficiency)
    elif stage == "Estimation plot":
        y = st.selectbox("Outcome", num_cols)
        g = st.selectbox("Group", cat_cols)
        two = st.checkbox("Two-panel (Gardner-Altman)", value=True)
        show(dp.estimation_plot(data, y, g, two_panel=two, seed=1),
             f"dp.estimation_plot({data_call}, {y!r}, {g!r}, two_panel={two})",
             deficiency)
    elif stage in {"PCA biplot", "Scree", "Clusters", "Dendrogram", "Silhouette"}:
        g = _opt("Colour by (optional)", cat_cols) if stage == "PCA biplot" else None
        fn = {"PCA biplot": "pca_plot", "Scree": "scree_plot",
              "Clusters": "cluster_plot", "Dendrogram": "dendrogram_plot",
              "Silhouette": "silhouette_plot"}[stage]
        plot = (dp.pca_plot(data, group=g) if stage == "PCA biplot"
                else getattr(dp, fn)(data))
        show(plot, f"dp.{fn}({data_call})", deficiency)
    elif stage in {"ROC curve", "PR curve", "Gains", "Lift", "Calibration",
                   "Threshold"}:
        target = st.selectbox("Binary outcome (0/1)", binary01_cols)
        score = st.selectbox("Score / probability", num_cols)
        fn = {"ROC curve": "roc_curve_plot", "PR curve": "pr_curve_plot",
              "Gains": "gain_plot", "Lift": "lift_plot",
              "Calibration": "calibration_plot", "Threshold": "threshold_plot"}[stage]
        show(getattr(dp, fn)(data[target], data[score]),
             f"dp.{fn}({data_call}[{target!r}], {data_call}[{score!r}])", deficiency)
    elif stage == "Confusion matrix":
        target = st.selectbox("Binary outcome", binary_cols)
        pred = st.selectbox("Predicted label", binary_cols)
        show(dp.confusion_matrix_plot(data[target], data[pred], normalise="true"),
             f"dp.confusion_matrix_plot({data_call}[{target!r}], {data_call}[{pred!r}], normalise='true')",
             deficiency)
    elif stage == "Kaplan-Meier":
        time = st.selectbox("Time", num_cols)
        event = st.selectbox("Event (1=event, 0=censored)", num_cols)
        g = _opt("Group (optional)", cat_cols)
        rt = st.checkbox("Number-at-risk table", value=True)
        gv = None if g is None else data[g]
        show(dp.survival_plot(data[time], data[event], group=gv, risk_table=rt,
                              legend_inside=legend_in),
             f"dp.survival_plot({data_call}[{time!r}], {data_call}[{event!r}], "
             f"group=..., risk_table={rt}, legend_inside={legend_in})",
             deficiency)
    elif family.startswith("Models"):
        outcome = st.selectbox("Outcome (numeric)", num_cols)
        predictors = st.multiselect(
            "Predictors", [c for c in data.columns if c != outcome],
            default=[c for c in data.columns if c != outcome][:2])
        if predictors:
            model = _fit_ols(outcome, predictors)
            rhs = " + ".join(predictors)
            if stage == "Coefficients":
                show(dp.coefficient_plot(model),
                     f"dp.coefficient_plot(smf.ols('{outcome} ~ {rhs}', data).fit())",
                     deficiency)
            elif stage == "Residual diagnostics":
                show(dp.residual_diagnostics_plot(model),
                     "dp.residual_diagnostics_plot(model)", deficiency)
            elif stage == "Effect of a predictor":
                var = st.selectbox("Predictor to vary", [p for p in predictors if p in num_cols])
                show(dp.effects_plot(model, var),
                     f"dp.effects_plot(model, {var!r})", deficiency)
        else:
            st.info("Choose at least one predictor.")
except Exception as exc:  # surface a friendly message rather than a stack trace
    st.error(f"Could not draw that plot for this data: {exc}")
