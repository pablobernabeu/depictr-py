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

stage = st.selectbox(
    "Plot",
    ["Distribution", "Categorical", "Correlation heatmap", "Missingness map",
     "ROC curve", "Survival"],
)

try:
    if stage == "Distribution":
        x = st.selectbox("Numeric variable", num_cols)
        group = st.selectbox("Group (optional)", ["(none)"] + cat_cols)
        g = None if group == "(none)" else group
        show(dp.explore_distribution(data, x, group=g, kind="both"),
             f"dp.explore_distribution({data_call}, {x!r}, group={g!r}, kind='both')",
             deficiency)
    elif stage == "Categorical":
        x = st.selectbox("Category", cat_cols)
        group = st.selectbox("Group (optional)", ["(none)"] + cat_cols)
        g = None if group == "(none)" else group
        show(dp.explore_categorical(data, x, group=g),
             f"dp.explore_categorical({data_call}, {x!r}, group={g!r})", deficiency)
    elif stage == "Correlation heatmap":
        show(dp.correlation_heatmap(data),
             f"dp.correlation_heatmap({data_call})", deficiency)
    elif stage == "Missingness map":
        show(dp.missingness_map(data),
             f"dp.missingness_map({data_call})", deficiency)
    elif stage == "ROC curve":
        target = st.selectbox("Binary outcome", [c for c in data.columns
                                                 if data[c].nunique() == 2])
        score = st.selectbox("Score", num_cols)
        show(dp.roc_curve_plot(data[target], data[score]),
             f"dp.roc_curve_plot({data_call}[{target!r}], {data_call}[{score!r}])",
             deficiency)
    elif stage == "Survival":
        time = st.selectbox("Time", num_cols)
        event = st.selectbox("Event (1=event, 0=censored)", num_cols)
        group = st.selectbox("Group (optional)", ["(none)"] + cat_cols)
        g = None if group == "(none)" else data[group]
        show(dp.survival_plot(data[time], data[event], group=g),
             f"dp.survival_plot({data_call}[{time!r}], {data_call}[{event!r}], group=...)",
             deficiency)
except Exception as exc:  # surface a friendly message rather than a stack trace
    st.error(f"Could not draw that plot for this data: {exc}")
