"""The parity manifest: every argument of the R twin is accounted for here.

``tests/data/r_formals.json`` holds the formal arguments of every function the
R twin exports, written by ``dev/r_formals.R`` from a checkout of the R
package. Each R argument is matched with the Python parameter of the same name,
or with the one ``RENAMED`` gives, or is listed in ``R_ONLY``. Each Python
parameter is matched with an R argument or listed in ``PYTHON_ONLY``. A
signature change in either twin therefore fails here until the manifest says
what changed. The README's Status section lists the divergences a user would
notice, and the last test checks that it names each function only the R twin
has.

To port an argument, take it out of ``R_ONLY``. After a change to the R twin,
rerun ``Rscript dev/r_formals.R ../depictr`` and update the lists. The R
argument ``...`` is matched like any other: with ``*plots``, ``models`` or
``**kwargs`` where the Python parameter plays its part, and listed in
``R_ONLY`` otherwise.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

import depictr as dp

ROOT = Path(__file__).resolve().parents[1]
R_FORMALS = json.loads(
    (ROOT / "tests" / "data" / "r_formals.json").read_text(encoding="utf-8"))
R_FUNCTIONS = R_FORMALS["functions"]

# R's arguments that Python names differently, by function, from R's name to
# Python's.
_LABELS = {"x": "y_true", "score": "y_score"}
RENAMED = {
    "acf_plot": {"lag_max": "lags", "type": "kind"},
    "arrange_plots": {"...": "plots"},
    "binned_residual_plot": {"bins": "n_bins"},
    "calibration_plot": _LABELS,
    "coefficient_plot": {"x": "model"},
    "compare_models": {"...": "models"},
    "confusion_matrix_plot": {"x": "y_true", "predicted": "y_pred"},
    "decompose_plot": {"frequency": "period"},
    "dendrogram_plot": {"x": "data"},
    "depictr_palette": {"type": "kind"},
    "effects_plot": {"predictor": "var"},
    "explore_distribution": {"type": "kind"},
    "gain_plot": _LABELS,
    "group_comparison_plot": {"y": "x"},
    "interaction_plot": {"predictor": "x", "moderator": "group"},
    "lift_plot": _LABELS,
    "outlier_plot": {"y": "x"},
    "pca_plot": {"x": "data"},
    "power_curve_plot": {"x": "data"},
    "pr_curve_plot": _LABELS,
    "qq_plot": {"x": "model"},
    "raincloud_plot": {"y": "x"},
    "random_effects_plot": {"x": "model"},
    "roc_curve_plot": _LABELS,
    "save_plot": {"...": "kwargs"},
    "scale_color_depictr": {"...": "kwargs"},
    "scale_colour_depictr": {"...": "kwargs"},
    "scale_fill_depictr": {"...": "kwargs"},
    "scree_plot": {"x": "data"},
    "seasonal_plot": {"frequency": "period"},
    "survival_plot": {"status": "event"},
    "threshold_plot": _LABELS,
    "tidy_estimates": {"x": "model"},
}

# R's arguments with no Python counterpart yet, by function. Most are
# cosmetic (a colour, a palette, an axis title), and a plotnine object takes
# those with +. The README's Status section names the ones that change what a
# figure shows or how it is computed.
R_ONLY = {
    "acf_plot": ("conf_level", "x_lab", "y_lab"),
    "arrange_plots": ("guides", "subtitle"),
    "binned_residual_plot": ("type", "point_colour", "band_colour"),
    "calibration_plot": ("colour",),
    "cluster_plot": (
        "scale", "suggest_k", "k_range", "hulls", "label_centers",
        "point_alpha", "palette", "seed", "nstart", "iter.max",
    ),
    "coefficient_plot": (
        "point_colour", "reference_colour", "point_size", "line_size", "facet",
        "scales", "standardise", "sd_multiple", "subtitle", "x_lab",
    ),
    "compare_models": (
        "names", "order", "dodge_width", "palette", "point_size", "line_size",
        "facet", "scales", "legend_title", "legend_ncol", "subtitle", "x_lab",
    ),
    "confusion_matrix_plot": ("threshold",),
    "correlation_heatmap": ("use", "palette", "reorder"),
    "decompose_plot": ("method", "robust", "s_window", "confidence", "level"),
    "dendrogram_plot": (
        "distance", "scale", "k", "horizontal", "labels", "palette",
    ),
    "dumbbell_plot": ("sort", "point_size", "palette", "x_lab", "y_lab"),
    "ecdf_plot": ("reference_quantiles", "palette", "x_lab", "y_lab"),
    "effects_plot": ("rug", "colour", "x_lab", "y_lab"),
    "estimation_plot": (
        "show_points", "point_alpha", "palette", "y_lab", "heights",
    ),
    "explore_bivariate": ("method", "palette", "x_lab", "y_lab"),
    "explore_categorical": (
        "position", "sort", "horizontal", "palette", "x_lab",
    ),
    "explore_distribution": ("position", "palette", "facet", "x_lab"),
    "explore_pairs": ("group", "max_cols", "point_alpha", "method", "palette"),
    "frequentist_bayesian_plot": (
        "conf_level", "labels", "interaction", "intercept", "facet", "scales",
        "note_frequentist_no_prior", "vertical_line_at_x", "subtitle", "x_lab",
        "caption", "...",
    ),
    "gain_plot": ("colour", "legend_inside"),
    "group_comparison_plot": (
        "show_points", "point_alpha", "differences", "reference", "n_boot",
        "palette", "x_lab", "y_lab",
    ),
    "influence_plot": ("n_label", "colour"),
    "interaction_plot": ("moderator_values", "palette", "x_lab", "y_lab"),
    "lift_plot": ("colour", "legend_inside"),
    "missingness_map": ("cols", "show_pct", "colours"),
    "model_report": ("predictor", "standardise", "subtitle"),
    "outlier_plot": (
        "group", "type", "flag", "outlier_colour", "palette", "y_lab",
    ),
    "pca_plot": ("components", "scale", "loadings", "point_alpha", "palette"),
    "posterior_plot": (
        "style", "point", "widths", "interaction", "reference_line", "rope",
        "pd", "facet", "scales", "colour", "fill", "rope_fill", "rope_alpha",
        "x_lab", "caption",
    ),
    "power_curve_plot": ("x_lab", "x_breaks", "x_expand", "interaction"),
    "pr_curve_plot": ("colour", "f1"),
    "qq_plot": (
        "colour", "x_lab", "y_lab", "band", "band_type", "level", "n_sim",
        "band_fill", "seed",
    ),
    "raincloud_plot": ("width", "point_alpha", "palette", "x_lab", "y_lab"),
    "random_effects_plot": ("conf_level", "sort", "point_colour", "x_lab"),
    "residual_diagnostics_plot": ("which", "ncol", "point_alpha", "smooth"),
    "ridgeline_plot": (
        "overlap", "alpha", "scale_height", "palette", "x_lab", "y_lab",
    ),
    "roc_curve_plot": (
        "colour", "ci", "conf_level", "youden", "legend_inside",
    ),
    "scale_color_depictr": ("palette", "na.value"),
    "scale_colour_depictr": ("palette", "na.value"),
    "scale_fill_depictr": ("palette", "na.value"),
    "scatter_trend": ("se", "point_alpha", "palette", "x_lab", "y_lab"),
    "scree_plot": ("scale", "n"),
    "seasonal_plot": (
        "style", "season_labels", "means", "point", "palette", "x_lab", "y_lab",
    ),
    "silhouette_plot": ("clusters", "scale", "distance", "palette"),
    "survival_plot": ("median_line", "risk_breaks", "palette"),
    "threshold_plot": ("metrics", "mark"),
    # R's tidy_estimates() is an S3 generic, whose methods take the rest.
    "tidy_estimates": ("...",),
    "timeseries_plot": (
        "time", "value", "group", "forecast", "frequency", "level", "lambda",
        "palette", "point", "x_lab", "y_lab",
    ),
    "vif_plot": ("palette",),
}

# Python's parameters with no R counterpart, by function.
PYTHON_ONLY = {
    "arrange_plots": ("tag_size",),
    # n_bins is R's bins under the name scikit-learn uses, and the alias
    # bins carries R's name.
    "calibration_plot": ("n_bins", "strategy"),
    # Additive or multiplicative. R decomposes by STL or classically, and
    # its classical decomposition is additive.
    "decompose_plot": ("model",),
    "estimation_plot": ("two_panel", "seed"),
    # R's silhouette_plot() takes the cluster assignments, and this one runs
    # k-means for k clusters.
    "silhouette_plot": ("k",),
    "simulate_figure": ("dpi",),
}

# Functions only one twin exports.
R_ONLY_FUNCTIONS = {
    "depictr_options", "k_diagnostic", "model_fit_table",
    "optimizer_fixef_plot", "palette_preview", "ts_forecast",
}
# The R twin ships the datasets as data, not functions, and keeps the brand
# and accent colours and its inside-legend theme internal.
PYTHON_ONLY_FUNCTIONS = {
    "clinical_trial", "crop_yield", "lexical_decision", "monthly_sales",
    "wellbeing_survey", "depictr_brand", "depictr_accent", "legend_inside",
}

PY_FUNCTIONS = {name: getattr(dp, name) for name in dp.__all__
                if inspect.isfunction(getattr(dp, name))}
SHARED = sorted(set(R_FUNCTIONS) & set(PY_FUNCTIONS))


def _python_parameters(name):
    return list(inspect.signature(PY_FUNCTIONS[name]).parameters)


def test_the_manifest_was_written_from_the_r_twin():
    assert re.fullmatch(r"depictr \d+\.\d+\.\d+(\.\d+)? \(R\)",
                        R_FORMALS["source"])
    assert len(R_FUNCTIONS) > 50


def test_every_function_is_shared_or_listed_as_one_twins():
    assert set(R_FUNCTIONS) - set(PY_FUNCTIONS) == R_ONLY_FUNCTIONS
    assert set(PY_FUNCTIONS) - set(R_FUNCTIONS) == PYTHON_ONLY_FUNCTIONS


@pytest.mark.parametrize("name", SHARED)
def test_every_r_argument_has_a_python_counterpart_or_is_listed(name):
    renamed = RENAMED.get(name, {})
    python = _python_parameters(name)
    missing = tuple(argument for argument in R_FUNCTIONS[name]
                    if renamed.get(argument, argument) not in python)
    assert missing == R_ONLY.get(name, ()), (
        f"{name}(): R arguments without a Python counterpart are {missing}. "
        "Update R_ONLY, RENAMED and the README's Status section.")


@pytest.mark.parametrize("name", SHARED)
def test_every_python_parameter_has_an_r_counterpart_or_is_listed(name):
    renamed = RENAMED.get(name, {})
    matched = {renamed.get(argument, argument)
               for argument in R_FUNCTIONS[name]}
    extra = tuple(p for p in _python_parameters(name) if p not in matched)
    assert extra == PYTHON_ONLY.get(name, ()), (
        f"{name}(): Python parameters without an R counterpart are {extra}. "
        "Update PYTHON_ONLY and RENAMED.")


def test_every_rename_names_arguments_both_twins_have():
    for name, renames in RENAMED.items():
        assert name in SHARED, name
        python = _python_parameters(name)
        for r_name, py_name in renames.items():
            assert r_name in R_FUNCTIONS[name], (name, r_name)
            assert py_name in python, (name, py_name)


def test_the_lists_name_only_shared_functions():
    assert set(R_ONLY) <= set(SHARED)
    assert set(PYTHON_ONLY) <= set(SHARED)


def test_the_readme_status_names_every_function_only_r_has():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = readme.split("\n## Status\n", 1)[1].split("\n## ", 1)[0]
    for name in sorted(R_ONLY_FUNCTIONS):
        assert f"`{name}`" in status, name
