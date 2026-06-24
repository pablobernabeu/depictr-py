"""depictr: a unified, colourblind-safe toolkit for publication-ready
statistical visualisation.

depictr gives one consistent, accessible, plotnine-based look to the whole
statistical workflow -- exploratory analysis, model estimates, diagnostics,
classification, multivariate analysis, survival and time series -- so a coherent
figure set no longer means stitching together half a dozen libraries with
mismatched defaults. Every function returns a plotnine object you can extend
with ``+``, and the colourblind-safe Okabe-Ito palette is the default, validated
by a built-in colour-vision-deficiency safety check.

It is the Python sibling of the depictr R package
(https://github.com/pablobernabeu/depictr).
"""

from __future__ import annotations

from .bivariate import explore_bivariate, scatter_trend
from .classification import (
    calibration_plot,
    confusion_matrix_plot,
    gain_plot,
    lift_plot,
    pr_curve_plot,
    roc_curve_plot,
    threshold_plot,
)
from .compose import save_plot
from .cvd import palette_safety, simulate_cvd
from .data import DATASETS, clinical_trial, crop_yield, lexical_decision, wellbeing_survey
from .diagnostics import binned_residual_plot, influence_plot, qq_plot, vif_plot
from .distributions_extra import (
    dumbbell_plot,
    ecdf_plot,
    group_comparison_plot,
    outlier_plot,
    ridgeline_plot,
)
from .eda import (
    correlation_heatmap,
    explore_categorical,
    explore_distribution,
    missingness_map,
)
from .estimation import estimation_plot
from .models import coefficient_plot, tidy_estimates
from .multivariate import (
    cluster_plot,
    dendrogram_plot,
    pca_plot,
    scree_plot,
    silhouette_plot,
)
from .palette import depictr_accent, depictr_brand, depictr_palette
from .survival import survival_plot
from .tables import summary_table
from .theme import (
    scale_color_depictr,
    scale_colour_depictr,
    scale_fill_depictr,
    theme_depictr,
)
from .timeseries import acf_plot, decompose_plot, seasonal_plot, timeseries_plot

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Theme and palette
    "theme_depictr", "scale_colour_depictr", "scale_color_depictr",
    "scale_fill_depictr", "depictr_palette", "depictr_brand", "depictr_accent",
    # Accessibility
    "palette_safety", "simulate_cvd",
    # Exploratory analysis
    "explore_distribution", "explore_categorical", "explore_bivariate",
    "scatter_trend", "correlation_heatmap", "missingness_map", "ecdf_plot",
    "ridgeline_plot", "dumbbell_plot", "outlier_plot", "group_comparison_plot",
    # Estimation and tables
    "estimation_plot", "summary_table",
    # Model estimates
    "coefficient_plot", "tidy_estimates",
    # Diagnostics
    "qq_plot", "influence_plot", "vif_plot", "binned_residual_plot",
    # Classification
    "roc_curve_plot", "pr_curve_plot", "confusion_matrix_plot",
    "calibration_plot", "gain_plot", "lift_plot", "threshold_plot",
    # Multivariate
    "pca_plot", "scree_plot", "cluster_plot", "dendrogram_plot",
    "silhouette_plot",
    # Survival
    "survival_plot",
    # Time series
    "acf_plot", "decompose_plot", "seasonal_plot", "timeseries_plot",
    # Composition
    "save_plot",
    # Data
    "crop_yield", "wellbeing_survey", "lexical_decision", "clinical_trial",
    "DATASETS",
]
