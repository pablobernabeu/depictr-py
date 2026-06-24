"""depictr: a unified, colourblind-safe toolkit for publication-ready
statistical visualisation.

depictr gives one consistent, accessible, plotnine-based look to the whole
statistical workflow -- exploratory analysis, model estimates, diagnostics,
classification, survival and more -- so a coherent figure set no longer means
stitching together half a dozen libraries with mismatched defaults. Every
function returns a plotnine object you can extend with ``+``, and the
colourblind-safe Okabe-Ito palette is the default, validated by a built-in
colour-vision-deficiency safety check.

It is the Python sibling of the depictr R package
(https://github.com/pablobernabeu/depictr).
"""

from __future__ import annotations

from .classification import (
    calibration_plot,
    confusion_matrix_plot,
    gain_plot,
    pr_curve_plot,
    roc_curve_plot,
)
from .compose import save_plot
from .cvd import palette_safety, simulate_cvd
from .data import DATASETS, clinical_trial, crop_yield, lexical_decision, wellbeing_survey
from .eda import (
    correlation_heatmap,
    explore_categorical,
    explore_distribution,
    missingness_map,
)
from .models import coefficient_plot, tidy_estimates
from .palette import depictr_accent, depictr_brand, depictr_palette
from .survival import survival_plot
from .theme import scale_color_depictr, scale_colour_depictr, scale_fill_depictr, theme_depictr

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Theme and palette
    "theme_depictr", "scale_colour_depictr", "scale_color_depictr",
    "scale_fill_depictr", "depictr_palette", "depictr_brand", "depictr_accent",
    # Accessibility
    "palette_safety", "simulate_cvd",
    # EDA
    "explore_distribution", "explore_categorical", "correlation_heatmap",
    "missingness_map",
    # Models
    "coefficient_plot", "tidy_estimates",
    # Classification
    "roc_curve_plot", "pr_curve_plot", "confusion_matrix_plot",
    "calibration_plot", "gain_plot",
    # Survival
    "survival_plot",
    # Composition
    "save_plot",
    # Data
    "crop_yield", "wellbeing_survey", "lexical_decision", "clinical_trial",
    "DATASETS",
]
