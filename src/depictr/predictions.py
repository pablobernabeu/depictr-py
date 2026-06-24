"""Model-prediction plots for fitted statsmodels results.

These turn a fitted model into the predictions readers actually want to see: how
the response moves as one predictor varies, how that movement differs across the
levels of a grouping variable (the interaction), and how a coefficient compares
across competing models. The predictions and their confidence bands come from
statsmodels' ``get_prediction``; the figures are re-drawn under the depictr theme
so a prediction plot sits in the same visual language as the rest of a report.

For a single-predictor effect the other numeric predictors are held at their mean
and categoricals at their most common (reference) level, the usual convention for
an "effects" plot, so the line shows the partial relationship rather than a
marginal average over the sample.

Install the optional dependency with ``pip install depictr[models]``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_line,
    geom_ribbon,
    ggplot,
    labs,
    theme,
)

from .palette import BRAND
from .theme import scale_colour_depictr, scale_fill_depictr, theme_depictr


def _require_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The prediction plots need statsmodels. Install it with "
            "`pip install depictr[models]`."
        ) from exc
    return sm


def _is_fitted_model(obj) -> bool:
    """True for a fitted statsmodels result that can build a prediction grid."""
    return hasattr(obj, "get_prediction") and hasattr(obj, "model")


def _model_frame(model) -> pd.DataFrame:
    """The original data the model was fitted on, with its source column names."""
    frame = getattr(model.model.data, "frame", None)
    if frame is None:
        raise ValueError(
            "This model does not retain its source data, so a prediction grid "
            "cannot be built. Fit it from a formula and a DataFrame "
            "(statsmodels' smf.ols / smf.glm)."
        )
    return frame


def _response_label(model) -> str:
    """A readable name for the response, unwrapping a formula Q(\"...\") guard."""
    name = str(model.model.endog_names)
    return re.sub(r'Q\("([^"]+)"\)', r"\1", name)


def _predictor_columns(model, frame: pd.DataFrame) -> list[str]:
    """Source columns that are predictors (the response column dropped)."""
    response = _response_label(model)
    return [c for c in frame.columns if c != response]


def _reference_row(frame: pd.DataFrame, predictors: list[str]) -> dict:
    """Hold each predictor at a single value: numeric at its mean, else the mode.

    The mode is the most common observed level, which for a treatment-coded
    categorical is its reference unless one level happens to be rarer.
    """
    row = {}
    for col in predictors:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s):
            row[col] = s.mean()
        else:
            row[col] = s.mode().iloc[0]
    return row


def _prediction_band(model, grid: pd.DataFrame, conf_level: float) -> pd.DataFrame:
    """Run get_prediction over a grid and return mean and CI on the grid's index.

    Both OLS and GLM expose ``mean``, ``mean_ci_lower`` and ``mean_ci_upper`` from
    ``summary_frame``; for a GLM these are on the response scale (probabilities),
    which is what a reader expects from a prediction plot.
    """
    summary = model.get_prediction(grid).summary_frame(alpha=1 - conf_level)
    return pd.DataFrame({
        "fit": summary["mean"].to_numpy(),
        "lower": summary["mean_ci_lower"].to_numpy(),
        "upper": summary["mean_ci_upper"].to_numpy(),
    })


def effects_plot(model, var, conf_level: float = 0.95, n: int = 100, title=None):
    """Predicted response as one predictor varies, with a confidence band.

    The predictor ``var`` is swept across its observed range while every other
    numeric predictor is held at its mean and every categorical at its reference
    (most common) level. The line is the predicted response and the ribbon its
    confidence band, both read from statsmodels' ``get_prediction``.

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS/GLM result from a formula and a DataFrame (so the source
        data is retained for building the grid).
    var : str
        The predictor to vary along the x-axis. Must be a numeric column.
    conf_level : float
        Confidence level for the band.
    n : int
        Number of points across the predictor's range.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    _require_statsmodels()
    if not _is_fitted_model(model):
        raise TypeError("effects_plot needs a fitted statsmodels result.")

    frame = _model_frame(model)
    if var not in frame.columns:
        raise KeyError(f"{var!r} is not a predictor in the fitted model.")
    if not pd.api.types.is_numeric_dtype(frame[var]):
        raise TypeError(
            f"effects_plot varies {var!r} continuously, so it must be numeric. "
            "Use interaction_plot for a categorical predictor."
        )

    predictors = _predictor_columns(model, frame)
    others = [c for c in predictors if c != var]
    grid = pd.DataFrame({var: np.linspace(frame[var].min(), frame[var].max(), n)})
    for col, value in _reference_row(frame, others).items():
        grid[col] = value

    band = _prediction_band(model, grid, conf_level)
    df = pd.concat([grid[[var]].reset_index(drop=True), band], axis=1)

    return (
        ggplot(df, aes(x=var, y="fit"))
        + geom_ribbon(aes(ymin="lower", ymax="upper"), fill=BRAND, alpha=0.2)
        + geom_line(color=BRAND, size=0.9)
        + labs(x=var, y=f"Predicted {_response_label(model)}", title=title)
        + theme_depictr()
    )


def interaction_plot(model, x, group, conf_level: float = 0.95, n: int = 100,
                     band: bool = True, title=None):
    """Predicted response across ``x`` for each level of a categorical ``group``.

    One coloured line per group shows how the predicted response changes with
    ``x`` within that group, so a fan of non-parallel lines is the visible mark of
    an interaction. The remaining predictors are held at their mean (numeric) or
    reference level (categorical).

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS/GLM result from a formula and a DataFrame.
    x : str
        The numeric predictor on the x-axis.
    group : str
        The categorical predictor whose levels are drawn as separate lines.
    conf_level : float
        Confidence level for the bands.
    n : int
        Number of points across the x range, per group.
    band : bool
        Whether to draw a confidence ribbon behind each line.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    _require_statsmodels()
    if not _is_fitted_model(model):
        raise TypeError("interaction_plot needs a fitted statsmodels result.")

    frame = _model_frame(model)
    for name in (x, group):
        if name not in frame.columns:
            raise KeyError(f"{name!r} is not a predictor in the fitted model.")
    if not pd.api.types.is_numeric_dtype(frame[x]):
        raise TypeError(f"interaction_plot needs a numeric x; {x!r} is not.")

    predictors = _predictor_columns(model, frame)
    others = [c for c in predictors if c not in (x, group)]
    held = _reference_row(frame, others)
    x_seq = np.linspace(frame[x].min(), frame[x].max(), n)
    levels = list(pd.unique(frame[group]))

    parts = []
    for level in levels:
        grid = pd.DataFrame({x: x_seq})
        grid[group] = level
        for col, value in held.items():
            grid[col] = value
        piece = pd.concat(
            [grid[[x]].reset_index(drop=True), _prediction_band(model, grid, conf_level)],
            axis=1,
        )
        piece[group] = str(level)
        parts.append(piece)
    df = pd.concat(parts, ignore_index=True)

    p = ggplot(df, aes(x=x, y="fit", color=group))
    if band:
        p = p + geom_ribbon(aes(ymin="lower", ymax="upper", fill=group),
                            alpha=0.15, color=None)
    return (
        p
        + geom_line(size=0.9)
        + scale_colour_depictr(len(levels))
        + scale_fill_depictr(len(levels))
        + labs(x=x, y=f"Predicted {_response_label(model)}",
               color=group, fill=group, title=title)
        + theme_depictr()
    )


def compare_models(models, intercept: bool = False, conf_level: float = 0.95,
                   title=None):
    """Dodged forest comparing each coefficient across several models.

    Each model contributes its estimates and confidence intervals (read with
    :func:`depictr.models.tidy_estimates`, so a fitted model or a tidy frame both
    work); the terms share a y-axis and the models are dodged apart and coloured,
    so a coefficient that shifts from one specification to the next is easy to
    spot.

    Parameters
    ----------
    models : dict
        ``{name: fitted_model_or_tidy_frame}``. The keys label and colour the
        models.
    intercept : bool
        Whether to keep the intercept term.
    conf_level : float
        Confidence level passed to :func:`depictr.models.tidy_estimates`.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    from .models import tidy_estimates

    if not isinstance(models, dict) or not models:
        raise ValueError("`models` must be a non-empty {name: model} dict.")

    frames = []
    for name, model in models.items():
        est = tidy_estimates(model, conf_level=conf_level).copy()
        est["model"] = str(name)
        frames.append(est)
    est = pd.concat(frames, ignore_index=True)

    if not intercept:
        est = est[~est["term"].str.fullmatch(r"(?i)\(?intercept\)?|const")]
    if est.empty:
        raise ValueError("Nothing left to plot after dropping the intercept.")

    # First-seen order of terms and models. Terms run up the axis so the first
    # reads at the top, models keep their dict order in the legend.
    term_levels = list(dict.fromkeys(est["term"]))
    model_levels = [str(k) for k in models]
    est["model"] = pd.Categorical(est["model"], categories=model_levels, ordered=True)

    # Dodge the models by hand on a numeric y. position_dodge keys off the x
    # interval for a horizontal errorbar, so it cannot separate models that share
    # an x range; an explicit per-model offset within each term's slot does.
    n_models = len(model_levels)
    base_y = {term: len(term_levels) - 1 - i for i, term in enumerate(term_levels)}
    spread = 0.6  # total height a term's models occupy
    if n_models == 1:
        offset = {model_levels[0]: 0.0}
    else:
        offset = {m: (j / (n_models - 1) - 0.5) * spread
                  for j, m in enumerate(model_levels)}
    est["y"] = [base_y[t] + offset[m] for t, m in zip(est["term"], est["model"])]

    from plotnine import (
        element_blank,
        geom_errorbarh,
        geom_point,
        geom_vline,
        scale_y_continuous,
    )

    return (
        ggplot(est, aes(x="estimate", y="y", color="model"))
        + geom_vline(xintercept=0, linetype="dashed", color="#9e9e9e")
        + geom_errorbarh(aes(xmin="conf_low", xmax="conf_high"),
                         height=0.0, size=0.7)
        + geom_point(size=2.4)
        + scale_colour_depictr(n_models, name="Model")
        + scale_y_continuous(
            breaks=[base_y[t] for t in term_levels],
            labels=term_levels,
            limits=(-0.5, len(term_levels) - 0.5),
        )
        + labs(x="Estimate", y=None, color="Model", title=title)
        + theme_depictr()
        + theme(panel_grid_major_y=element_blank())
    )
