"""Multi-panel regression reports for fitted statsmodels results.

A single diagnostic plot answers one question; a regression check usually means
looking at several at once. These functions gather the everyday diagnostics into
one composed figure so the whole picture sits on a single page in the depictr
visual language. The four-panel residual report mirrors the set R draws from
``plot.lm`` (Fox & Weisberg, 2019), and the dashboard pairs the coefficient plot
with the two panels people read first.

The panels reuse the single-plot diagnostics in :mod:`depictr.diagnostics` and
the forest plot in :mod:`depictr.models`, and join them with
:func:`depictr.compose.arrange_plots`.

Install the optional dependency with ``pip install depictr[models]``.

References
----------
Fox, J., & Weisberg, S. (2019). An R companion to applied regression (3rd ed.).
Sage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_hline,
    geom_point,
    geom_smooth,
    ggplot,
    labs,
)

from .compose import arrange_plots
from .diagnostics import (
    _is_fitted_model,
    _require_statsmodels,
    _studentised_residuals,
    qq_plot,
)
from .models import coefficient_plot
from .palette import BRAND
from .theme import theme_depictr


def _check_model(model, func_name: str) -> None:
    """Confirm the argument is a fitted statsmodels result, or explain why not."""
    _require_statsmodels()
    if not _is_fitted_model(model):
        raise TypeError(f"{func_name} needs a fitted statsmodels result.")


def _residuals_vs_fitted(model, title=None):
    """Residuals against fitted values, with a lowess smoother and a zero line.

    A flat smoother near zero means the linear fit has caught the structure; a
    bend or funnel points to a missing term or non-constant variance.
    """
    df = pd.DataFrame({
        "fitted": np.asarray(model.fittedvalues),
        "resid": np.asarray(model.resid),
    })
    return (
        ggplot(df, aes("fitted", "resid"))
        + geom_hline(yintercept=0, linetype="dashed", color="#9e9e9e")
        + geom_point(color=BRAND, size=2, alpha=0.6)
        + geom_smooth(method="lowess", se=False, color="#e69f00", size=0.9)
        + labs(x="Fitted values", y="Residuals", title=title)
        + theme_depictr()
    )


def _scale_location(model, title=None):
    """Scale-location plot: sqrt of the absolute studentised residuals vs fitted.

    Taking the square root of the absolute residual flattens the cloud, so a
    rising or falling smoother reads as a change in spread (heteroscedasticity)
    rather than in level.
    """
    fitted = np.asarray(model.fittedvalues)
    root_abs = np.sqrt(np.abs(_studentised_residuals(model)))
    df = pd.DataFrame({"fitted": fitted, "root_abs": root_abs})
    return (
        ggplot(df, aes("fitted", "root_abs"))
        + geom_point(color=BRAND, size=2, alpha=0.6)
        + geom_smooth(method="lowess", se=False, color="#e69f00", size=0.9)
        + labs(x="Fitted values",
               y="√|Standardised residuals|", title=title)
        + theme_depictr()
    )


def _residuals_vs_leverage(model, title=None):
    """Studentised residuals against leverage, with a zero line.

    Leverage measures how unusual an observation's predictors are. A point that
    sits far right and far from zero is the kind that can pull the fit on its
    own, so it is worth a second look.
    """
    infl = model.get_influence()
    df = pd.DataFrame({
        "leverage": np.asarray(infl.hat_matrix_diag),
        "studentised": _studentised_residuals(model),
    })
    return (
        ggplot(df, aes("leverage", "studentised"))
        + geom_hline(yintercept=0, linetype="dashed", color="#9e9e9e")
        + geom_point(color=BRAND, size=2, alpha=0.6)
        + geom_smooth(method="lowess", se=False, color="#e69f00", size=0.9)
        + labs(x="Leverage (hat value)", y="Studentised residuals", title=title)
        + theme_depictr()
    )


def residual_diagnostics_plot(model, title=None):
    """Four-panel residual report for a fitted OLS or GLM.

    Composes the four checks R draws from ``plot.lm`` into a 2x2 grid: residuals
    against fitted values, the scale-location plot, a normal quantile-quantile
    plot of the residuals, and residuals against leverage. Read together they
    cover the usual assumptions of a linear fit: the right functional form,
    constant variance, approximately normal residuals, and no single point
    dominating the fit.

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS/GLM result, the kind returned by ``smf.ols(...).fit()``.
    title : str, optional
        Accepted for API symmetry, but dropped with a warning: plotnine
        compositions cannot carry a figure-level title, so the grid keeps its
        per-panel titles instead.

    Returns
    -------
    plotnine.composition.Compose
        A 2x2 composition with ``.draw`` and ``.save``.

    References
    ----------
    Fox, J., & Weisberg, S. (2019). An R companion to applied regression
    (3rd ed.). Sage.

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> p = dp.residual_diagnostics_plot(model)
    """
    _check_model(model, "residual_diagnostics_plot")
    return arrange_plots(
        _residuals_vs_fitted(model, title="Residuals vs fitted"),
        _scale_location(model, title="Scale-location"),
        qq_plot(model, title="Normal Q-Q"),
        _residuals_vs_leverage(model, title="Residuals vs leverage"),
        ncol=2,
        title=title,
    )


def model_report(model, title=None):
    """One-figure dashboard pairing the coefficient plot with key diagnostics.

    Puts the three plots a reader usually wants first side by side: the
    coefficient (forest) plot of the estimates and their confidence intervals,
    the residuals-vs-fitted plot, and the normal quantile-quantile plot. It is a
    quick "is the model worth trusting, and what does it say" view, with the full
    set of checks left to :func:`residual_diagnostics_plot`.

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS/GLM result exposing ``params`` and ``conf_int`` (for the
        coefficient plot) as well as fitted values and residuals.
    title : str, optional
        Accepted for API symmetry, but dropped with a warning: plotnine
        compositions cannot carry a figure-level title, so the dashboard keeps
        its per-panel titles instead.

    Returns
    -------
    plotnine.composition.Compose
        A composition with ``.draw`` and ``.save``.

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> p = dp.model_report(model)
    """
    _check_model(model, "model_report")
    return arrange_plots(
        coefficient_plot(model, title="Coefficients"),
        _residuals_vs_fitted(model, title="Residuals vs fitted"),
        qq_plot(model, title="Normal Q-Q"),
        ncol=3,
        title=title,
    )
