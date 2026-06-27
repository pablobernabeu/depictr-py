"""Regression-diagnostic plots for fitted statsmodels results.

These cover the everyday checks on a fitted OLS or GLM: the normality of the
residuals, points that pull the fit around, collinearity among the predictors,
and the fit of a logistic model averaged over its predicted probabilities. The
quantities come from statsmodels (its influence machinery and variance-inflation
factor); each figure is re-drawn under the depictr theme so a diagnostic sits in
the same visual language as the rest of a report.

Install the optional dependency with ``pip install depictr[diagnostics]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    geom_abline,
    geom_col,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    labs,
    scale_size_area,
)

from .palette import ACCENT, BRAND
from .theme import theme_depictr


def _require_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The diagnostic plots need statsmodels. Install it with "
            "`pip install depictr[diagnostics]`."
        ) from exc
    return sm


def _is_fitted_model(obj) -> bool:
    """True for a fitted statsmodels result (OLS/GLM), as opposed to raw data."""
    return hasattr(obj, "get_influence") and hasattr(obj, "model")


def _studentised_residuals(model) -> np.ndarray:
    """Studentised residuals, whichever name the influence object uses.

    OLS exposes ``resid_studentized_internal`` and GLM ``resid_studentized``;
    both are the residual divided by its estimated standard deviation, so a value
    is roughly on a standard-normal scale.
    """
    infl = model.get_influence()
    for name in ("resid_studentized_internal", "resid_studentized"):
        if hasattr(infl, name):
            return np.asarray(getattr(infl, name))
    # Last resort: fall back to the raw residuals.
    return np.asarray(model.resid)


def qq_plot(model, title=None):
    """Normal quantile-quantile plot of the residuals, with a reference line.

    The sample residual quantiles are plotted against the matching standard-normal
    quantiles. Points near the line mean the residuals are close to normal; a
    systematic bend away from it flags skew or heavy tails. The line is drawn
    through the first and third quartiles, the robust choice R's ``qqline`` uses,
    so a few outliers do not tilt it.

    Parameters
    ----------
    model : statsmodels results object or array-like
        A fitted OLS/GLM result, from which the internally studentised residuals
        are read (the raw residuals if studentised ones are not available), or a
        1-D array of residuals to plot directly.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    from scipy import stats

    if _is_fitted_model(model):
        resid = _studentised_residuals(model)
    else:
        resid = np.asarray(model, dtype=float).ravel()

    # probplot returns the theoretical (osm) and ordered sample (osr) quantiles.
    theoretical, ordered = stats.probplot(resid, dist="norm", fit=False)
    df = pd.DataFrame({"theoretical": theoretical, "sample": ordered})

    # Reference line through the quartiles, matching R's qqline.
    q_theory = stats.norm.ppf([0.25, 0.75])
    q_sample = np.percentile(resid, [25, 75])
    slope = (q_sample[1] - q_sample[0]) / (q_theory[1] - q_theory[0])
    intercept = q_sample[0] - slope * q_theory[0]

    return (
        ggplot(df, aes("theoretical", "sample"))
        + geom_abline(intercept=intercept, slope=slope,
                      linetype="dashed", color="#9e9e9e")
        + geom_point(color=BRAND, size=2, alpha=0.7)
        + labs(x="Theoretical quantiles", y="Sample quantiles", title=title)
        + theme_depictr()
    )


def influence_plot(model, title=None):
    """Bubble plot of studentised residuals against leverage.

    Each point is one observation; the bubble area is proportional to its Cook's
    distance, the standard measure of how much the fit would move if the point
    were dropped (Cook, 1977). Points to the right have high leverage (an unusual
    predictor combination), points far above or below zero are poorly fitted, and
    large bubbles are the ones that combine both into real influence.

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS/GLM result.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    References
    ----------
    Cook, R. D. (1977). Detection of influential observation in linear
    regression. Technometrics, 19(1), 15-18.
    https://doi.org/10.1080/00401706.1977.10489493
    """
    _require_statsmodels()
    if not _is_fitted_model(model):
        raise TypeError("influence_plot needs a fitted statsmodels result.")

    infl = model.get_influence()
    df = pd.DataFrame({
        "leverage": np.asarray(infl.hat_matrix_diag),
        "studentised": _studentised_residuals(model),
        "cooks": np.asarray(infl.cooks_distance[0]),
    })
    return (
        ggplot(df, aes("leverage", "studentised", size="cooks"))
        + geom_hline(yintercept=0, linetype="dashed", color="#9e9e9e")
        + geom_point(color=BRAND, alpha=0.5)
        + scale_size_area(max_size=10, name="Cook's distance")
        + labs(x="Leverage (hat value)", y="Studentised residual", title=title)
        + theme_depictr()
    )


def vif_plot(model, title=None):
    """Horizontal bar chart of the variance inflation factor per predictor.

    The variance inflation factor (VIF) measures how much a coefficient's
    variance is inflated by collinearity with the other predictors. A VIF of 1
    means no collinearity; the reference line sits at 5, a common threshold above
    which collinearity is usually treated as a concern. The intercept is dropped,
    as its VIF is not interpretable in the usual way.

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS/GLM result with a design matrix of two or more predictors.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    _require_statsmodels()
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if not _is_fitted_model(model):
        raise TypeError("vif_plot needs a fitted statsmodels result.")

    exog = np.asarray(model.model.exog)
    names = list(model.model.exog_names)
    rows = []
    for i, name in enumerate(names):
        if name.lower() in {"intercept", "const"}:
            continue
        rows.append({"term": name, "vif": variance_inflation_factor(exog, i)})
    if not rows:
        raise ValueError("Need at least one non-intercept predictor for a VIF.")
    df = pd.DataFrame(rows).sort_values("vif")
    # Lock the order so the largest VIF reads at the top after the flip.
    df["term"] = pd.Categorical(df["term"], categories=list(df["term"]), ordered=True)

    return (
        ggplot(df, aes("term", "vif"))
        + geom_col(fill=BRAND, width=0.7)
        + geom_hline(yintercept=5, linetype="dashed", color=ACCENT)
        + coord_flip()
        + labs(x=None, y="Variance inflation factor", title=title)
        + theme_depictr(grid="x")
    )


def binned_residual_plot(model, n_bins=None, title=None):
    """Binned residual plot for a binomial GLM (Gelman & Hill, 2007).

    Raw residuals from a logistic model are uninformative point by point, so the
    fitted probabilities are split into equal-count bins and the mean residual is
    plotted against the mean fitted value in each bin. The grey band is plus or
    minus two standard errors of the bin mean; under a well-fitting model about
    95% of the points fall inside it. A run of points outside the band, or a clear
    trend in them, points to a missing predictor or the wrong functional form.

    Parameters
    ----------
    model : statsmodels results object
        A fitted binomial GLM (logistic regression).
    n_bins : int, optional
        Number of bins. Defaults to the square root of the sample size, rounded,
        the rule of thumb in Gelman & Hill.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    References
    ----------
    Gelman, A., & Hill, J. (2007). Data analysis using regression and
    multilevel/hierarchical models. Cambridge University Press.
    """
    sm = _require_statsmodels()
    if not _is_fitted_model(model):
        raise TypeError("binned_residual_plot needs a fitted statsmodels GLM.")
    # OLS results have no `family`; only a binomial GLM is meaningful here.
    family = getattr(model, "family", None)
    if not isinstance(family, sm.families.Binomial):
        raise ValueError(
            "binned_residual_plot expects a binomial GLM "
            "(family=sm.families.Binomial())."
        )

    fitted = np.asarray(model.fittedvalues)
    resid = np.asarray(model.resid_response)
    n = len(fitted)
    if n_bins is None:
        n_bins = max(int(round(np.sqrt(n))), 1)
    n_bins = min(n_bins, n)

    # Equal-count bins: rank the fitted values, then split into n_bins groups.
    order = np.argsort(fitted)
    bin_id = np.empty(n, dtype=int)
    bin_id[order] = np.floor(np.arange(n) / n * n_bins).astype(int)

    rows = []
    for b in range(n_bins):
        mask = bin_id == b
        if not mask.any():
            continue
        r = resid[mask]
        mean_fitted = fitted[mask].mean()
        mean_resid = r.mean()
        # Two-SE band: the standard error of a binomial residual scales with the
        # spread of the fitted probabilities in the bin (Gelman & Hill, 2007).
        se = 2 * np.sqrt(np.mean(fitted[mask] * (1 - fitted[mask])) / mask.sum())
        rows.append({"fitted": mean_fitted, "resid": mean_resid,
                     "lower": -se, "upper": se})
    df = pd.DataFrame(rows).sort_values("fitted")

    return (
        ggplot(df, aes("fitted", "resid"))
        + geom_ribbon(aes(ymin="lower", ymax="upper"),
                      fill="#9e9e9e", alpha=0.25)
        + geom_line(aes(y="upper"), color="#9e9e9e", linetype="dashed")
        + geom_line(aes(y="lower"), color="#9e9e9e", linetype="dashed")
        + geom_hline(yintercept=0, color="#9e9e9e")
        + geom_point(color=BRAND, size=2.2)
        + labs(x="Mean predicted probability", y="Mean residual", title=title)
        + theme_depictr()
    )
