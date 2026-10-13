"""Regression-diagnostic plots for fitted statsmodels results.

These cover the everyday checks on a fitted OLS or GLM: the normality of the
residuals, points that pull the fit around, collinearity among the predictors,
and the fit of a logistic model averaged over its predicted probabilities. The
quantities come from statsmodels (its influence machinery and variance-inflation
factor); each figure is re-drawn under the depictr theme so a diagnostic sits in
the same visual language as the rest of a report.

``statsmodels`` arrives with the core install (plotnine itself requires it);
the ``depictr[models]`` extra remains to pin the tested floor.
"""

from __future__ import annotations

import math
import numbers

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
            "`pip install depictr[models]`."
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
    return _raw_residuals(model)[0]


def _is_glm(model) -> bool:
    """True for a fitted GLM of any family."""
    from statsmodels.genmod.generalized_linear_model import GLM

    return isinstance(getattr(model, "model", None), GLM)


def _is_binary(model) -> bool:
    """True for a fitted Logit or Probit model, statsmodels' binary models."""
    from statsmodels.discrete.discrete_model import BinaryModel

    return isinstance(getattr(model, "model", None), BinaryModel)


def _is_binomial(model) -> bool:
    """True for a binomial GLM, or a Logit or Probit fit."""
    sm = _require_statsmodels()
    if _is_binary(model):
        return True
    return _is_glm(model) and isinstance(model.family, sm.families.Binomial)


def _fitted_values(model) -> np.ndarray:
    """Fitted values on the scale of the response.

    A GLM's ``fittedvalues`` are its fitted means, but a Logit or Probit
    result's are its linear predictor. Its fitted probabilities come from
    ``predict()``, so the two fits of one binomial model plot alike.
    """
    if _is_binary(model):
        return np.asarray(model.predict(), dtype=float)
    return np.asarray(model.fittedvalues, dtype=float)


def _binomial_scale(model, fitted) -> float:
    """The scale of a binomial GLM, counting the trials of every row.

    The scale is 1 unless the fit estimated one. statsmodels' Pearson
    estimate, ``fit(scale="X2")``, leaves out the trials of a two-column
    (successes, failures) response, although its ``pearson_chi2`` counts them,
    so the estimate falls short by about the mean number of trials. Where the
    fit's scale is that estimate, it is recomputed as ``pearson_chi2`` over
    ``df_resid``, the dispersion R's quasibinomial family estimates.
    """
    scale = float(model.scale)
    n_trials = np.asarray(getattr(model.model, "n_trials", 1.0), dtype=float)
    if scale == 1.0 or np.all(n_trials == 1):
        return scale
    # statsmodels' own Pearson estimate, which sums over rows without their
    # trials. A scale that matches it came from fit(scale="X2"), and one that
    # does not (a deviance estimate or a fixed value) is kept as it is.
    endog = np.asarray(model.model.endog, dtype=float)
    without_trials = np.sum(model.model.iweights * (endog - fitted) ** 2
                            / model.family.variance(fitted)) / model.df_resid
    if np.isclose(scale, without_trials, rtol=1e-8, atol=0.0):
        return float(model.pearson_chi2 / model.df_resid)
    return scale


def _raw_residuals(model):
    """The residuals of a fit, with the axis label that names them.

    statsmodels gives a GLM no ``resid``, so its deviance residuals are used,
    the type R's ``residuals.glm()`` returns by default. A Logit or Probit
    result calls them ``resid_dev``.
    """
    if _is_glm(model):
        return np.asarray(model.resid_deviance, dtype=float), "Deviance residuals"
    if _is_binary(model):
        return np.asarray(model.resid_dev, dtype=float), "Deviance residuals"
    return np.asarray(model.resid, dtype=float), "Residuals"


def _quantile_residuals(model, seed=None):
    """Randomised quantile residuals of a GLM, Logit or Probit fit.

    The residual of observation i is the standard-normal quantile of u_i
    (Dunn & Smyth, 1996). For a continuous response, u_i is the fitted
    distribution function at the observation. For a discrete response, it is
    drawn uniformly between the fitted distribution function just below the
    observation and at it. Under a correctly specified model, the residuals are
    independent and standard normal whatever the family, so a normal Q-Q plot
    of them is readable even for a binary or count response.

    Variance weights w_i enter each fitted distribution as they enter the fit,
    through the variance scale * V(mu_i) / w_i. The Gaussian standard
    deviation is therefore sqrt(scale / w_i) and the Gamma shape w_i / scale,
    and binomial weights count trials, as do the totals of a two-column
    (successes, failures) response. Poisson residuals do not use the weights.
    An observation with zero weight takes no part in the fit and gets NaN.

    The supported families are those of the R twin's ``quantile_residuals()``:
    binomial (including Logit and Probit fits), Poisson, Gaussian and Gamma.
    For any other model the deviance residuals come back, flagged, so that a
    caller can label them as such.

    Parameters
    ----------
    model : statsmodels results object
        A fitted GLM, Logit or Probit model.
    seed : int, optional
        Seed for the randomisation of binomial and Poisson residuals. The draws
        come from a generator of their own, so numpy's global random state is
        left alone. The R twin draws from R's generator, so the same seed gives
        other randomised residuals there.

    Returns
    -------
    tuple of (numpy.ndarray, bool)
        One residual per observation of the fit, and whether they are quantile
        residuals (``False`` where the deviance residuals stand in).

    References
    ----------
    Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals.
    Journal of Computational and Graphical Statistics, 5(3), 236-244.
    https://doi.org/10.1080/10618600.1996.10474708
    """
    from scipy import stats

    sm = _require_statsmodels()
    families = sm.families
    supported = (families.Binomial, families.Poisson, families.Gaussian,
                 families.Gamma)
    if _is_binary(model):
        family = families.Binomial()
        mu = _fitted_values(model)
        weights = np.ones_like(mu)
        trials = weights
    elif _is_glm(model) and isinstance(model.family, supported):
        family = model.family
        mu = _fitted_values(model)
        weights = np.asarray(model.model.var_weights, dtype=float)
        trials = np.asarray(getattr(model.model, "n_trials", 1.0), dtype=float)
    else:
        return _raw_residuals(model)[0], False
    y = np.asarray(model.model.endog, dtype=float)
    rng = np.random.default_rng(seed)

    with np.errstate(divide="ignore", invalid="ignore"):
        if isinstance(family, families.Binomial):
            # A proportion over its trials, or a 0/1 outcome over one trial.
            size = trials * weights
            counts = np.round(y * size)
            below = stats.binom.cdf(counts - 1, size, mu)
            at = stats.binom.cdf(counts, size, mu)
            u = rng.uniform(np.minimum(below, at), np.maximum(below, at))
        elif isinstance(family, families.Poisson):
            u = rng.uniform(stats.poisson.cdf(y - 1, mu), stats.poisson.cdf(y, mu))
        elif isinstance(family, families.Gaussian):
            u = stats.norm.cdf(y, loc=mu, scale=np.sqrt(model.scale / weights))
        else:
            # Gamma: shape w_i / scale and mean mu_i, so variance
            # scale * mu_i^2 / w_i.
            shape = weights / model.scale
            u = stats.gamma.cdf(y, a=shape, scale=mu / shape)

    eps = np.finfo(float).eps
    resid = stats.norm.ppf(np.clip(u, eps, 1 - eps))
    resid[weights == 0] = np.nan
    return resid, True


def qq_plot(model, title=None):
    """Normal quantile-quantile plot of the residuals, with a reference line.

    The sample residual quantiles are plotted against the matching standard-normal
    quantiles. Points near the line mean the residuals are close to normal; a
    systematic bend away from it flags skew or heavy tails. The line is drawn
    through the first and third quartiles, the robust choice R's ``qqline`` uses,
    so a few outliers do not tilt it.

    The residuals of a linear model should follow the line. The studentised
    residuals of a non-Gaussian GLM are not expected to, even when the model
    is correctly specified: those of a binary model take only two values at
    each fitted value. A bend then says little about the fit. For a GLM,
    :func:`depictr.residual_diagnostics_plot` and :func:`depictr.model_report`
    plot randomised quantile residuals (Dunn & Smyth, 1996), which are standard
    normal under a correct model.

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

    References
    ----------
    Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals.
    Journal of Computational and Graphical Statistics, 5(3), 236-244.
    https://doi.org/10.1080/10618600.1996.10474708

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> p = dp.qq_plot(model)
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

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> p = dp.influence_plot(model)
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

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> p = dp.vif_plot(model)
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
        + labs(x="", y="Variance inflation factor", title=title)
        + theme_depictr(grid="x")
    )


def binned_residual_plot(model, n_bins=None, title=None, band="model"):
    """Binned residual plot for a binomial model (Gelman & Hill, 2007).

    Raw residuals from a logistic model are uninformative point by point, so the
    fitted probabilities are split into equal-count bins and the mean residual is
    plotted against the mean fitted value in each bin. The bins are the R
    twin's. Rows are ranked by their fitted probability with ties in row order,
    and the ranks are cut at multiples of n / n_bins. A row with zero weight
    takes no part in the fit and is left out.

    The grey band is plus or minus two standard errors of a bin's mean
    residual. By default (``band="model"``), they are the standard errors the
    fitted model implies. The model gives a row with fitted probability p_i
    over m_i trials, taken as a proportion, the variance
    scale * p_i * (1 - p_i) / m_i. Taking this as the variance of the row's
    residual, the band gives the mean of the n_b residuals in a bin the
    standard error sqrt(sum of their variances) / n_b. The fit absorbs a little
    of each row's variance, so a residual varies slightly less than this and
    the band errs on the wide side. Under a well-fitting model, about 95% of
    the points fall inside the band. A run of points outside it, or a clear
    trend in them, points to a missing predictor or the wrong functional form.

    The scale is 1 unless the GLM was fitted with an estimated one, the
    counterpart of the dispersion of R's quasibinomial family. statsmodels'
    Pearson estimate, ``fit(scale="X2")``, leaves out the trials of a
    two-column (successes, failures) response, so for such a fit the scale is
    recomputed as ``pearson_chi2 / df_resid``, which counts them, as R does.

    ``band="empirical"`` takes the standard error from the residuals in the bin,
    as their standard deviation over sqrt(n_b), the band of R's
    ``arm::binnedplot()``. It collapses in a bin whose outcomes are all alike,
    such as a bin without events, and so flags such bins even when the model is
    right.

    Parameters
    ----------
    model : statsmodels results object
        A fitted binomial GLM (logistic regression), or a Logit or Probit
        result, whose fitted probabilities are used.
    n_bins : int, optional
        Number of bins, at least 2. Defaults to the square root of the sample
        size, rounded down and kept between 2 and half the sample size, the
        rule of thumb in Gelman & Hill.
    title : str, optional
        Plot title.
    band : {"model", "empirical"}, default "model"
        Where the standard errors come from: the variance the fitted model
        implies for each residual, or the spread of the residuals in each bin.

    Returns
    -------
    plotnine.ggplot
        Its ``data`` hold one row per bin, with the columns ``fitted``,
        ``resid``, ``n``, ``se2`` (the band's half-width), ``outside`` (whether
        the point lies beyond the band), ``lower`` and ``upper``.

    References
    ----------
    Gelman, A., & Hill, J. (2007). Data analysis using regression and
    multilevel/hierarchical models. Cambridge University Press.

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.api as sm
    >>> import statsmodels.formula.api as smf
    >>> ct = dp.clinical_trial()
    >>> glm = smf.glm("adverse_event ~ biomarker + age", ct,
    ...               family=sm.families.Binomial()).fit()
    >>> p = dp.binned_residual_plot(glm)
    """
    _require_statsmodels()
    from scipy import stats

    if not _is_fitted_model(model):
        raise TypeError("binned_residual_plot needs a fitted statsmodels GLM.")
    # Only a binomial model is meaningful here: a binomial GLM, or a Logit or
    # Probit fit of the same kind of outcome.
    if not _is_binomial(model):
        raise ValueError(
            "binned_residual_plot expects a binomial GLM "
            "(family=sm.families.Binomial()) or a Logit or Probit fit."
        )
    if band not in ("model", "empirical"):
        raise ValueError("`band` must be 'model' or 'empirical'.")

    fitted = _fitted_values(model)
    # Residuals of the response as a proportion, as in the R twin. statsmodels
    # holds a two-column (successes, failures) response as proportions, but
    # its resid_response multiplies them back up to counts of successes.
    resid = np.asarray(model.model.endog, dtype=float) - fitted
    if _is_glm(model):
        # Trials per row, from a two-column response or from var_weights on a
        # proportion, as in _quantile_residuals().
        trials = (np.asarray(getattr(model.model, "n_trials", 1.0), dtype=float)
                  * np.asarray(model.model.var_weights, dtype=float))
        scale = _binomial_scale(model, fitted)
    else:
        trials = np.ones_like(fitted)
        scale = 1.0
    trials = np.broadcast_to(trials, fitted.shape)
    # A row with zero weight takes no part in the fit, and the model gives its
    # residual no finite variance.
    keep = np.isfinite(fitted) & np.isfinite(resid) & (trials > 0)
    fitted, resid, trials = fitted[keep], resid[keep], trials[keep]
    n = len(fitted)
    if n < 4:
        raise ValueError("Too few observations for a binned-residual plot.")
    if n_bins is None:
        # floor(sqrt(n)), kept between 2 and n // 2, as in the R twin.
        n_bins = max(2, min(math.isqrt(n), n // 2))
    elif (isinstance(n_bins, bool) or not isinstance(n_bins, numbers.Real)
          or not n_bins >= 2):
        raise ValueError("`n_bins` must be a single number >= 2.")
    else:
        n_bins = int(n_bins)

    # Equal-count bins as in the R twin: rank the fitted values, breaking ties
    # by row order as R's rank(ties.method = "first") does, and cut the ranks
    # at multiples of n / n_bins.
    rank = stats.rankdata(fitted, method="ordinal")
    bin_id = np.minimum(np.ceil(rank / (n / n_bins)), n_bins)
    # The variance of each residual under the model.
    resid_var = scale * fitted * (1 - fitted) / trials

    rows = []
    # The bins are numbered by rank, so they come in order of mean fitted value.
    for b in np.unique(bin_id):
        mask = bin_id == b
        nb = int(mask.sum())
        r = resid[mask]
        if band == "model":
            # The mean of independent residuals: the root of their summed
            # variances over their count.
            se = np.sqrt(resid_var[mask].sum()) / nb
        elif nb > 1:
            se = r.std(ddof=1) / np.sqrt(nb)
        else:
            se = np.nan
        rows.append({"fitted": fitted[mask].mean(), "resid": r.mean(), "n": nb,
                     "se2": 2 * se})
    df = pd.DataFrame(rows)
    df["outside"] = np.isfinite(df["se2"]) & (df["resid"].abs() > df["se2"])
    df["lower"] = -df["se2"]
    df["upper"] = df["se2"]

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
