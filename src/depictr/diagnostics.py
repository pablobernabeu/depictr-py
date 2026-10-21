"""Regression-diagnostic plots for fitted statsmodels results.

These cover the everyday checks on a fitted OLS or GLM: the normality of the
residuals, points that pull the fit around, collinearity among the predictors,
and the fit of a logistic model averaged over its predicted probabilities. The
quantities come from statsmodels (its influence machinery and the coefficient
covariance of the fit); each figure is re-drawn under the depictr theme so a
diagnostic sits in the same visual language as the rest of a report.

``statsmodels`` arrives with the core install (plotnine itself requires it);
the ``depictr[models]`` extra remains to pin the tested floor.
"""

from __future__ import annotations

import math
import numbers
import warnings

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

from .palette import BRAND
from .theme import theme_depictr


def _require_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The diagnostic plots need statsmodels. Install it with "
            '`pip install "depictr[models]"`.'
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
    [`depictr.residual_diagnostics_plot`][] and [`depictr.model_report`][]
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


def _model_terms(model):
    """The design columns of each term of a fitted model, and its intercept.

    statsmodels up to 0.14 keeps a formula's patsy DesignInfo as
    ``data.design_info``, and from 0.15 as ``data.model_spec``, which holds a
    formulaic ModelSpec when the formula engine is formulaic. Each names the
    terms and the columns they fill, and the intercept is the term
    "Intercept" or "1". The spec is read according to its own type, because
    statsmodels' FormulaManager reads one through whichever engine is set at
    the time, which need not be the engine that built it. A model fitted from
    arrays has no terms, so each column stands for itself, and the intercept
    is the constant column statsmodels found.

    Returns
    -------
    tuple of (dict, str or None)
        The column indices of each term, keyed by its name, in model order,
        and the intercept's key, or None for a model without one.
    """
    names = [str(name) for name in model.model.exog_names]
    data = model.model.data
    spec = getattr(data, "model_spec", None)
    if spec is None:
        spec = getattr(data, "design_info", None)
    if spec is None:
        const = getattr(data, "const_idx", None)
        terms = {name: [i] for i, name in enumerate(names)}
        return terms, None if const is None else names[const]
    if hasattr(spec, "term_name_slices"):  # patsy's DesignInfo
        slices = dict(spec.term_name_slices)
    else:  # formulaic's ModelSpec
        slices = {str(term): columns for term, columns in spec.term_slices.items()}
    positions = range(len(names))
    terms = {name: list(positions[columns]) for name, columns in slices.items()}
    intercept = next((name for name in ("Intercept", "1") if name in terms), None)
    return terms, intercept


def _aliased_columns(design, tol=1e-7):
    """Indices of the design columns that lie in the span of those before them.

    R's lm() finds aliased coefficients this way, with this tolerance. Working
    from the first column, a column is kept when what is left of it, once the
    columns already kept are projected out, is at least ``tol`` of its length.
    Otherwise, it is aliased, and lm() reports its coefficient as NA.
    """
    basis = np.zeros((design.shape[0], 0))
    aliased = []
    for j in range(design.shape[1]):
        column = design[:, j]
        rest = column - basis @ (basis.T @ column)
        # A second pass keeps the basis orthogonal to working precision.
        rest -= basis @ (basis.T @ rest)
        length = np.linalg.norm(rest)
        if length <= tol * np.linalg.norm(column):
            aliased.append(j)
        else:
            basis = np.column_stack([basis, rest / length])
    return aliased


def _refuse_aliased(terms):
    """Refuse a VIF for aliased coefficients, with the R twin's sentence."""
    raise ValueError(
        f"VIF is undefined when coefficients are aliased: {', '.join(terms)}. "
        "Drop the redundant term(s) and refit."
    )


def _has_coefficient_covariance(obj) -> bool:
    """True for a fitted statsmodels result that holds its coefficients' covariance.

    With the design, that covariance is all a VIF needs. WLS and GLS results
    lack the influence machinery that _is_fitted_model() looks for, so a fit
    of any of statsmodels' linear regression models counts as well, as a
    weighted lm() does in the R twin.
    """
    from statsmodels.regression.linear_model import RegressionModel

    fitted = _is_fitted_model(obj) or isinstance(getattr(obj, "model", None),
                                                 RegressionModel)
    return fitted and getattr(obj, "normalized_cov_params", None) is not None


def _gvif_terms(model) -> pd.DataFrame:
    """Generalised variance inflation factors, one per model term.

    A port of the R twin's internal ``gvif_terms()``. Returns one row per term
    other than the intercept, in model order, with its degrees of freedom (the
    columns it spans), the GVIF of Fox and Monette (1992) and the adjusted
    value GVIF^(1/(2 df)).
    """
    terms, intercept = _model_terms(model)
    exog = np.asarray(model.model.exog, dtype=float)
    # A weighted fit's whitened design gives a row of zero weight no say, as
    # R's lm() leaves such rows out before it looks for aliasing.
    design = np.asarray(getattr(model.model, "wexog", exog), dtype=float)
    aliased = set(_aliased_columns(design))
    if aliased:
        _refuse_aliased([name for name, columns in terms.items()
                         if aliased.intersection(columns)])

    # A categorical with a single level fills no column under patsy or
    # formulaic, so its term has no coefficient and no VIF.
    predictors = {name: cols for name, cols in terms.items()
                  if name != intercept and cols}
    columns = [c for cols in predictors.values() for c in cols]
    if len(columns) < 2:
        raise ValueError("VIF needs at least two predictor columns.")
    if intercept is None:
        # The coefficients' covariance then comes from uncentred columns,
        # which makes predictors far from zero look collinear with each other.
        # car::vif() gives a warning for the same reason.
        warnings.warn(
            "The model has no intercept, so the VIFs come from uncentred "
            "columns and can be inflated by the predictors' means.",
            UserWarning, stacklevel=3,
        )

    # The GVIF compares the confidence region of a term's coefficients with
    # the one they would have if their columns were uncorrelated with the
    # rest. R is therefore the correlation matrix of the coefficient
    # estimates, as in the R twin and car::vif(). It comes from the unscaled
    # covariance, (X'WX)^-1, which statsmodels keeps as normalized_cov_params.
    # For an unweighted linear model with an intercept, the GVIF is the same
    # as from the correlations of the predictor columns. Only the covariance,
    # however, reflects the weights of a weighted fit or of a GLM.
    cov = np.asarray(model.normalized_cov_params, dtype=float)
    cov = cov[np.ix_(columns, columns)]
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    where = {c: i for i, c in enumerate(columns)}
    every = np.arange(len(columns))
    rows = []
    with np.errstate(divide="ignore", invalid="ignore"):
        det_all = np.linalg.det(corr)
        for name, cols in predictors.items():
            inside = np.array([where[c] for c in cols], dtype=int)
            outside = np.setdiff1d(every, inside)
            gvif = (np.linalg.det(corr[np.ix_(inside, inside)])
                    * np.linalg.det(corr[np.ix_(outside, outside)]) / det_all)
            rows.append({"term": name, "df": len(cols), "gvif": gvif,
                         "gvif_adj": gvif ** (1 / (2 * len(cols)))})
    res = pd.DataFrame(rows, columns=["term", "df", "gvif", "gvif_adj"])

    # A GVIF is at least 1 whenever the covariance has full rank. A value that
    # is not finite or not positive means rounding has left it singular,
    # which is aliasing by another route.
    bad = ~np.isfinite(res["gvif"]) | (res["gvif"] <= 0)
    if bad.any():
        _refuse_aliased(list(res.loc[bad, "term"]))
    return res


def vif_plot(model, threshold=5, title=None):
    """Horizontal bar chart of the variance inflation factor of each model term.

    The variance inflation factor (VIF) measures how much collinearity with
    the other predictors inflates the variance of a coefficient. A VIF of 1
    means no collinearity, and the reference line marks ``threshold``, 5 by
    default, a common rule of thumb above which collinearity is treated as a
    concern.

    A term that spans several design columns, such as a factor with more than
    two levels, gets one generalised VIF, the GVIF of Fox and Monette (1992).
    Let R be the correlation matrix of the coefficient estimates other than
    the intercept, R11 its block for the term and R22 the block for the other
    terms. Then GVIF = det(R11) det(R22) / det(R). Unlike the VIFs of a
    factor's separate dummy columns, it does not depend on which level is the
    reference. For a term of one column, it is the ordinary VIF. When any term
    spans several columns, the bars show GVIF^(1/(2 df)) and the line moves to
    the square root of ``threshold``. GVIF^(1/(2 df)) puts terms of different
    sizes on one scale, and for a single column it is the square root of the
    VIF.

    The correlations come from the fit's unscaled coefficient covariance,
    ``normalized_cov_params``, so a weighted fit or a GLM is measured with the
    weights of its fit. For an unweighted linear model with an intercept, they
    give the values the formula gives with R the correlation matrix of the
    predictor columns, as Fox and Monette state it. The values are those of
    the R twin and of R's ``car::vif()``. A Probit result differs slightly,
    because statsmodels takes its covariance from the observed information
    matrix, where R's ``glm()`` uses the expected one. A GLM with a probit
    link gives R's values.

    The columns are grouped into terms by the model's formula, whether
    statsmodels built it with patsy or with formulaic. A model fitted from
    arrays has no formula, so each column is a term of its own, and the dummy
    columns of a factor are not grouped. A model whose coefficients are
    aliased has no VIF, so it raises ``ValueError`` naming the terms they
    belong to, as does a model with fewer than two predictor columns. A pandas
    categorical keeps a category that no row takes, after rows are filtered
    out for instance. Such a category fills a column of zeros, which counts as
    aliased, so drop it with ``Series.cat.remove_unused_categories()`` before
    the fit. R's ``lm()`` drops such a level itself. A model without an
    intercept warns, because its VIFs then come from uncentred columns.

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS, WLS, GLM, Logit or Probit result with two or more
        predictor columns.
    threshold : float, default 5
        The VIF taken as the rule of thumb, drawn as the reference line. On
        the GVIF^(1/(2 df)) scale, it is drawn at ``sqrt(threshold)``.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
        Its ``data`` hold one row per term, with the columns ``term``, ``df``,
        ``gvif`` and ``vif``, the value drawn.

    References
    ----------
    Fox, J., & Monette, G. (1992). Generalized collinearity diagnostics.
    Journal of the American Statistical Association, 87(417), 178-183.
    https://doi.org/10.1080/01621459.1992.10475190

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> p = dp.vif_plot(model)

    A factor with several levels gets one generalised VIF:

    >>> ws = dp.wellbeing_survey()
    >>> fit = smf.ols("life_satisfaction ~ stress + region + education", ws).fit()
    >>> p = dp.vif_plot(fit)
    """
    _require_statsmodels()
    if not _has_coefficient_covariance(model):
        raise TypeError("vif_plot needs a fitted statsmodels result.")
    if (isinstance(threshold, bool) or not isinstance(threshold, numbers.Real)
            or not math.isfinite(threshold) or threshold <= 0):
        raise ValueError("`threshold` must be a single positive number.")

    res = _gvif_terms(model)
    # A term of one column has an ordinary VIF, drawn against the threshold
    # itself. A term of several columns has no scalar VIF. When there is one,
    # every term is drawn as GVIF^(1/(2 df)) and the line moves to the square
    # root of the threshold, as in the R twin.
    if (res["df"] > 1).any():
        value, line = res["gvif_adj"], math.sqrt(threshold)
        axis = r"$\mathrm{GVIF}^{1/(2\,\mathrm{df})}$"
    else:
        value, line, axis = res["gvif"], threshold, "Variance inflation factor"
    df = res[["term", "df", "gvif"]].assign(vif=value)
    df = df.sort_values("vif", kind="stable")
    # Lock the order so the largest VIF reads at the top after the flip.
    df["term"] = pd.Categorical(df["term"], categories=list(df["term"]), ordered=True)

    return (
        ggplot(df, aes("term", "vif"))
        + geom_col(fill=BRAND, width=0.7)
        + geom_hline(yintercept=line, color="#666666")
        + coord_flip()
        + labs(x="", y=axis, title=title)
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
    https://doi.org/10.1017/CBO9780511790942

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
