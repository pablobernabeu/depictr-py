"""Model-estimate plots that accept a fitted model *or* a tidy data frame.

The dual input is the point: hand :func:`coefficient_plot` a fitted statsmodels
result and it reads the estimates and confidence intervals itself, or hand it a
tidy data frame of estimates from any source (a Bayesian fit, a bootstrap, a
table from a paper) and it plots those directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_errorbarh,
    geom_point,
    geom_vline,
    ggplot,
    labs,
)

from .palette import depictr_brand
from .theme import theme_depictr

# Tidy columns every downstream function relies on.
_TIDY = ["term", "estimate", "conf_low", "conf_high"]


def tidy_estimates(model, conf_level: float = 0.95) -> pd.DataFrame:
    """Coerce a fitted model or an estimate table into one tidy frame.

    Parameters
    ----------
    model : statsmodels results object or pandas.DataFrame
        A fitted model exposing ``params`` and ``conf_int`` (for example a
        statsmodels ``OLS``, ``GLM`` or ``MixedLM`` result, fitted from a formula
        or from arrays), or a data frame already carrying the estimates.
    conf_level : float
        Confidence level for the interval when reading it from a model.

    Returns
    -------
    pandas.DataFrame
        Columns ``term``, ``estimate``, ``conf_low``, ``conf_high``.

    Notes
    -----
    For a mixed model, only the fixed effects are kept, as the R twin keeps
    ``lme4::fixef()``. The variance and covariance entries that follow them in
    a ``MixedLM`` result's ``params`` are not coefficients, and statsmodels
    reports them divided by the residual scale. The intervals kept are Wald
    intervals on the normal distribution, as the R twin computes them for an
    lme4 fit.

    A model fitted from arrays has unnamed estimates, so its terms are named
    from the model's ``exog_names`` (``const``, ``x1``, ...), as statsmodels'
    own summary names them. Terms read from a model are always strings, as in
    R, so the numbered columns of a data frame made from an array give the
    terms ``"0"``, ``"1"`` and so on.

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> dp.tidy_estimates(model).columns.tolist()
    ['term', 'estimate', 'conf_low', 'conf_high']
    """
    if isinstance(model, pd.DataFrame):
        return _tidy_from_frame(model)
    if hasattr(model, "params") and hasattr(model, "conf_int"):
        return _tidy_from_results(model, conf_level)
    raise TypeError(
        "`model` must be a fitted statsmodels result or a tidy DataFrame with "
        "columns term/estimate/conf_low/conf_high."
    )


def _tidy_from_results(model, conf_level: float) -> pd.DataFrame:
    """Read the estimates and intervals of a fitted model, fixed effects only."""
    params = model.params
    estimate = np.asarray(params, dtype=float)
    ci = np.asarray(model.conf_int(alpha=1 - conf_level), dtype=float)
    # A formula fit labels its params with a pandas index. A fit from arrays
    # returns a bare ndarray, and the names live on the model. The type test
    # matters because a list also has an index attribute, a method.
    if isinstance(params, pd.Series):
        names = params.index
    else:
        names = getattr(getattr(model, "model", None), "exog_names", None)
    k_fe = getattr(model, "k_fe", None)
    if k_fe is not None:
        # A MixedLM result's params hold the k_fe fixed effects first, then the
        # random-effect variances and covariances and any variance components.
        estimate, ci = estimate[:k_fe], ci[:k_fe]
        fe_params = getattr(model, "fe_params", None)
        if isinstance(fe_params, pd.Series):
            names = fe_params.index
        elif names is not None:
            names = names[:k_fe]
    if names is None or len(names) != len(estimate):
        raise TypeError(
            "Cannot name the estimates of `model`: its `params` have no index "
            "and its model's `exog_names` do not give one name per estimate. "
            "Pass a tidy DataFrame with columns term/estimate/conf_low/conf_high."
        )
    return pd.DataFrame({
        # Strings, as the R twin returns them. A data frame made from an array
        # numbers its columns, and the plots cannot filter integer terms.
        "term": [str(name) for name in names],
        "estimate": estimate,
        "conf_low": ci[:, 0],
        "conf_high": ci[:, 1],
    })


def _tidy_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a user-supplied estimate frame to the tidy column names."""
    rename = {
        "conf.low": "conf_low", "conf.high": "conf_high",
        "lower": "conf_low", "upper": "conf_high",
        "ci_low": "conf_low", "ci_high": "conf_high",
        "coef": "estimate", "coefficient": "estimate",
    }
    out = df.rename(columns=rename).copy()
    missing = [c for c in _TIDY if c not in out.columns]
    if missing:
        raise ValueError(f"estimate frame is missing column(s): {missing}")
    return out[_TIDY]


def coefficient_plot(model, intercept: bool = False, order: str = "model",
                     conf_level: float = 0.95, title=None):
    """Forest (dot-and-whisker) plot of model coefficients.

    Parameters
    ----------
    model : statsmodels results object or pandas.DataFrame
        See :func:`tidy_estimates`.
    intercept : bool
        Whether to include the intercept term.
    order : {"model", "ascending", "descending"}
        Order of the terms. ``"ascending"`` puts the smallest estimate at the
        bottom and the largest at the top, so estimates increase up the axis.
        ``"descending"`` does the reverse, and ``"model"`` keeps the model's
        order from top to bottom. The R twin reads ``order`` the same way.
    conf_level : float
        Confidence level passed to :func:`tidy_estimates`.
    title : str, optional

    Returns
    -------
    plotnine.ggplot

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> p = dp.coefficient_plot(model, title="Drivers of crop yield")
    """
    est = tidy_estimates(model, conf_level=conf_level)
    if not intercept:
        est = est[~est["term"].str.fullmatch(r"(?i)\(?intercept\)?|const")]
    # plotnine draws the first category of the y axis at the bottom. A sorted
    # order is therefore listed as it stands, so estimates increase up the axis
    # under "ascending", while model order is reversed so that the first term
    # reads at the top. The stable sort keeps tied terms in model order, as R's
    # order() does.
    if order == "ascending":
        est = est.sort_values("estimate", kind="stable")
    elif order == "descending":
        est = est.sort_values("estimate", ascending=False, kind="stable")
    levels = list(est["term"])
    if order not in ("ascending", "descending"):
        levels = levels[::-1]
    est = est.assign(term=pd.Categorical(est["term"], categories=levels, ordered=True))
    return (
        ggplot(est, aes(x="estimate", y="term"))
        + geom_vline(xintercept=0, linetype="dashed", color="#9e9e9e")
        + geom_errorbarh(aes(xmin="conf_low", xmax="conf_high"), height=0.15,
                         color=depictr_brand(), size=0.8)
        + geom_point(color=depictr_brand(), size=2.6)
        + labs(x="Estimate", y="", title=title)
        + theme_depictr()
    )
