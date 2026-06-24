"""Model-estimate plots that accept a fitted model *or* a tidy data frame.

The dual input is the point: hand :func:`coefficient_plot` a fitted statsmodels
result and it reads the estimates and confidence intervals itself, or hand it a
tidy data frame of estimates from any source (a Bayesian fit, a bootstrap, a
table from a paper) and it plots those directly.
"""

from __future__ import annotations

import pandas as pd
from plotnine import (
    aes,
    geom_errorbarh,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    scale_y_discrete,
    theme,
)

from .palette import BRAND, depictr_brand
from .theme import theme_depictr

# Tidy columns every downstream function relies on.
_TIDY = ["term", "estimate", "conf_low", "conf_high"]


def tidy_estimates(model, conf_level: float = 0.95) -> pd.DataFrame:
    """Coerce a fitted model or an estimate table into one tidy frame.

    Parameters
    ----------
    model : statsmodels results object or pandas.DataFrame
        A fitted model exposing ``params`` and ``conf_int`` (for example a
        statsmodels ``OLS``/``GLM`` result), or a data frame already carrying the
        estimates.
    conf_level : float
        Confidence level for the interval when reading it from a model.

    Returns
    -------
    pandas.DataFrame
        Columns ``term``, ``estimate``, ``conf_low``, ``conf_high``.
    """
    if isinstance(model, pd.DataFrame):
        return _tidy_from_frame(model)
    if hasattr(model, "params") and hasattr(model, "conf_int"):
        ci = model.conf_int(alpha=1 - conf_level)
        ci = pd.DataFrame(ci)
        ci.columns = ["conf_low", "conf_high"][: ci.shape[1]]
        out = pd.DataFrame({
            "term": list(model.params.index),
            "estimate": model.params.to_numpy(),
            "conf_low": ci.iloc[:, 0].to_numpy(),
            "conf_high": ci.iloc[:, 1].to_numpy(),
        })
        return out
    raise TypeError(
        "`model` must be a fitted statsmodels result or a tidy DataFrame with "
        "columns term/estimate/conf_low/conf_high."
    )


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
        Order of the terms up the axis.
    conf_level : float
        Confidence level passed to :func:`tidy_estimates`.
    title : str, optional

    Returns
    -------
    plotnine.ggplot
    """
    est = tidy_estimates(model, conf_level=conf_level)
    if not intercept:
        est = est[~est["term"].str.fullmatch(r"(?i)\(?intercept\)?|const")]
    if order == "ascending":
        est = est.sort_values("estimate")
    elif order == "descending":
        est = est.sort_values("estimate", ascending=False)
    # Reverse so the first term reads at the top.
    levels = list(est["term"])[::-1]
    est = est.assign(term=pd.Categorical(est["term"], categories=levels, ordered=True))
    return (
        ggplot(est, aes(x="estimate", y="term"))
        + geom_vline(xintercept=0, linetype="dashed", color="#9e9e9e")
        + geom_errorbarh(aes(xmin="conf_low", xmax="conf_high"), height=0.15,
                         color=depictr_brand(), size=0.8)
        + geom_point(color=depictr_brand(), size=2.6)
        + labs(x="Estimate", y=None, title=title)
        + theme_depictr()
    )
