"""Plots for mixed-effects models fitted with statsmodels.

The estimation is left to statsmodels' ``MixedLM``; this module reads the fitted
result and re-draws it under the depictr theme. The one plot here is the
caterpillar plot of the predicted random effects (the BLUPs), the standard way to
look at how group levels deviate from the population fit.

Install the optional dependency with ``pip install depictr[mixed]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    facet_wrap,
    geom_errorbar,
    geom_hline,
    geom_point,
    ggplot,
    labs,
    theme,
)

from .palette import BRAND, depictr_brand
from .theme import theme_depictr


def _require_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The mixed-effects plots need statsmodels. Install it with "
            "`pip install depictr[mixed]`."
        ) from exc
    return sm


def random_effects_plot(model, title=None):
    """Caterpillar plot of the predicted random effects (BLUPs).

    Each point is one group level's predicted random effect, the deviation of
    that group from the population fit. Levels are sorted by their effect, so the
    plot reads as a tilted ladder from the most negative to the most positive
    group, against a zero reference line. A group whose interval clears zero
    stands apart from the average; one straddling zero does not.

    The points come straight from ``model.random_effects``. When the fit exposes
    the conditional covariance of each group's effects
    (``model.random_effects_cov``), the diagonal gives a conditional standard
    error and the plot adds a 95% interval (the point estimate plus or minus
    1.96 standard errors); without it the points are drawn on their own. If the
    model has more than one random-effect term per group (a random intercept and
    one or more random slopes), the terms are shown in separate panels.

    Parameters
    ----------
    model : statsmodels MixedLMResults
        A fitted ``MixedLM`` result, as returned by ``MixedLM(...).fit()`` or
        ``smf.mixedlm(...).fit()``.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot

    Notes
    -----
    The interval is the conditional one around each predicted effect, not a
    confidence interval for a fixed parameter. It reflects how precisely that
    group's effect is pinned down given the fitted variance components.
    """
    _require_statsmodels()
    if not hasattr(model, "random_effects"):
        raise TypeError(
            "random_effects_plot needs a fitted statsmodels MixedLM result "
            "(one exposing `random_effects`)."
        )

    effects = model.random_effects
    if not effects:
        raise ValueError("The fitted model has no random effects to plot.")

    # Conditional covariance per group, when statsmodels provides it. The
    # diagonal of each group's matrix is the conditional variance of its BLUPs.
    cov = getattr(model, "random_effects_cov", None)

    rows = []
    for group, eff in effects.items():
        eff = pd.Series(eff)
        se = None
        if cov is not None and group in cov:
            var = np.diag(np.asarray(cov[group]))
            se = pd.Series(np.sqrt(var), index=eff.index)
        for term in eff.index:
            value = float(eff[term])
            this_se = float(se[term]) if se is not None else None
            rows.append({
                "group": str(group),
                "term": str(term),
                "effect": value,
                "low": value - 1.96 * this_se if this_se is not None else np.nan,
                "high": value + 1.96 * this_se if this_se is not None else np.nan,
            })
    df = pd.DataFrame(rows)
    has_interval = df["low"].notna().any()

    # One panel per random-effect term; sort the group axis within each term so
    # the most negative effect sits at the bottom after the flip.
    df = df.sort_values(["term", "effect"]).reset_index(drop=True)
    # A categorical group axis with a per-term ordering: build the level list in
    # plotting order. When there are several terms the same group appears under
    # each, so the order is taken from the (already sorted) appearance.
    order = list(dict.fromkeys(df["group"]))
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)

    p = (
        ggplot(df, aes(x="group", y="effect"))
        + geom_hline(yintercept=0, linetype="dashed", color="#9e9e9e")
    )
    if has_interval:
        p = p + geom_errorbar(aes(ymin="low", ymax="high"), width=0,
                              color=depictr_brand(), size=0.7, na_rm=True)
    p = (
        p
        + geom_point(color=depictr_brand(), size=2.4)
        + coord_flip()
        + labs(x=None, y="Predicted random effect", title=title)
        + theme_depictr()
    )

    n_terms = df["term"].nunique()
    if n_terms > 1:
        p = p + facet_wrap("term", scales="free_y")
    return p
