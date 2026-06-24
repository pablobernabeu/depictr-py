"""Forest plots for posterior or bootstrap draws.

A Bayesian fit or a bootstrap gives a sample of draws for each parameter rather
than a single estimate and a closed-form interval. These two functions summarise
that sample as a forest: the median as a point, with a thick inner band and a
thin outer band for two credible intervals. The default bands are the central
66% and 95%, the pair the ``bayesplot`` and ``brms`` packages draw, chosen so the
eye reads the bulk of the posterior (the inner band) without losing the tails
(the outer band).

:func:`frequentist_bayesian_plot` puts a frequentist estimate alongside the
posterior, one row per shared term, so the two ways of quantifying uncertainty
sit side by side for the same model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    geom_hline,
    geom_linerange,
    geom_point,
    ggplot,
    labs,
    position_dodge,
)

from .palette import BRAND
from .theme import scale_colour_depictr, theme_depictr

# Inner and outer credible-interval masses, as the bayesplot/brms defaults.
_INNER = 0.66
_OUTER = 0.95


def _draws_to_frame(draws) -> pd.DataFrame:
    """Coerce a draws container to a DataFrame, one column per parameter.

    Accepts a :class:`pandas.DataFrame` (returned as-is) or a mapping from
    parameter name to a 1-D array of draws. The arrays in a mapping may differ
    in length, so they are aligned into a frame column by column.
    """
    if isinstance(draws, pd.DataFrame):
        return draws
    if isinstance(draws, dict):
        if not draws:
            raise ValueError("`draws` is empty.")
        return pd.DataFrame({k: pd.Series(np.asarray(v).ravel())
                             for k, v in draws.items()})
    raise TypeError(
        "`draws` must be a pandas DataFrame (one column per parameter) or a "
        "dict of 1-D arrays of draws."
    )


def _summarise_draws(draws, labels=None) -> pd.DataFrame:
    """Median and the two credible intervals for every parameter column.

    Returns a tidy frame with columns ``term``, ``median``, and the inner/outer
    interval bounds. ``labels`` remaps the raw parameter names to display names.
    """
    frame = _draws_to_frame(draws)
    inner_lo, inner_hi = (1 - _INNER) / 2, (1 + _INNER) / 2
    outer_lo, outer_hi = (1 - _OUTER) / 2, (1 + _OUTER) / 2
    rows = []
    for name in frame.columns:
        col = frame[name].to_numpy(dtype=float)
        col = col[~np.isnan(col)]
        if col.size == 0:
            raise ValueError(f"parameter {name!r} has no non-missing draws.")
        ql = np.quantile(col, [outer_lo, inner_lo, inner_hi, outer_hi])
        rows.append({
            "term": str(name),
            "median": float(np.median(col)),
            "outer_low": ql[0], "inner_low": ql[1],
            "inner_high": ql[2], "outer_high": ql[3],
        })
    out = pd.DataFrame(rows)
    if labels is not None:
        out["term"] = out["term"].map(lambda t: labels.get(t, t))
    return out


def posterior_plot(draws, labels=None, title=None):
    """Forest plot of posterior (or bootstrap) draws.

    Each parameter is one row: the median as a point, a thick inner band for the
    central 66% credible interval and a thin outer band for the central 95%. The
    first parameter reads at the top.

    Parameters
    ----------
    draws : pandas.DataFrame or dict of array-like
        The draws, one column (or dict entry) per parameter. A dict maps a
        parameter name to a 1-D array of draws; the arrays may differ in length.
    labels : dict, optional
        Remap raw parameter names to display names, ``{raw: shown}``. Names not
        in the mapping are left unchanged.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    est = _summarise_draws(draws, labels=labels)
    # Reverse so the first parameter sits at the top once the axes are flipped.
    levels = list(est["term"])[::-1]
    est = est.assign(term=pd.Categorical(est["term"], categories=levels,
                                         ordered=True))
    # geom_linerange is vertical (x, ymin, ymax), so build it upright on the
    # estimate scale and flip to a horizontal forest at the end.
    return (
        ggplot(est, aes(x="term"))
        + geom_hline(yintercept=0, linetype="dashed", color="#9e9e9e")
        + geom_linerange(aes(ymin="outer_low", ymax="outer_high"),
                         color=BRAND, size=0.8)
        + geom_linerange(aes(ymin="inner_low", ymax="inner_high"),
                         color=BRAND, size=1.8)
        + geom_point(aes(y="median"), color=BRAND, size=2.8)
        + coord_flip()
        + labs(x=None, y="Estimate", title=title)
        + theme_depictr()
    )


def frequentist_bayesian_plot(frequentist, bayesian, title=None):
    """Overlay a frequentist estimate against a Bayesian posterior per term.

    For each term the two sources share a row, offset slightly so they do not
    overlap. The frequentist side shows the point estimate with its confidence
    interval; the Bayesian side shows the posterior median with the inner 66% and
    outer 95% credible intervals. The sources are told apart by colour (brand
    blue for frequentist, accent orange for Bayesian).

    Only terms present in both sources are drawn. The frequentist confidence
    level is whatever :func:`depictr.models.tidy_estimates` reads from the model
    (95% for a fitted statsmodels result), matching the Bayesian outer band.

    Parameters
    ----------
    frequentist : statsmodels results object or pandas.DataFrame
        A fitted model or a tidy estimate frame, as accepted by
        :func:`depictr.models.tidy_estimates`.
    bayesian : pandas.DataFrame or dict of array-like
        Posterior draws, one column (or dict entry) per term, as accepted by
        :func:`posterior_plot`.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    from .models import tidy_estimates

    freq = tidy_estimates(frequentist).rename(columns={
        "estimate": "median", "conf_low": "outer_low", "conf_high": "outer_high",
    })
    # The frequentist CI has no inner band; leave it absent for that source.
    freq = freq.assign(inner_low=np.nan, inner_high=np.nan,
                       source="Frequentist")
    bayes = _summarise_draws(bayesian).assign(source="Bayesian")

    shared = [t for t in freq["term"] if t in set(bayes["term"])]
    if not shared:
        raise ValueError(
            "the frequentist and Bayesian inputs share no term names."
        )
    both = pd.concat([freq[freq["term"].isin(shared)],
                      bayes[bayes["term"].isin(shared)]], ignore_index=True)

    # First shared term at the top once flipped; sources keyed to brand vs accent.
    both = both.assign(
        term=pd.Categorical(both["term"], categories=shared[::-1], ordered=True),
        source=pd.Categorical(both["source"],
                              categories=["Frequentist", "Bayesian"]),
    )
    dodge = position_dodge(width=0.5)
    return (
        ggplot(both, aes(x="term", color="source"))
        + geom_hline(yintercept=0, linetype="dashed", color="#9e9e9e")
        + geom_linerange(aes(ymin="outer_low", ymax="outer_high"),
                         position=dodge, size=0.8)
        + geom_linerange(aes(ymin="inner_low", ymax="inner_high"),
                         position=dodge, size=1.8, na_rm=True)
        + geom_point(aes(y="median"), position=dodge, size=2.8)
        + scale_colour_depictr(name="Source")
        + coord_flip()
        + labs(x=None, y="Estimate", title=title)
        + theme_depictr()
    )
