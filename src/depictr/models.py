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
    scale_x_log10,
)

from ._checks import _check_conf_level
from .palette import depictr_brand
from .terms import (
    _INTERCEPT_PATTERN,
    _check_distinct,
    _check_label_count,
    _check_labels,
    _design_labels,
    _term_labels,
    _warn_unused_labels,
    interaction_separator,
)
from .theme import theme_depictr

# Tidy columns every downstream function relies on.
_TIDY = ["term", "estimate", "conf_low", "conf_high"]
# The orders coefficient_plot() takes. "none" is the R twin's name for "model".
_ORDERS = ("model", "none", "ascending", "descending")


def tidy_estimates(model, conf_level: float = 0.95,
                   exponentiate: bool = False) -> pd.DataFrame:
    """Coerce a fitted model or an estimate table into one tidy frame.

    Parameters
    ----------
    model : statsmodels results object or pandas.DataFrame
        A fitted model exposing ``params`` and ``conf_int`` (for example a
        statsmodels ``OLS``, ``GLM`` or ``MixedLM`` result, fitted from a formula
        or from arrays), or a data frame of estimates with the columns ``term``,
        ``estimate``, ``conf_low`` and ``conf_high``. broom's ``conf.low`` and
        ``conf.high``, ``lower`` and ``upper`` or ``ci_low`` and ``ci_high``
        are read as the interval limits, and ``coef`` or ``coefficient`` as the
        estimate. Other columns are dropped, and a frame without the four
        raises ``ValueError``.
    conf_level : float
        Confidence level for the interval when reading it from a model, a
        single number strictly between 0 and 1.
    exponentiate : bool
        Whether to exponentiate the estimates and interval limits, which turns
        log-odds into odds ratios and log rates into rate ratios. A data frame
        is read as estimates on the link scale. Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        Columns ``term``, ``estimate``, ``conf_low``, ``conf_high``.

    Notes
    -----
    The intervals are those of the result's ``conf_int()``, so they follow its
    ``use_t``. An OLS result uses t. A GLM uses z by default, as statsmodels'
    own GLM summary does, and t on the residual degrees of freedom when fitted
    with ``use_t=True``. The R twin gives a family whose dispersion is
    estimated (Gaussian, Gamma, inverse Gaussian) the t interval that R's
    ``summary.glm()`` tests with, so fit such a GLM with ``use_t=True`` to get
    the R twin's intervals. At 12 residual degrees of freedom, the t interval
    is about 11% wider than the z interval, and at 100 about 1% wider.

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
    _check_conf_level(conf_level)
    _check_exponentiate(exponentiate)
    if isinstance(model, pd.DataFrame):
        return _exponentiate(_tidy_from_frame(model), exponentiate)
    if hasattr(model, "params") and hasattr(model, "conf_int"):
        return _exponentiate(_tidy_from_results(model, conf_level), exponentiate)
    raise TypeError(
        "`model` must be a fitted statsmodels result or a tidy DataFrame with "
        "columns term/estimate/conf_low/conf_high."
    )


def _check_exponentiate(exponentiate) -> None:
    """Refuse an ``exponentiate`` that is not a single True or False."""
    if not isinstance(exponentiate, (bool, np.bool_)):
        raise TypeError("`exponentiate` must be True or False.")


def _exponentiate(est: pd.DataFrame, exponentiate: bool) -> pd.DataFrame:
    """Exponentiate the estimates and interval limits of a tidy frame."""
    if not exponentiate:
        return est
    out = est.copy()
    for col in ("estimate", "conf_low", "conf_high"):
        out[col] = np.exp(out[col].to_numpy(dtype=float))
    return out


# statsmodels' discrete models carry no family object: the class fixes the
# link, logit for Logit and log for the count models.
_CLASS_LINKS = {
    "logit": "logit",
    "poisson": "log",
    "negativebinomial": "log",
    "negativebinomialp": "log",
    "generalizedpoisson": "log",
}
_COUNT_MODELS = {"poisson", "negativebinomial", "negativebinomialp",
                 "generalizedpoisson"}


def _ratio_label(model) -> str:
    """Name an exponentiated estimate by the model's family and link.

    exp() of a logit coefficient is an odds ratio. exp() of a log-link
    coefficient is a rate ratio in a count model and a risk ratio in a
    binomial one. Anything else, a data frame or an identity link among them,
    is "exp(Estimate)": exp() of a slope on log(y), for one, is a ratio of
    geometric means. The R twin names them the same way.
    """
    if isinstance(model, pd.DataFrame):
        return "exp(Estimate)"
    inner = getattr(model, "model", None)
    family = getattr(inner, "family", None)
    if family is not None:
        name = type(family).__name__.lower()
        # Lower-cased, since older statsmodels releases name the links in
        # lower case (logit, log) and newer ones in title case.
        link = type(getattr(family, "link", None)).__name__.lower()
    else:
        name = type(inner).__name__.lower()
        link = _CLASS_LINKS.get(name, "")
    if link == "logit":
        return "Odds ratio"
    if link == "log":
        if name == "binomial":
            return "Risk ratio"
        if name in _COUNT_MODELS:
            return "Rate ratio"
    return "exp(Estimate)"


def _reference_position(reference_line, exponentiate: bool):
    """Where the reference line goes: 0, 1 for ratios, or None for no line."""
    if reference_line is None:
        return 1.0 if exponentiate else 0.0
    value = float(reference_line)
    return None if np.isnan(value) else value


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
                     conf_level: float = 0.95, title=None, reference_line=None,
                     exponentiate: bool = False, labels=None,
                     interaction: str = "times"):
    """Forest (dot-and-whisker) plot of model coefficients.

    By default each term is labelled for a reader. The coefficient of a factor
    level names its variable and level, so patsy's ``treatment[T.enhanced]``
    is drawn as ``treatment: enhanced``, the contrast of enhanced with the
    reference level, and ``rainfall:treatment[T.enhanced]`` as
    ``rainfall × treatment: enhanced``. For a formula fit, the levels are read
    from the model's design, so a level holding a colon or a bracket is kept
    whole. Other names, and the terms of a data frame, are tidied with
    [`format_terms`][depictr.format_terms], which also drops the ``Q()`` and
    ``C()`` wrappers of a formula. The R twin labels its terms by the same
    rules.

    Parameters
    ----------
    model : statsmodels results object or pandas.DataFrame
        See [`tidy_estimates`][depictr.tidy_estimates].
    intercept : bool
        Whether to include the intercept term.
    order : {"model", "none", "ascending", "descending"}
        Order of the terms. ``"ascending"`` puts the smallest estimate at the
        bottom and the largest at the top, so estimates increase up the axis.
        ``"descending"`` does the reverse, and ``"model"`` keeps the model's
        order from top to bottom. ``"none"``, the R twin's name for model
        order, is accepted as an alias. The R twin reads ``order`` the same
        way. Any other value raises ``ValueError``.
    conf_level : float
        Confidence level passed to [`tidy_estimates`][depictr.tidy_estimates],
        a single number strictly between 0 and 1.
    title : str, optional
    reference_line : float, optional
        Position of the dashed vertical reference line. ``None``, the default,
        puts it at 0, or at 1 when ``exponentiate`` is set, where a ratio of 1
        means no effect. ``float("nan")`` leaves it out, as ``NA`` does in the
        R twin.
    exponentiate : bool
        Whether to draw the exponentiated estimates and intervals on a log10
        axis, which shows the coefficients of a logistic regression as odds
        ratios and those of a Poisson or negative binomial model as rate
        ratios. The axis is labelled "Odds ratio" for a logit link, "Rate
        ratio" for a log link in a count model, "Risk ratio" for a binomial
        model with a log link and "exp(Estimate)" otherwise. A data frame
        ``model`` must hold estimates on the link scale. Defaults to
        ``False``.
    labels : dict, list or "raw", optional
        Display labels for the terms. A dict maps raw term names to labels,
        such as ``{"soil_ph": "Soil pH"}``, and leaves the other terms to the
        default labels. A key that matches no term warns, since a mistyped one
        would otherwise change nothing. A list gives one label per term in
        model order (the order [`tidy_estimates`][depictr.tidy_estimates]
        returns, after the intercept is dropped unless ``intercept=True``),
        whatever ``order`` is, so each label stays with its term when the
        terms are sorted. Under the default order, it reads from top to
        bottom. ``"raw"`` draws the names as the model gives them. Two terms
        may not share a label.
    interaction : {"times", "asterisk", "colon", "space"}
        How an interaction is written in the default labels, passed to
        [`format_terms`][depictr.format_terms]: a multiplication sign by
        default.

    Returns
    -------
    plotnine.ggplot
        The plot's data keeps the raw names in ``term`` and the drawn ones in
        ``label``.

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + treatment', cy).fit()
    >>> p = dp.coefficient_plot(model, title="Drivers of crop yield")
    >>> list(p.data["label"].cat.categories)[::-1]
    ['treatment: enhanced', 'fertiliser', 'rainfall']
    >>> p = dp.coefficient_plot(model, labels={"treatment[T.enhanced]": "Enhanced"})

    Odds ratios from a logistic regression, on a log axis with the line at 1:

    >>> ld = dp.lexical_decision()
    >>> logit = smf.logit("accuracy ~ word_frequency + condition", ld).fit(disp=0)
    >>> p = dp.coefficient_plot(logit, exponentiate=True)
    """
    _check_conf_level(conf_level)
    if not isinstance(order, str) or order not in _ORDERS:
        # Unchecked, an unknown order, a typo among them, would fall back to
        # model order without a word. R's match.arg() refuses it.
        raise ValueError(
            "`order` must be one of 'model' (or its alias 'none'), "
            "'ascending' or 'descending'."
        )
    _check_exponentiate(exponentiate)
    interaction_separator(interaction)
    _check_labels(labels)
    line = _reference_position(reference_line, exponentiate)
    est = tidy_estimates(model, conf_level=conf_level, exponentiate=exponentiate)
    # Judged against every term, so labelling one that is merely not drawn,
    # such as the dropped intercept, is not reported as a typo.
    _warn_unused_labels(est["term"], labels)
    auto = {} if isinstance(labels, str) else _design_labels(model, interaction)
    intercept_dropped = False
    if not intercept:
        is_intercept = est["term"].str.fullmatch(_INTERCEPT_PATTERN)
        intercept_dropped = bool(is_intercept.any())
        est = est[~is_intercept]
    # Dropping the only term, the intercept of an intercept-only model, would
    # leave a themed but empty panel. The R twin's check and message.
    if est.empty:
        raise ValueError("No terms left to plot (did you drop the only term?).")
    # A list is read in model order, so it is attached to the rows before they
    # are sorted, and each label travels with its own term.
    if isinstance(labels, (list, tuple)):
        _check_label_count(labels, len(est), "the model has", "in model order",
                           intercept_dropped)
        est = est.assign(label=[str(label) for label in labels])
    else:
        est = est.assign(label=_term_labels(est["term"], labels, auto, interaction))
    _check_distinct(est["term"], est["label"])
    # plotnine draws the first category of the y axis at the bottom. A sorted
    # order is therefore listed as it stands, so estimates increase up the axis
    # under "ascending", while model order is reversed so that the first term
    # reads at the top. The stable sort keeps tied terms in model order, as R's
    # order() does.
    if order == "ascending":
        est = est.sort_values("estimate", kind="stable")
    elif order == "descending":
        est = est.sort_values("estimate", ascending=False, kind="stable")
    terms, drawn = list(est["term"]), list(est["label"])
    if order not in ("ascending", "descending"):
        terms, drawn = terms[::-1], drawn[::-1]
    est = est.assign(
        term=pd.Categorical(est["term"], categories=list(dict.fromkeys(terms)),
                            ordered=True),
        label=pd.Categorical(est["label"], categories=drawn, ordered=True),
    )
    p = ggplot(est, aes(x="estimate", y="label"))
    if line is not None:
        p = p + geom_vline(xintercept=line, linetype="dashed", color="#9e9e9e")
    p = (
        p
        + geom_errorbarh(aes(xmin="conf_low", xmax="conf_high"), height=0.15,
                         color=depictr_brand(), size=0.8)
        + geom_point(color=depictr_brand(), size=2.6)
        + labs(x=_ratio_label(model) if exponentiate else "Estimate", y="",
               title=title)
        + theme_depictr()
    )
    # Ratios are symmetric on a log scale, where 0.5 and 2 sit equally far
    # from 1.
    return p + scale_x_log10() if exponentiate else p
