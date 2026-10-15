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
marginal average over the sample. The predictors are the columns the formula
names, and their means, levels and ranges are taken over the rows the model was
fitted to. A model fitted with an ``offset`` or ``exposure`` is predicted at the
mean offset over those rows on the link scale, which for an exposure is its
geometric mean, as in the R twin.

``statsmodels`` arrives with the core install (plotnine itself requires it);
the ``depictr[models]`` extra remains to pin the tested floor.
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

from ._checks import _check_conf_level
from .palette import BRAND
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
from .theme import scale_colour_depictr, scale_fill_depictr, theme_depictr


def _require_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The prediction plots need statsmodels. Install it with "
            '`pip install "depictr[models]"`.'
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


def _estimation_frame(model) -> pd.DataFrame:
    """The rows of the source data the model was fitted to.

    statsmodels keeps the whole source frame, including rows its missing-value
    handling dropped, and records the positions of those rows in
    ``missing_row_idx`` and the labels of the rows it used in ``row_labels``.
    Typical values and ranges are taken over the rows it used, so they are
    statistics of the estimation sample. Positions pick them out even when the
    frame's index repeats a label, as after ``pd.concat()``.
    """
    frame = _model_frame(model)
    data = model.model.data
    dropped = getattr(data, "missing_row_idx", None) or []
    keep = np.ones(len(frame), dtype=bool)
    keep[np.asarray(dropped, dtype=int)] = False
    if keep.sum() == len(model.model.endog):
        return frame[keep]
    rows = getattr(data, "row_labels", None)
    if rows is not None and frame.index.is_unique:
        return frame.loc[rows]
    raise ValueError(
        "The rows this model was fitted to cannot be found in its source data, "
        "so a prediction grid cannot be built."
    )


def _response_label(model) -> str:
    """A readable name for the response, unwrapping a formula Q(\"...\") guard."""
    name = str(model.model.endog_names)
    return re.sub(r'Q\("([^"]+)"\)', r"\1", name)


# In a formula: a Q("...") name, any other string literal or an identifier.
_FORMULA_TOKEN = re.compile(
    r"""Q\(\s*(?P<quoted>"[^"]*"|'[^']*')\s*\)"""
    r"""|(?P<string>"[^"]*"|'[^']*')"""
    r"""|(?P<name>[^\W\d]\w*)"""
)


def _predictor_columns(model, frame: pd.DataFrame) -> list[str]:
    """Source columns the formula names as predictors, in formula order.

    The right-hand side names a column as an identifier, as in ``np.log(rainfall)``
    or ``C(treatment)``, or inside ``Q("...")``. A column it never names is no
    predictor, since holding or sweeping it changes nothing the model predicts.
    A model without a formula falls back to every column but the response.
    """
    response = _response_label(model)
    formula = getattr(model.model, "formula", None)
    if not formula:
        return [c for c in frame.columns if c != response]
    formula = str(formula)
    # The first "~" outside a string literal separates the response.
    masked = re.sub(r""""[^"]*"|'[^']*'""", lambda m: " " * len(m.group()), formula)
    rhs = formula[masked.find("~") + 1:]
    named = []
    for token in _FORMULA_TOKEN.finditer(rhs):
        if token.group("quoted"):
            named.append(token.group("quoted")[1:-1])
        elif token.group("name"):
            named.append(token.group("name"))
    columns = set(frame.columns)
    return [c for c in dict.fromkeys(named) if c in columns and c != response]


def _formula_design(model):
    """How a formula fit rebuilds its design for new rows, or None without one.

    Returns a function from rows to their design matrix, with the columns each
    term fills, keyed by the term's name. statsmodels up to 0.14 keeps the
    formula's patsy DesignInfo as ``data.design_info``. From 0.15 it keeps it as
    ``data.model_spec``, which holds a formulaic ModelSpec instead when the
    formula engine is set to formulaic. Looking for ``design_info`` alone
    skipped the grid check under 0.15 without a word. A statistic of one row
    can be NaN, as x.std() is, and either engine would drop such a row as
    missing, so nothing counts as missing here.
    """
    data = model.model.data
    spec = getattr(data, "model_spec", None)
    if spec is None:
        spec = getattr(data, "design_info", None)
    if spec is None:
        return None
    if hasattr(spec, "get_model_matrix"):  # formulaic's ModelSpec
        def build(rows):
            return spec.get_model_matrix(rows, na_action="ignore", output="numpy")

        terms = {str(term): columns for term, columns in spec.term_slices.items()}
    else:  # patsy's DesignInfo
        from patsy import NAAction, build_design_matrices

        keep_nan = NAAction(NA_types=[])

        def build(rows):
            return build_design_matrices([spec], rows, NA_action=keep_nan)[0]

        terms = {term.name(): columns for term, columns in spec.term_slices.items()}
    return build, terms


def _check_grid_terms(model, frame: pd.DataFrame, predictors: list[str]) -> None:
    """Refuse a formula term that a prediction grid would change.

    get_prediction() rebuilds every term from the grid's rows alone. The
    stateful transforms of patsy and formulaic, such as ``center()`` and
    ``bs()``, keep what they learnt from the fit. An expression such as
    ``I(x - x.mean())`` takes a statistic of whatever rows it is given, so over
    a grid it shifts. As in the R twin, each estimation row at a numeric
    predictor's minimum or maximum is rebuilt on its own and compared with the
    fit's design matrix, since such a statistic lies furthest from the row's own
    value there.
    """
    design = _formula_design(model)
    if design is None:
        return
    build, terms = design
    exog = np.asarray(model.model.exog, dtype=float)
    rows = {0}
    for col in predictors:
        if pd.api.types.is_numeric_dtype(frame[col]):
            values = frame[col].to_numpy(dtype=float, na_value=np.nan)
            if np.isfinite(values).any():
                rows.update((int(np.nanargmin(values)), int(np.nanargmax(values))))
    unsafe = []
    for i in sorted(rows):
        try:
            rebuilt = np.asarray(build(frame.iloc[[i]]), dtype=float)
        except Exception:  # each engine raises its own errors, and others, on a bad rebuild
            rebuilt = np.empty((0, exog.shape[1]))
        if rebuilt.shape != (1, exog.shape[1]):
            unsafe = None
            break
        same = np.isclose(rebuilt[0], exog[i], rtol=1e-9, atol=1e-12, equal_nan=True)
        for name, columns in terms.items():
            if not same[columns].all() and name not in unsafe:
                unsafe.append(name)
    if unsafe is None:
        what = "A term of this model cannot be computed from a single row of its data"
    elif len(unsafe) == 1:
        what = (f"The term `{unsafe[0]}` depends on statistics of the rows it is "
                "computed from")
    elif unsafe:
        what = ("The terms " + " and ".join(f"`{t}`" for t in unsafe)
                + " depend on statistics of the rows they are computed from")
    else:
        return
    several = bool(unsafe) and len(unsafe) > 1
    raise ValueError(
        f"{what}. get_prediction() computes every term afresh over a prediction "
        f"grid, so transform the {'variables' if several else 'variable'} in the "
        "data and refit the model."
    )


def _offset_arguments(model, n: int) -> dict:
    """Hold an offset or exposure at its mean over the estimation rows.

    statsmodels predicts new rows at an offset of zero unless one is passed. An
    offset expression cannot in general be inverted to a value of its variable,
    so the offset is held at its mean on the link scale, as in the R twin. The
    model stores an exposure as its log, so the exposure is held at its
    geometric mean.
    """
    arguments = {}
    offset = getattr(model.model, "offset", None)
    if offset is not None:
        arguments["offset"] = np.full(n, float(np.mean(offset)))
    exposure = getattr(model.model, "exposure", None)
    if exposure is not None:
        arguments["exposure"] = np.full(n, float(np.exp(np.mean(exposure))))
    return arguments


def _prediction_label(model) -> str:
    """The y-axis label, which names the convention for an offset model."""
    label = f"Predicted {_response_label(model)}"
    if _offset_arguments(model, 1):
        label += " at the mean offset"
    return label


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
    which is what a reader expects from a prediction plot. An offset or exposure
    is held at its mean (see ``_offset_arguments``).
    """
    prediction = model.get_prediction(grid, **_offset_arguments(model, len(grid)))
    summary = prediction.summary_frame(alpha=1 - conf_level)
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
    confidence band, both read from statsmodels' ``get_prediction``. The band
    takes t or z as the result's ``use_t`` says, so a GLM whose dispersion is
    estimated, fitted with ``use_t=True``, gets the t band of the R twin.

    The predictors are the columns the formula names, so a column of the source
    frame outside the model plays no part. The range, means and levels come from
    the rows the model was fitted to, without the rows its missing-value
    handling dropped. A formula term computed from statistics of its rows, such
    as ``I(x - x.mean())``, is refused with ``ValueError``, since
    ``get_prediction()`` would compute it afresh over the grid. patsy's
    ``center()`` and ``standardize()`` keep the values of the fit and are safe.
    A model fitted with an ``offset`` is predicted at the mean offset over the
    estimation rows, and one fitted with an ``exposure`` at the geometric mean
    exposure. The y-axis label then ends in "at the mean offset".

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS/GLM result from a formula and a DataFrame (so the source
        data is retained for building the grid).
    var : str
        The predictor to vary along the x-axis: a numeric column that the
        formula names.
    conf_level : float
        Confidence level for the band, a single number strictly between 0 and
        1.
    n : int
        Number of points across the predictor's range.
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
    >>> p = dp.effects_plot(model, "fertiliser")
    """
    _require_statsmodels()
    if not _is_fitted_model(model):
        raise TypeError("effects_plot needs a fitted statsmodels result.")
    _check_conf_level(conf_level)

    frame = _estimation_frame(model)
    predictors = _predictor_columns(model, frame)
    if var not in predictors:
        raise KeyError(f"{var!r} is not a predictor in the fitted model.")
    if not pd.api.types.is_numeric_dtype(frame[var]):
        raise TypeError(
            f"effects_plot varies {var!r} continuously, so it must be numeric. "
            "Use interaction_plot for a categorical predictor."
        )
    _check_grid_terms(model, frame, predictors)

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
        + labs(x=var, y=_prediction_label(model), title=title)
        + theme_depictr()
    )


def interaction_plot(model, x, group, conf_level: float = 0.95, n: int = 100,
                     band: bool = True, title=None):
    """Predicted response across ``x`` for each level of a categorical ``group``.

    One coloured line per group shows how the predicted response changes with
    ``x`` within that group, so a fan of non-parallel lines is the visible mark of
    an interaction. The remaining predictors are held at their mean (numeric) or
    reference level (categorical). As in :func:`effects_plot`, the predictors are
    the columns the formula names, the range, means and levels come from the
    rows the model was fitted to, a term computed from statistics of its rows is
    refused, and an offset or exposure is held at its mean.

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS/GLM result from a formula and a DataFrame.
    x : str
        The numeric predictor on the x-axis.
    group : str
        The categorical predictor whose levels are drawn as separate lines. A
        level that occurs only in rows the fit dropped is not drawn.
    conf_level : float
        Confidence level for the bands, a single number strictly between 0
        and 1.
    n : int
        Number of points across the x range, per group.
    band : bool
        Whether to draw a confidence ribbon behind each line.
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
    >>> model = smf.ols('Q("yield") ~ fertiliser * treatment', cy).fit()
    >>> p = dp.interaction_plot(model, "fertiliser", "treatment")
    """
    _require_statsmodels()
    if not _is_fitted_model(model):
        raise TypeError("interaction_plot needs a fitted statsmodels result.")
    _check_conf_level(conf_level)

    frame = _estimation_frame(model)
    predictors = _predictor_columns(model, frame)
    for name in (x, group):
        if name not in predictors:
            raise KeyError(f"{name!r} is not a predictor in the fitted model.")
    if not pd.api.types.is_numeric_dtype(frame[x]):
        raise TypeError(f"interaction_plot needs a numeric x; {x!r} is not.")
    _check_grid_terms(model, frame, predictors)

    others = [c for c in predictors if c not in (x, group)]
    held = _reference_row(frame, others)
    x_seq = np.linspace(frame[x].min(), frame[x].max(), n)
    # dropna before the levels are taken: the loop below coerces each level with
    # str(), which would turn a missing group value into a real "nan" line and
    # legend entry. Every other grouped function in the package drops first.
    levels = list(pd.unique(frame[group].dropna()))
    if not levels:
        raise ValueError(f"{group!r} has no non-missing values to group by.")

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
        + labs(x=x, y=_prediction_label(model),
               color=group, fill=group, title=title)
        + theme_depictr()
    )


def compare_models(models, intercept: bool = False, conf_level: float = 0.95,
                   title=None, reference_line=None, exponentiate: bool = False,
                   labels=None, interaction: str = "times"):
    """Dodged forest comparing each coefficient across several models.

    Each model contributes its estimates and confidence intervals (read with
    :func:`depictr.models.tidy_estimates`, so a fitted model or a tidy frame both
    work); the terms share a y-axis and the models are dodged apart and coloured,
    so a coefficient that shifts from one specification to the next is easy to
    spot.

    The terms are labelled as in :func:`depictr.coefficient_plot`: a factor
    level names its variable and level (``treatment: enhanced``), read from
    the design of each fitted model, and other names are tidied with
    :func:`depictr.format_terms`. Terms that take the same label share a row,
    so ``stress`` and a brms-style ``b_stress`` are drawn as one term, as in
    the R twin.

    Parameters
    ----------
    models : dict
        ``{name: fitted_model_or_tidy_frame}``. The keys label and colour the
        models.
    intercept : bool
        Whether to keep the intercept term.
    conf_level : float
        Confidence level passed to :func:`depictr.models.tidy_estimates`, a
        single number strictly between 0 and 1.
    title : str, optional
        Plot title.
    reference_line : float, optional
        Position of the dashed vertical reference line. ``None``, the default,
        puts it at 0, or at 1 when ``exponentiate`` is set. ``float("nan")``
        leaves it out.
    exponentiate : bool
        Whether to draw every model's exponentiated estimates and intervals on
        a log10 axis, as odds or rate ratios. A tidy frame among the models
        must hold estimates on the link scale, since it is exponentiated too.
        The axis is named as in :func:`depictr.coefficient_plot` when the
        fitted models agree on the ratio, and "exp(Estimate)" when they do
        not. Defaults to ``False``.
    labels : dict, list or "raw", optional
        Display labels for the terms. A dict maps raw term names to labels and
        leaves the other terms to the default labels, and a key that matches
        no term warns. A list gives one label per row, in the order the terms
        first appear across the models, which is the order they are drawn
        from the top, after the intercept is dropped unless
        ``intercept=True``. Terms drawn on one row take one label. ``"raw"``
        draws the names as the models give them.
    interaction : {"times", "asterisk", "colon", "space"}
        How an interaction is written in the default labels, passed to
        :func:`depictr.format_terms`.

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
    >>> reduced = smf.ols('Q("yield") ~ fertiliser', cy).fit()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> p = dp.compare_models({"Reduced": reduced, "Full": model})
    """
    from .models import (
        _check_exponentiate,
        _ratio_label,
        _reference_position,
        tidy_estimates,
    )

    if not isinstance(models, dict) or not models:
        raise ValueError("`models` must be a non-empty {name: model} dict.")
    _check_conf_level(conf_level)
    _check_exponentiate(exponentiate)
    interaction_separator(interaction)
    _check_labels(labels)
    line = _reference_position(reference_line, exponentiate)

    frames = []
    for name, model in models.items():
        est = tidy_estimates(model, conf_level=conf_level,
                             exponentiate=exponentiate).copy()
        est["model"] = str(name)
        frames.append(est)
    est = pd.concat(frames, ignore_index=True)
    _warn_unused_labels(est["term"], labels)

    # The ratio is named after the fitted models. Tidy frames carry no family,
    # and models of different families share only the neutral name.
    x_label = "Estimate"
    if exponentiate:
        named = {_ratio_label(m) for m in models.values()
                 if not isinstance(m, pd.DataFrame)}
        x_label = named.pop() if len(named) == 1 else "exp(Estimate)"

    # The factor labels of every fitted model, so a factor term that only a
    # later model holds is labelled too. Where two models hold the same
    # column, the first one's label is kept.
    auto = {}
    if not isinstance(labels, str):
        for model in models.values():
            for column, label in _design_labels(model, interaction).items():
                auto.setdefault(column, label)

    intercept_dropped = False
    if not intercept:
        is_intercept = est["term"].str.fullmatch(_INTERCEPT_PATTERN)
        intercept_dropped = bool(is_intercept.any())
        est = est[~is_intercept]
    if est.empty:
        raise ValueError("Nothing left to plot after dropping the intercept.")

    # Rows are aligned by label, as in the R twin, so terms of different models
    # that take the same label share one. A list gives one label per row, in
    # the order the terms first appear, and is matched to the default labels
    # in that order.
    positional = isinstance(labels, (list, tuple))
    drawn = _term_labels(est["term"], None if positional else labels, auto,
                         interaction)
    if positional:
        rows = list(dict.fromkeys(drawn))
        _check_label_count(labels, len(rows), "the models have",
                           "in the order they first appear", intercept_dropped)
        given = dict(zip(rows, (str(label) for label in labels), strict=True))
        drawn = [given[label] for label in drawn]
    est = est.assign(label=drawn)
    # Two terms of one model under one label would be drawn on top of each
    # other.
    for _, part in est.groupby("model", sort=False):
        _check_distinct(part["term"], part["label"])

    # First-seen order of terms and models. Terms run up the axis so the first
    # reads at the top, models keep their dict order in the legend.
    term_levels = list(dict.fromkeys(est["label"]))
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
    # strict=True: both are columns of the same frame, so unequal lengths cannot
    # happen and a silent truncation would misplace every later estimate.
    est["y"] = [base_y[t] + offset[m]
                for t, m in zip(est["label"], est["model"], strict=True)]
    # Listed from the bottom of the axis up, as coefficient_plot() lists them.
    est["label"] = pd.Categorical(est["label"], categories=term_levels[::-1],
                                  ordered=True)

    from plotnine import (
        element_blank,
        geom_errorbarh,
        geom_point,
        geom_vline,
        scale_x_log10,
        scale_y_continuous,
    )

    p = ggplot(est, aes(x="estimate", y="y", color="model"))
    if line is not None:
        p = p + geom_vline(xintercept=line, linetype="dashed", color="#9e9e9e")
    p = (
        p
        + geom_errorbarh(aes(xmin="conf_low", xmax="conf_high"),
                         height=0.0, size=0.7)
        + geom_point(size=2.4)
        + scale_colour_depictr(n_models, name="Model")
        + scale_y_continuous(
            breaks=[base_y[t] for t in term_levels],
            labels=term_levels,
            limits=(-0.5, len(term_levels) - 0.5),
        )
        + labs(x=x_label, y="", color="Model", title=title)
        + theme_depictr()
        + theme(panel_grid_major_y=element_blank())
    )
    return p + scale_x_log10() if exponentiate else p
