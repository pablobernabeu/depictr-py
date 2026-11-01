"""Multi-panel regression reports for fitted statsmodels results.

A single diagnostic plot answers one question; a regression check usually means
looking at several at once. These functions gather the everyday diagnostics into
one composed figure so the whole picture sits on a single page in the depictr
visual language. The four-panel residual report mirrors the set R draws from
``plot.lm`` (Fox & Weisberg, 2019), and the dashboard pairs the coefficient plot
with the two panels people read first. As in the R twin, a GLM, Logit or Probit
fit gets panels made for it. They plot randomised quantile residuals (Dunn &
Smyth, 1996) and, for a binomial model, binned residuals (Gelman & Hill, 2007).

The panels reuse the single-plot diagnostics in :mod:`depictr.diagnostics` and
the forest plot in :mod:`depictr.models`, and join them with
:func:`depictr.compose.arrange_plots`.

``statsmodels`` arrives with the core install (plotnine itself requires it);
the ``depictr[models]`` extra remains to pin the tested floor.

References
----------
Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals. Journal of
Computational and Graphical Statistics, 5(3), 236-244.
https://doi.org/10.1080/10618600.1996.10474708

Fox, J., & Weisberg, S. (2019). An R companion to applied regression (3rd ed.).
Sage.

Gelman, A., & Hill, J. (2007). Data analysis using regression and
multilevel/hierarchical models. Cambridge University Press.
https://doi.org/10.1017/CBO9780511790942
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
    _fitted_values,
    _is_binary,
    _is_binomial,
    _is_fitted_model,
    _is_glm,
    _quantile_residuals,
    _raw_residuals,
    _require_statsmodels,
    _studentised_residuals,
    binned_residual_plot,
    qq_plot,
)
from .models import coefficient_plot
from .palette import BRAND, REFERENCE
from .theme import theme_depictr


def _check_model(model, func_name: str) -> None:
    """Confirm the argument is a fitted statsmodels result, or explain why not."""
    _require_statsmodels()
    if not _is_fitted_model(model):
        raise TypeError(f"{func_name} needs a fitted statsmodels result.")


def _residuals_vs_fitted(model, title=None):
    """Residuals against fitted values, with a lowess smoother and a zero line.

    A flat smoother near zero means the linear fit has caught the structure; a
    bend or funnel points to a missing term or non-constant variance. A GLM,
    Logit or Probit fit plots its deviance residuals against its fitted means.
    """
    resid, label = _raw_residuals(model)
    df = pd.DataFrame({"fitted": _fitted_values(model), "resid": resid})
    return (
        ggplot(df, aes("fitted", "resid"))
        + geom_hline(yintercept=0, linetype="dashed", color="#9e9e9e")
        + geom_point(color=BRAND, size=2, alpha=0.6)
        + geom_smooth(method="lowess", se=False, color=REFERENCE, size=0.9)
        + labs(x="Fitted values", y=label, title=title)
        + theme_depictr()
    )


def _scale_location(model, title=None):
    """Scale-location plot: sqrt of the absolute studentised residuals vs fitted.

    Taking the square root of the absolute residual flattens the cloud, so a
    rising or falling smoother reads as a change in spread (heteroscedasticity)
    rather than in level.
    """
    fitted = _fitted_values(model)
    root_abs = np.sqrt(np.abs(_studentised_residuals(model)))
    df = pd.DataFrame({"fitted": fitted, "root_abs": root_abs})
    return (
        ggplot(df, aes("fitted", "root_abs"))
        + geom_point(color=BRAND, size=2, alpha=0.6)
        + geom_smooth(method="lowess", se=False, color=REFERENCE, size=0.9)
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
        + geom_smooth(method="lowess", se=False, color=REFERENCE, size=0.9)
        + labs(x="Leverage (hat value)", y="Studentised residuals", title=title)
        + theme_depictr()
    )


def _quantile_qq(model, seed=None):
    """Normal Q-Q plot of a GLM's randomised quantile residuals.

    A family with no quantile residual plots its deviance residuals, and the
    axis says so. An observation with zero weight has no residual and is left
    out.
    """
    resid, is_quantile = _quantile_residuals(model, seed=seed)
    label = "Quantile residuals" if is_quantile else "Deviance residuals"
    return (qq_plot(resid[np.isfinite(resid)], title="Normal Q-Q")
            + labs(y=label))


def _residual_and_qq_panels(model, glm_panels, seed):
    """The residual panel and the Q-Q panel for ``model``.

    A GLM, Logit or Probit fit gets the GLM panels unless ``glm_panels`` is
    false, as in the R twin. They are a Q-Q plot of its quantile residuals and
    a residual panel: binned residuals for a binomial model, and the deviance
    residuals against the fitted values for other families. The R twin bins
    those too, but :func:`depictr.binned_residual_plot` takes binomial models
    only. Any other model gets the classic panels.
    """
    if glm_panels and (_is_glm(model) or _is_binary(model)):
        if _is_binomial(model):
            resid_panel = binned_residual_plot(model, title="Binned residuals")
        else:
            resid_panel = _residuals_vs_fitted(model, title="Residuals vs fitted")
        return resid_panel, _quantile_qq(model, seed=seed)
    return (_residuals_vs_fitted(model, title="Residuals vs fitted"),
            qq_plot(model, title="Normal Q-Q"))


def residual_diagnostics_plot(model, title=None, glm_panels=True, seed=None):
    """Four-panel residual report for a fitted OLS, GLM, Logit or Probit model.

    Composes the four checks R draws from ``plot.lm`` into a 2x2 grid: residuals
    against fitted values, the scale-location plot, a normal quantile-quantile
    plot of the residuals, and residuals against leverage. Read together they
    cover the usual assumptions of a linear fit: the right functional form,
    constant variance, approximately normal residuals, and no single point
    dominating the fit.

    A GLM, Logit or Probit fit gets panels made for it (``glm_panels=True``), as
    in the R twin. The Q-Q panel plots randomised quantile residuals (Dunn &
    Smyth, 1996), which are standard normal under a correctly specified model
    whatever the family. The studentised residuals of a binary or count model
    are not expected to be normal, even when the model is right. The quantile
    residuals of a binomial or Poisson fit are randomised, so a departure worth
    acting on is one that persists when the panel is drawn again with another
    ``seed``. A binomial model's residual panel is
    [`depictr.binned_residual_plot`][] (Gelman & Hill, 2007). Other families
    plot their deviance residuals against the fitted values. The R twin bins
    them too, but the binned plot here takes binomial models only. A family
    other than the binomial, Poisson, Gaussian and Gamma has no quantile
    residual here. Its Q-Q panel plots the deviance residuals, and the axis
    says so. A Logit or Probit fit is plotted against its fitted probabilities,
    as the binomial GLM of the same model is.

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS, GLM, Logit or Probit result, such as
        ``smf.ols(...).fit()``.
    title : str, optional
        Accepted for API symmetry, but dropped with a warning: plotnine
        compositions cannot carry a figure-level title, so the grid keeps its
        per-panel titles instead.
    glm_panels : bool, default True
        For a GLM, Logit or Probit fit, whether to draw the quantile-residual
        and binned-residual panels. ``False`` draws the classic ones, which plot
        its deviance and studentised residuals. Ignored for any other model.
    seed : int, optional
        Seed for the randomisation of the quantile residuals of a binomial or
        Poisson fit, for a reproducible Q-Q panel. numpy's global random state
        is left alone.

    Returns
    -------
    plotnine.composition.Compose
        A 2x2 composition with ``.draw`` and ``.save``.

    References
    ----------
    Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals.
    Journal of Computational and Graphical Statistics, 5(3), 236-244.
    https://doi.org/10.1080/10618600.1996.10474708

    Fox, J., & Weisberg, S. (2019). An R companion to applied regression
    (3rd ed.). Sage.

    Gelman, A., & Hill, J. (2007). Data analysis using regression and
    multilevel/hierarchical models. Cambridge University Press.
    https://doi.org/10.1017/CBO9780511790942

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> p = dp.residual_diagnostics_plot(model)

    A logistic GLM gets binned residuals and a Q-Q plot of quantile residuals,
    whose randomisation the seed fixes:

    >>> import statsmodels.api as sm
    >>> ct = dp.clinical_trial()
    >>> glm = smf.glm("adverse_event ~ biomarker + age + arm", ct,
    ...               family=sm.families.Binomial()).fit()
    >>> p = dp.residual_diagnostics_plot(glm, seed=1)
    """
    _check_model(model, "residual_diagnostics_plot")
    resid_panel, qq = _residual_and_qq_panels(model, glm_panels, seed)
    return arrange_plots(
        resid_panel,
        _scale_location(model, title="Scale-location"),
        qq,
        _residuals_vs_leverage(model, title="Residuals vs leverage"),
        ncol=2,
        title=title,
        # Untagged, as in the R twin, which composes these panels with
        # patchwork directly.
        tag_levels=None,
    )


def model_report(model, title=None, glm_panels=True, seed=None):
    """One-figure dashboard pairing the coefficient plot with key diagnostics.

    Puts the three plots a reader usually wants first side by side: the
    coefficient (forest) plot of the estimates and their confidence intervals,
    a residual panel, and the normal quantile-quantile plot. It is a quick "is
    the model worth trusting, and what does it say" view, with the full set of
    checks left to [`depictr.residual_diagnostics_plot`][]. A GLM, Logit or
    Probit fit gets the panels that function draws for it: a Q-Q plot of
    randomised quantile residuals and, for a binomial model, binned residuals
    in place of the scatter.

    Parameters
    ----------
    model : statsmodels results object
        A fitted OLS, GLM, Logit or Probit result exposing ``params`` and
        ``conf_int`` (for the coefficient plot) as well as fitted values and
        residuals.
    title : str, optional
        Accepted for API symmetry, but dropped with a warning: plotnine
        compositions cannot carry a figure-level title, so the dashboard keeps
        its per-panel titles instead.
    glm_panels : bool, default True
        For a GLM, Logit or Probit fit, whether to draw the quantile-residual
        and binned-residual panels. ``False`` draws the classic ones. Ignored
        for any other model.
    seed : int, optional
        Seed for the randomisation of the quantile residuals of a binomial or
        Poisson fit, for a reproducible Q-Q panel. numpy's global random state
        is left alone.

    Returns
    -------
    plotnine.composition.Compose
        A composition with ``.draw`` and ``.save``, its panels tagged A, B and
        C, as the R twin tags its own.

    Examples
    --------
    >>> import depictr as dp
    >>> import statsmodels.formula.api as smf
    >>> cy = dp.crop_yield()
    >>> model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph', cy).fit()
    >>> p = dp.model_report(model)
    >>> ct = dp.clinical_trial()
    >>> logit = smf.logit("adverse_event ~ biomarker + age + arm", ct).fit(disp=0)
    >>> p = dp.model_report(logit, seed=1)
    """
    _check_model(model, "model_report")
    resid_panel, qq = _residual_and_qq_panels(model, glm_panels, seed)
    return arrange_plots(
        coefficient_plot(model, title="Coefficients"),
        resid_panel,
        qq,
        ncol=3,
        title=title,
        tag_levels="A",
    )
