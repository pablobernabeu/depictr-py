# Model estimates and diagnostics

Fit a model once, then read it from several angles: a coefficient forest, the
predicted effect of a predictor, an estimation plot of a group difference, and a
panel of regression diagnostics.

```python exec="1" html="1" session="models"
import sys; sys.path.insert(0, "docs")

from _exec import show
```

```python exec="1" html="1" source="material-block" session="models"
import statsmodels.formula.api as smf

import depictr as dp

cy = dp.crop_yield()
# Q() quotes "yield" because it is a Python keyword.
model = smf.ols('Q("yield") ~ fertiliser + rainfall + soil_ph + treatment', cy).fit()
```

## Coefficient forest

A dot-and-whisker forest of the coefficients.

```python exec="1" html="1" source="material-block" session="models"
p = dp.coefficient_plot(model, title="Drivers of crop yield")
print(show(p))
```

## Predicted effect

The predicted response as one predictor varies, with a confidence band.

```python exec="1" html="1" source="material-block" session="models"
p = dp.effects_plot(model, "fertiliser")
print(show(p))
```

## Interaction

One line per treatment level, showing how the predicted fertiliser slope differs
between them. The diverging lines are the fertiliser-by-treatment interaction.

```python exec="1" html="1" source="material-block" session="models"
m_int = smf.ols('Q("yield") ~ fertiliser * treatment + rainfall', cy).fit()
p = dp.interaction_plot(m_int, "fertiliser", "treatment")
print(show(p))
```

## Comparing models

A reduced and the full model side by side, so a coefficient that moves across
specifications stands out. The dict keys label and colour the models.

```python exec="1" html="1" source="material-block" session="models"
reduced = smf.ols('Q("yield") ~ fertiliser + rainfall', cy).fit()
p = dp.compare_models({"Reduced": reduced, "Full": model})
print(show(p))
```

## Estimation plot

A two-panel Gardner-Altman estimation plot of the group difference.

```python exec="1" html="1" source="material-block" session="models"
p = dp.estimation_plot(cy, "yield", "treatment", two_panel=True, seed=1)
print(show(p, width=9))
```

## Residual diagnostics

The four-panel residual-diagnostic dashboard.

```python exec="1" html="1" source="material-block" session="models"
p = dp.residual_diagnostics_plot(model)
print(show(p, width=9, height=7))
```

## Multicollinearity

A variance-inflation-factor bar per predictor, with a reference line at the usual
threshold of 5. Bars past it flag predictors whose variance is inflated by
collinearity.

```python exec="1" html="1" source="material-block" session="models"
p = dp.vif_plot(model)
print(show(p))
```

## Binned residuals for a logistic model

Raw residuals from a logistic fit take few values, so the mean residual per
fitted-value bin against a two-standard-error band is the readable check.

```python exec="1" html="1" source="material-block" session="models"
import statsmodels.api as sm

ct = dp.clinical_trial()
glm = smf.glm("adverse_event ~ biomarker + age", ct,
              family=sm.families.Binomial()).fit()
p = dp.binned_residual_plot(glm)
print(show(p))
```

## Random effects

A caterpillar plot of the conditional modes from a mixed model. Each point is one
region's predicted departure from the population fit, sorted against a zero
reference line.

```python exec="1" html="1" source="material-block" session="models"
wb = dp.wellbeing_survey().dropna(subset=["life_satisfaction", "sleep_hours"])
mm = smf.mixedlm("life_satisfaction ~ stress + sleep_hours", wb,
                 groups=wb["region"]).fit()
p = dp.random_effects_plot(mm)
print(show(p))
```

## Posteriors and the frequentist overlay

A Bayesian fit or a bootstrap gives a sample of draws per term rather than a
single interval. Here the draws are simulated around the fitted estimates to
stand in for a posterior.

```python exec="1" html="1" source="material-block" session="models"
import numpy as np

rng = np.random.default_rng(0)
terms = [t for t in model.params.index if t != "Intercept"]
draws = {t: rng.normal(model.params[t], model.bse[t], 2000) for t in terms}
p = dp.posterior_plot(draws)
print(show(p))
```

The frequentist estimate and the posterior share a row and are told apart by
colour, so only the posterior shows the full shape.

```python exec="1" html="1" source="material-block" session="models"
p = dp.frequentist_bayesian_plot(model, draws)
print(show(p))
```

## Power curve

A power curve read from a table you have already computed, here three effect
sizes from a two-sample t-test power calculation. depictr does no power
calculation of its own; statsmodels supplies the table and depictr draws it. The
dashed reference line at 0.8 marks the conventional target, and the sample size
at which each curve crosses it is the number needed per group to detect that
effect.

```python exec="1" html="1" source="material-block" session="models"
import pandas as pd
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()
rows = []
for label, d in [("small (d = 0.2)", 0.2),
                 ("medium (d = 0.5)", 0.5),
                 ("large (d = 0.8)", 0.8)]:
    for n in range(25, 425, 25):
        rows.append({"n": n, "effect": label,
                     "power": analysis.power(effect_size=d, nobs1=n,
                                             ratio=1.0, alpha=0.05)})
power = pd.DataFrame(rows)
p = dp.power_curve_plot(power, n="n", power="power", group="effect")
print(show(p))
```

## A one-figure model report

The coefficient forest with the residuals-versus-fitted and normal
quantile-quantile panels, composed into a single figure for a quick review.

```python exec="1" html="1" source="material-block" session="models"
p = dp.model_report(model)
print(show(p, width=9, height=8))
```

## Arranging plots

Any plots can be placed in one figure with `arrange_plots`, the building block
behind the composed reports above.

```python exec="1" html="1" source="material-block" session="models"
p = dp.arrange_plots(dp.qq_plot(model), dp.influence_plot(model), ncol=2)
print(show(p, width=9))
```
