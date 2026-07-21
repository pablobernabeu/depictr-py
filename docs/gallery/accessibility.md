# Accessibility: a validated colourblind-safe palette

depictr defaults to the Okabe-Ito palette and checks that choice rather than
asserting it: a Machado-2009 simulator and a CIE-Lab distance test confirm the
colours stay distinguishable under each colour-vision deficiency.

```python exec="1" html="1" session="a11y"
import sys; sys.path.insert(0, "docs")

from _exec import pre, show
```

## The safety report

The report gives the worst-case minimum distance between any two palette colours,
across normal vision and each deficiency at full severity.

```python exec="1" html="1" source="material-block" session="a11y"
import depictr as dp

report = dp.palette_safety()
print(pre(repr(report)))
```

## The palette in use

A grouped distribution in the default palette, a histogram with its density
curve. `legend_inside` tucks the legend into the empty top-right corner instead
of a separate margin.

```python exec="1" html="1" source="material-block" session="a11y"
ld = dp.lexical_decision()
p = dp.explore_distribution(ld, "RT", group="condition", kind="both",
                            legend_inside=True)
print(show(p))
```

## The palette as a deficiency sees it

The same simulator drives the safety report above and the web app's
colour-vision toggle. Below it maps every palette colour through each
deficiency in turn, so the rows can be read against one another. The tightest
pair is the reddish purple and the grey under deuteranopia, which is where the
7.4 in the report comes from, and even that stays well above the threshold of 5.

```python exec="1" html="1" source="material-block" session="a11y"
import pandas as pd
from plotnine import (aes, element_blank, geom_tile, ggplot, labs,
                      scale_fill_identity, scale_y_discrete, theme)

palette = dp.depictr_palette()
conditions = ["normal", *dp.cvd.DEFICIENCIES]

rows = []
for condition in conditions:
    # simulate_cvd takes the three deficiencies only; normal vision is the
    # palette itself, which is the same split palette_safety makes internally.
    seen = palette if condition == "normal" else dp.simulate_cvd(palette, condition)
    rows += [{"condition": condition, "position": i, "colour": colour}
             for i, colour in enumerate(seen)]

swatches = pd.DataFrame(rows)
p = (ggplot(swatches, aes("factor(position)", "condition", fill="colour"))
     + geom_tile(width=0.92, height=0.82)
     + scale_fill_identity()
     # Discrete scales are drawn from the bottom up, so the order is reversed
     # to put normal vision at the top.
     + scale_y_discrete(limits=list(reversed(conditions)))
     + labs(x="Palette colour", y=None)
     + theme(axis_text_x=element_blank(), panel_grid=element_blank()))
print(show(p, width=8, height=2.6))
```

The hex codes behind those rows, for the deficiency the palette comes closest
to failing.

```python exec="1" html="1" source="material-block" session="a11y"
simulated = dp.simulate_cvd(palette, "deutan")
lines = [f"{original}  ->  {seen}"
         for original, seen in zip(palette, simulated)]
print(pre("\n".join(lines)))
```
