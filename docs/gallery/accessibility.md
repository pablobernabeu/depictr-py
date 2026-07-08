# Accessibility: a validated colourblind-safe palette

depictr defaults to the Okabe-Ito palette and checks that choice rather than
asserting it: a Machado-2009 simulator and a CIE-Lab distance test confirm the
colours stay distinguishable under each colour-vision deficiency.

```python exec="1" session="a11y"
import io, base64, warnings

import matplotlib

matplotlib.use("Agg")  # render headlessly; no GUI backend during the build

from plotnine.exceptions import PlotnineWarning

warnings.filterwarnings("ignore", category=PlotnineWarning)


def show(p, width=8, height=5):
    buf = io.BytesIO()
    p.save(buf, format="png", width=width, height=height, dpi=110, verbose=False)
    print('<img alt="depictr plot" src="data:image/png;base64,'
          + base64.b64encode(buf.getvalue()).decode() + '">')
```

## The safety report

The report gives the worst-case minimum distance between any two palette colours,
across normal vision and each deficiency at full severity.

```python exec="1" source="material-block" session="a11y"
import depictr as dp

report = dp.palette_safety()
print("<pre>" + repr(report) + "</pre>")
```

## The palette in use

A grouped distribution in the default palette, a histogram with its density
curve. `legend_inside` tucks the legend into the empty top-right corner instead
of a separate margin.

```python exec="1" source="material-block" session="a11y"
ld = dp.lexical_decision()
p = dp.explore_distribution(ld, "RT", group="condition", kind="both",
                            legend_inside=True)
show(p)
```
