# Accessibility: a validated colourblind-safe palette

depictr defaults to the Okabe-Ito palette and checks that choice rather than
asserting it. A Machado-2009 simulator and a CIE-Lab distance test confirm the
colours stay distinguishable under each colour-vision deficiency. A safe palette
is not a safe figure, though, so the same machinery also audits a finished plot.
That is the second half of this page.

```python exec="1" html="1" session="a11y"
import sys; sys.path.insert(0, "docs")

from _exec import pre, show, table
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
    seen = palette if condition == "normal" else dp.simulate_cvd(
        palette, condition
    )
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

## Where the guarantee stops

Eight colours is the guarantee, not a starting point. Asked for more,
`depictr_palette` interpolates between the Okabe-Ito colours, and the
in-between colours land close enough together that the same check fails them.
The package warns when that happens rather than handing back colours under a
promise it cannot keep, so the honest reading of the report below is that nine
groups is already the ceiling and ten is past it.

```python exec="1" html="1" source="material-block" session="a11y"
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the warning is the point; here we tabulate it
    sizes = {n: dp.palette_safety(dp.depictr_palette(n)) for n in (8, 9, 10, 12)}

lines = [f"n = {n:>2}   min Delta-E {r['min_delta_e']:>5}   safe: {r['safe']}"
         for n, r in sizes.items()]
print(pre("\n".join(lines)))
```

With more than eight groups, facet them, or map the variable to the sequential
ramp (`dp.depictr_palette(n, kind="sequential")`), which stays ordered and
legible at any length because lightness, not hue, does the work.

## Auditing the figure you are about to submit

The palette report above describes eight colours in the abstract. It cannot know
how many of them a given figure uses, what you replaced them with, how small the
text becomes once the figure is squeezed into a journal column, or whether the
only thing separating two groups is their colour. `check_figure` reads the built
plot and reports what it measured beside the threshold it was measured against.

```python exec="1" html="1" source="material-block" session="a11y"
p = dp.explore_distribution(ld, "RT", group="condition", kind="both",
                            legend_inside=True)
print(table(dp.check_figure(p)))
```

Two rows are worth dwelling on. `geometry_contrast` measures each encoding colour
against the panel background, and the palette's orange sits at 2.25 against white,
below the 3:1 that WCAG asks of a graphical object. `redundant_encoding` is zero
because nothing but colour tells the two conditions apart. The audit also takes a
stated output width, which is where most figure text quietly fails: text is drawn
in points, so a figure saved seven inches wide and then printed in an 8.9 cm
column arrives at half the size it looked on screen.

```python exec="1" html="1" source="material-block" session="a11y"
column = dp.check_figure(p, width_cm=8.9)
print(table(column[column["check"] == "text_size"]))
```

A figure built with two well-separated colours, a redundant shape and no
shrinking clears every row, so the audit is capable of both verdicts rather than
only of finding fault.

```python exec="1" html="1" source="material-block" session="a11y"
from plotnine import aes, geom_point, ggplot, scale_colour_manual

cy = dp.crop_yield()
mended = (ggplot(cy, aes("rainfall", "yield", colour="treatment",
                         shape="treatment"))
          + geom_point(alpha=0.7)
          + scale_colour_manual(values=["#005b96", "#d55e00"])
          + dp.theme_depictr())
print(table(dp.check_figure(mended)[["check", "measured", "threshold",
                                     "verdict"]]))
```

## Where the guarantee stops a second time: greyscale

One limitation belongs here, where the claim is made. The eight-colour palette
clears every colour-vision check and fails the greyscale check: its orange and its
sky blue differ by 0.79 in CIE lightness, so a black-and-white printer renders
them as the same grey. The Okabe-Ito guarantee is about hue confusion and was
never a claim about greyscale. The threshold has been left where it is rather than
moved so that depictr's own defaults pass.

```python exec="1" html="1" source="material-block" session="a11y"
eight = pd.DataFrame({"g": list("abcdefgh"), "x": range(8), "y": range(8)})
p8 = (ggplot(eight, aes("x", "y", colour="g")) + geom_point()
      + dp.scale_colour_depictr() + dp.theme_depictr())
report = dp.check_figure(p8)
print(table(report[report["check"].str.contains("separability")]))
```

A figure that may be printed in black and white wants fewer groups, a sequential
palette, or a redundant shape or line type. Four groups are not enough on their
own either: the bluish green and the vermillion differ by 3.55.
