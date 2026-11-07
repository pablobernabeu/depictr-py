# Accessibility: a validated colourblind-safe palette

depictr defaults to a palette based on the Okabe-Ito set. A colour-vision
simulator
([Machado, Oliveira and Fernandes, 2009](https://doi.org/10.1109/TVCG.2009.113))
and a CIEDE2000 colour-difference test confirm the colours stay distinguishable
under each colour-vision deficiency. A safe palette does not guarantee a safe
figure, so the same tools also audit a finished plot. The second half of this
page covers that audit.

```python exec="1" html="1" session="a11y"
import sys; sys.path.insert(0, "docs")

from _exec import image, pre, show, table
```

## The safety report

The report gives the worst-case minimum distance between any two palette colours,
across normal vision and each deficiency at full severity. The distance is the
CIEDE2000 colour difference
([Sharma, Wu and Dalal, 2005](https://doi.org/10.1002/col.20070)).

```python exec="1" html="1" source="material-block" session="a11y"
import depictr as dp

report = dp.palette_safety()
print(pre(repr(report)))
```

## The palette in use

A grouped distribution in the default palette, a histogram with its density
curve. `legend_inside` puts the legend in the empty top-right corner, so the
figure needs no separate margin.

```python exec="1" html="1" source="material-block" session="a11y"
ld = dp.lexical_decision()
p = dp.explore_distribution(ld, "RT", group="condition", kind="both",
                            legend_inside=True)
print(show(p))
```

## The palette as a deficiency sees it

The safety report above and the web app's colour-vision toggle use the same
simulator. The figure below maps every palette colour through each deficiency
in turn, so the rows can be read against one another. The closest pair is the
reddish purple and the grey under deuteranopia, whose distance of 6.43 in the
report still clears the threshold of 5.

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

## A finished figure as a deficiency sees it

`simulate_figure` applies the same simulation to a whole figure. It draws the
plot and passes every pixel through the transform, so the legend and the text
change with the data. The result is an array of pixels that matplotlib's
`imshow` or Streamlit's `st.image` can show. The web app draws its
colour-vision toggle this way. Below, the grouped distribution from the top of
the page appears under normal vision, which is severity zero, and then under
each deficiency at full severity.

```python exec="1" html="1" source="material-block" session="a11y"
import numpy as np

p = dp.explore_distribution(ld, "RT", group="condition", kind="both",
                            legend_inside=True)
views = [dp.simulate_figure(p, "deutan", severity=0, dpi=60)]
views += [dp.simulate_figure(p, deficiency, dpi=60)
          for deficiency in dp.cvd.DEFICIENCIES]
grid = np.concatenate([np.concatenate(views[:2], axis=1),
                       np.concatenate(views[2:], axis=1)])
print(image(grid, alt="The grouped distribution of reaction times four "
            "times over: under normal vision, then as readers with "
            "protanopia, deuteranopia and tritanopia see it."))
```

The R twin's `simulate_figure()` applies the same matrices to the figure's
vector elements and returns a gtable. A solid region comes out in the same
colour in both, to within the rounding of an eight-bit channel.

## Where the guarantee stops

The guarantee covers eight colours. When more are requested, `depictr_palette`
interpolates between its eight colours, and the interpolated colours lie close
enough together that the same check fails them. The package warns when that
happens, and the check in the report below agrees with the warning. Eight
colours pass, and nine already fail.

```python exec="1" html="1" source="material-block" session="a11y"
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the warning is the point; here we tabulate it
    sizes = {n: dp.palette_safety(dp.depictr_palette(n)) for n in (8, 9, 10, 12)}

lines = [f"n = {n:>2}   min Delta-E {r['min_delta_e']:>5}   safe: {r['safe']}"
         for n, r in sizes.items()]
print(pre("\n".join(lines)))
```

With more than eight groups, facet them or map the variable to the sequential
ramp, `dp.depictr_palette(n, kind="sequential")`. The ramp stays ordered and
legible at any length because lightness carries the ordering.

## Auditing the figure you are about to submit

The palette report above describes eight colours in the abstract. It cannot
show how many of them a given figure uses, what you replaced them with, how
small the text becomes once the figure is shrunk to fit a journal column, or
whether the only thing separating two groups is their colour. `check_figure`
reads the built plot and reports each value it measured next to the threshold
it was measured against.

```python exec="1" html="1" source="material-block" session="a11y"
p = dp.explore_distribution(ld, "RT", group="condition", kind="both",
                            legend_inside=True)
print(table(dp.check_figure(p)))
```

`geometry_contrast` measures each encoding colour, as it is drawn, against the
panel background. Even when opaque, the palette's orange falls below the 3:1
that WCAG asks of a graphical object, at 2.25:1, and it falls further when
drawn translucent, as the histogram's bars are. `redundant_encoding` is zero
because nothing but colour tells the two conditions apart.

The audit also takes a stated output width, which is where most figure text
fails without being noticed. Text is drawn in points, so the text of a figure
saved seven inches wide and then printed in an 8.9 cm column is half the size
it was on screen.

```python exec="1" html="1" source="material-block" session="a11y"
column = dp.check_figure(p, width_cm=8.9)
print(table(column[column["check"] == "text_size"]))
```

A figure saved at the size it will be printed is drawn at that size, so give
the saved width as both `width_cm` and `render_width_cm`, or leave both at
their defaults. Text drawn inside the panel, such as an annotation, a label or
a cell value, has a row of its own, `annotation_text_size`, held to the same
floor. That floor defaults to 6 pt, which is lenient. As their guidelines stood
on 30 September 2026, [PLOS](https://journals.plos.org/plosone/s/figures) and
[APA Style](https://apastyle.apa.org/style-grammar-guidelines/tables-figures/figures)
asked for at least 8 pt,
[Elsevier](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing)
for 7 pt and
[Nature](https://www.nature.com/documents/nature-final-artwork.pdf) for 5 to
7 pt.

Nature's range has a ceiling as well as a floor, since no text but the panel
labels may exceed 7 pt. Passing `max_text_pt=7` holds a figure to it in a row
of its own, `text_size_max`. `dp.save_plot(check=True)` runs the audit as it
saves, at the width it saves at, and warns about each check the figure fails.

The audit returns a pass as readily as a failure. A figure built with two
well-separated colours drawn opaque, a redundant shape and no shrinking passes
every row it can be held to. It draws no text inside the panel, so
`annotation_text_size` is not applicable.

```python exec="1" html="1" source="material-block" session="a11y"
from plotnine import aes, geom_point, ggplot, scale_colour_manual

cy = dp.crop_yield()
mended = (ggplot(cy, aes("rainfall", "yield", colour="treatment",
                         shape="treatment"))
          + geom_point()
          + scale_colour_manual(values=["#005b96", "#d55e00"])
          + dp.theme_depictr())
print(table(dp.check_figure(mended)[["check", "measured", "threshold",
                                     "verdict"]]))
```

The vermillion need not be typed by hand.
`dp.scale_colour_depictr(order="contrast")` draws the groups in the
contrast-first order of `dp.depictr_palette()`. It starts with the brand blue,
the vermillion and the reddish purple, the only three of the eight colours that
pass every colour row together. The bluish green comes fourth, and with it
four groups still keep 3:1 against white. In this order, the second group
takes the vermillion, which is also `dp.depictr_accent()`. The R twin can make
the order the default through its options. The Python package has no such
option, so add the scale, or `dp.scale_fill_depictr(order="contrast")`, to
each plot that should use it.

The audit does not measure a continuous colour or fill scale, such as a
heatmap's. Its colour rows then report 'not applicable' and give the reason,
and the cell values are measured as text inside the panel. A ramp is read by
its lightness, so prefer one whose lightness changes evenly, such as viridis
or depictr's sequential and diverging ramps, over a rainbow
([Crameri, Shephard and Heron, 2020](https://doi.org/10.1038/s41467-020-19160-7)).

```python exec="1" html="1" source="material-block" session="a11y"
heatmap = dp.check_figure(dp.correlation_heatmap(dp.wellbeing_survey()))
print(table(heatmap[heatmap["check"].isin(
    ["colour_separability", "text_size", "annotation_text_size"])]))
```

## Where the guarantee stops a second time: greyscale

The eight-colour palette passes every colour-vision check and fails the
greyscale check. Its orange and its sky blue differ by 0.79 in CIE lightness,
so a black-and-white printer renders them as the same grey. The Okabe-Ito
guarantee covers hue confusion alone. The check holds depictr's own palette to
the same threshold as any other.

```python exec="1" html="1" source="material-block" session="a11y"
eight = pd.DataFrame({"g": list("abcdefgh"), "x": range(8), "y": range(8)})
p8 = (ggplot(eight, aes("x", "y", colour="g")) + geom_point()
      + dp.scale_colour_depictr() + dp.theme_depictr())
report = dp.check_figure(p8)
print(table(report[report["check"].str.contains("separability")]))
```

A figure that may be printed in black and white should use fewer groups or a
sequential palette, or add a redundant shape or line type. Reducing the figure
to four groups is not enough on its own, because the bluish green and the
vermillion differ by 3.55.
