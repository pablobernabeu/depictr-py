# Exploring data

A first look at a dataset, all in one theme: distributions, categories,
correlations, cumulative distributions and a missing-data map. Every figure below
is produced by running the code shown.

```python exec="1" session="eda"
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

```python exec="1" source="material-block" session="eda"
import depictr as dp

wb = dp.wellbeing_survey()
ld = dp.lexical_decision()
```

## Cumulative distribution by group

An empirical cumulative distribution, with the legend tucked into the
bottom-right the curves leave empty once they saturate.

```python exec="1" source="material-block" session="eda"
p = dp.ecdf_plot(ld, "RT", group="condition", legend_inside=True)
show(p)
```

## A categorical comparison

```python exec="1" source="material-block" session="eda"
p = dp.explore_categorical(wb, "education", group="region")
show(p)
```

## Two variables, any types

`explore_bivariate` chooses the plot from the column types, here a numeric
response across a category.

```python exec="1" source="material-block" session="eda"
p = dp.explore_bivariate(ld, "condition", "RT")
show(p)
```

For two numeric columns it draws a scatter with a fitted trend, and a group
splits the trend line.

```python exec="1" source="material-block" session="eda"
cy = dp.crop_yield()
p = dp.scatter_trend(cy, "fertiliser", "yield", group="treatment")
show(p)
```

## Correlation heatmap

```python exec="1" source="material-block" session="eda"
p = dp.correlation_heatmap(wb)
show(p)
```

## Scatter-plot matrix

The pairwise relationships the heatmap condenses, drawn in full.

```python exec="1" source="material-block" session="eda"
p = dp.explore_pairs(cy, cols=["rainfall", "fertiliser", "soil_ph", "yield"])
show(p, width=8, height=7)
```

## Raincloud

Density, box and raw points together.

```python exec="1" source="material-block" session="eda"
p = dp.raincloud_plot(ld, "RT", group="condition")
show(p)
```

## Missing-data map

Columns are ordered most- to least-missing, so the legend sits inside the
top-right over the most-complete columns.

```python exec="1" source="material-block" session="eda"
p = dp.missingness_map(wb, legend_inside=True)
show(p)
```

## Group means over the raw data

Each group's mean with a confidence interval, drawn over its jittered
observations.

```python exec="1" source="material-block" session="eda"
p = dp.group_comparison_plot(ld, "RT", "condition")
show(p)
```

## Ridgeline

Stacked densities, one ridge per region, ordered by median.

```python exec="1" source="material-block" session="eda"
p = dp.ridgeline_plot(wb, "life_satisfaction", "region")
show(p)
```

## Dumbbell

Two age groups across the regions, each pair joined by a segment whose length is
the gap between them.

```python exec="1" source="material-block" session="eda"
import numpy as np

wb_age = wb.assign(age_group=np.where(wb["age"] < 50, "under 50", "50 or over"))
p = dp.dumbbell_plot(wb_age, "region", "life_satisfaction", "age_group",
                     legend_inside=True)
show(p)
```

## Outliers

A single variable's box plot with points beyond 1.5 times the interquartile
range flagged in the accent colour.

```python exec="1" source="material-block" session="eda"
p = dp.outlier_plot(cy, "yield")
show(p)
```

## A descriptive summary table

A "Table 1" of means, counts and missingness by group, returned as a data frame
ready for any formatter. Variables with missing values gain their own row.

```python exec="1" source="material-block" session="eda"
tab = dp.summary_table(wb, vars=["life_satisfaction", "income", "stress", "education"],
                       group="region")
print(tab.to_html(index=False))
```
