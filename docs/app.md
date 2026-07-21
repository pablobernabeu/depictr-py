# The web app

A Streamlit app gives depictr a gallery and a low-friction way to try the
package without writing code. It loads one of the bundled datasets, or a CSV you
upload, draws the chosen plot, shows the exact Python call that produced it, and
offers a colour-vision toggle that re-renders the figure as each deficiency would
see it.

## What it puts on screen

The sidebar holds the data and the two accessibility controls. The main pane
holds the plot family, the plot, its variables, the figure and the call that
produced it. Here it is with the crop-yield trial loaded and an empirical
cumulative distribution split by treatment.

![The depictr app with the crop-yield dataset loaded. The sidebar shows the dataset selector, a colour-vision selector reading "normal" and the palette's worst-case colour distance of 7.4. The main pane shows the first rows of the data, selectors reading Exploratory, ECDF, yield and treatment, a cumulative distribution with a blue curve for the enhanced treatment and an orange one for the standard treatment, and beside it the call dp.ecdf_plot(dp.crop_yield(), 'yield', group='treatment', legend_inside=False).](assets/app-plot-pane.png){ width="820" }

The call beside the figure is the point of the whole pane. Whatever you arrive
at by clicking, you can paste into a script and get the same figure back.

## The colour-vision selector

Changing "Simulate vision" in the sidebar re-renders the figure as a reader with
that deficiency would see it, through the same Machado-2009 simulation the
[accessibility page](gallery/accessibility.md) reports on. Below is the identical
plot with the selector set to deuteranopia. The blue is barely touched and the
orange has moved towards mustard. Surviving that shift is what the palette is
chosen for, and the two curves stay as easy to tell apart as before.

![The same app view with the colour-vision selector set to "deutan". The cumulative distribution is redrawn with the standard-treatment curve shifted from orange to mustard yellow while the enhanced-treatment curve stays blue, and the two remain clearly distinguishable.](assets/app-colour-vision.png){ width="820" }

## Run it

The app lives in the repository rather than the wheel, so run it from a clone:

```bash
git clone https://github.com/pablobernabeu/depictr-py
cd depictr-py
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

The app runs on your own machine, so nothing you load leaves it. Every plot it
draws is a plotnine object from the same functions documented in the
[reference](reference/theme-and-accessibility.md), so anything you find there can
be reproduced in a script with the call the app shows you.
