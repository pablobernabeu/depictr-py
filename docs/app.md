# The web app

A Streamlit app gives depictr a gallery and a low-friction way to try the
package without writing code. It loads one of the bundled datasets, or a CSV you
upload, draws the chosen plot, shows the exact Python call that produced it, and
offers a colour-vision toggle that re-renders the figure as each deficiency would
see it.

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
