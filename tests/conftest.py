"""Shared test configuration.

Force matplotlib's non-interactive Agg backend so plots render off-screen
(plotnine builds on matplotlib, and the default interactive backend needs a
display, which a test runner does not have).
"""

import matplotlib

matplotlib.use("Agg")
