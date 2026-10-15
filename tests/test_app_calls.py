"""Every call the gallery app shows draws the figure beside it.

The app's documentation promises that a call pasted from the app gives the same
figure back. Each pane is opened on a bundled dataset it can draw, and the image
the app renders is captured on its way to the page. The call shown beside it is
then run in a namespace holding only depictr, numpy and pandas, under the names
the calls use, and the two images must agree pixel for pixel.

A few plots jitter their points through numpy's global generator, which no
argument seeds, so the generator is put in the same state before each of the
two renders. What the comparison tests is everything the call controls. The
walk renders each figure twice, so it carries the ``slow`` marker.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

import depictr as dp  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

# The panes that crop_yield has no column for, opened on clinical_trial. Every
# other pane is opened on crop_yield, whose outcome, yield, is a Python keyword.
TRIAL_PANES = {"Dumbbell", "ROC curve", "PR curve", "Gains", "Lift",
               "Confusion matrix", "Threshold", "Kaplan-Meier"}

# The calibration curve is drawn on no dataset with its default choices, since
# none holds a probability to score it by. Its call is written by the same
# line as those of the other curves of its family, which are run.
NOT_RUN = {"Calibration"}

# Choices set once a pane is open, where its defaults would hide a fault. The
# biplot and the survival curves pass a group only when one is chosen, and the
# model panes must quote the keyword yield. Each set of choices is run in turn.
CHOICES = {
    "PCA biplot": [{"Colour by": "treatment"}],
    "Coefficients": [{"Outcome": "yield"}],
    "Residual diagnostics": [{"Outcome": "yield"}],
    "Effect of a predictor": [{"Outcome": "yield"}],
    "Kaplan-Meier": [{}, {"Group": "arm"}],
}


def _selectbox(at, label):
    """The selectbox whose label starts with ``label``."""
    return next(box for box in at.selectbox if box.label.startswith(label))


def _select(at, label, value):
    """Choose ``value`` in a selectbox, rerunning the app only on a change."""
    box = _selectbox(at, label)
    if box.value != value:
        box.select(value).run()


def _render(simulate, plot, args, kwargs):
    """``simulate(plot, ...)`` with numpy's global generator in a fixed state."""
    state = np.random.get_state()
    np.random.seed(0)
    try:
        return simulate(plot, *args, **kwargs)
    finally:
        np.random.set_state(state)


def _run(code):
    """The value of the last line of ``code``, run in a fresh namespace."""
    namespace = {"dp": dp, "np": np, "pd": pd}
    *body, last = ast.parse(code).body
    exec(compile(ast.Module(body=body, type_ignores=[]), "<call>", "exec"), namespace)
    return eval(compile(ast.Expression(last.value), "<call>", "eval"), namespace)


@pytest.mark.slow
def test_every_displayed_call_reproduces_its_figure(monkeypatch):
    pytest.importorskip("lifelines")
    pytest.importorskip("sklearn")
    pytest.importorskip("statsmodels")
    # The app runs in this process and reaches simulate_figure through the
    # depictr module, so a wrapper set there sees each figure the app draws.
    simulate = dp.simulate_figure
    drawn = []

    def capture(plot, *args, **kwargs):
        image = _render(simulate, plot, args, kwargs)
        drawn.append((image, args, kwargs))
        return image

    monkeypatch.setattr(dp, "simulate_figure", capture)

    offered, checked, failures = set(), set(), []
    for dataset in ("crop_yield", "clinical_trial"):
        at = AppTest.from_file(str(APP), default_timeout=300)
        at.run()
        _select(at, "Dataset", dataset)
        for family in _selectbox(at, "Family").options:
            _select(at, "Family", family)
            for stage in _selectbox(at, "Plot").options:
                offered.add(stage)
                if stage in NOT_RUN or (stage in TRIAL_PANES) != (dataset == "clinical_trial"):
                    continue
                for choices in CHOICES.get(stage, [{}]):
                    drawn.clear()
                    _select(at, "Plot", stage)
                    for label, value in choices.items():
                        _select(at, label, value)
                    if not drawn:
                        at.run()
                    where = f"{dataset} / {stage} {choices or ''}".rstrip()
                    if at.exception or at.error or at.info or not drawn:
                        failures.append(f"{where}: drew no figure")
                        continue
                    code = at.code[0].value
                    image, args, kwargs = drawn[-1]
                    try:
                        pasted = _render(simulate, _run(code), args, kwargs)
                    except Exception as exc:
                        failures.append(f"{where}: {type(exc).__name__}: {exc}\n{code}")
                        continue
                    if pasted.shape != image.shape:
                        failures.append(f"{where}: {pasted.shape} against {image.shape}\n{code}")
                    elif not np.array_equal(pasted, image):
                        share = np.mean(np.any(pasted != image, axis=-1))
                        failures.append(f"{where}: {share:.1%} of pixels differ\n{code}")
                    checked.add(stage)
    assert not failures, "\n\n".join(failures)
    # A pane added to the app is run here unless it is listed as not run.
    assert checked | NOT_RUN == offered
