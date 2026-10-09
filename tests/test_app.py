"""The gallery app, driven headless through Streamlit's AppTest."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _selectbox(at, label):
    """The selectbox whose label starts with ``label``."""
    return next(box for box in at.selectbox if box.label.startswith(label))


def _images(at):
    """The images the app has drawn.

    Streamlit 1.59 gave AppTest a typed image element, of type "image". Before
    it, an image was an unknown element named after its proto field, "imgs",
    so asking for one name alone finds no image under the other releases.
    """
    return [*at.get("image"), *at.get("imgs")]


def _survival_pane(dataset):
    """The app with ``dataset`` loaded and the Kaplan-Meier pane open."""
    # The timeout is an upper limit, so a generous one costs nothing when the
    # runs are quick. The first run imports the plotting stack and draws a
    # figure, and on a loaded machine one run took longer than the 120 seconds
    # first allowed here.
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    _selectbox(at, "Dataset").select(dataset).run()
    _selectbox(at, "Family").select("Survival").run()
    assert not at.exception
    return at


def test_kaplan_meier_pane_opens_on_the_event_column():
    # Regression: the Event selector offered every numeric column and opened on
    # the first of them, `time`, so the pane drew the follow-up times as event
    # codes (lifelines counts every non-zero value as an event).
    pytest.importorskip("lifelines")
    at = _survival_pane("clinical_trial")
    event = _selectbox(at, "Event")
    assert event.value == "event"
    assert "time" not in event.options
    assert not at.error
    assert len(_images(at)) == 1


def test_kaplan_meier_pane_explains_a_dataset_with_no_event_column():
    # crop_yield has no column coded 0/1 or 1/2, so there is nothing to offer
    # as the event indicator: the pane says so and draws no figure.
    at = _survival_pane("crop_yield")
    assert not [box for box in at.selectbox if box.label.startswith("Event")]
    assert "event column" in at.info[0].value
    assert not at.error
    assert not _images(at)
