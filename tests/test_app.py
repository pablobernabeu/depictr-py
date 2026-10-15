"""The gallery app, driven headless through Streamlit's AppTest."""

import re
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

# Twelve rows, so the pane the app opens on (the distribution of the first
# numeric column) has enough to draw. The accented header is the case that
# broke the app: Excel on Windows saves it in cp1252, as a byte UTF-8 rejects.
CSV_TEXT = "área,valor,grupo\n" + "".join(
    f"{i}.0,{(i * 7) % 5}.5,{'ab'[i % 2]}\n" for i in range(12)
)


def _selectbox(at, label):
    """The selectbox whose label starts with ``label``."""
    return next(box for box in at.selectbox if box.label.startswith(label))


def _upload(payload):
    """The app after ``payload`` is uploaded as a CSV file."""
    # AppTest learnt to upload a file after the oldest Streamlit the app
    # supports, so under such a release these tests have nothing to drive.
    if not hasattr(AppTest, "file_uploader"):
        pytest.skip("this Streamlit's AppTest cannot upload a file")
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    next(r for r in at.radio if r.label == "Data").set_value("Upload a CSV").run()
    at.file_uploader[0].set_value(("data.csv", payload, "text/csv")).run()
    return at


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


def test_the_colour_vision_selector_redraws_the_figure():
    # Under a deficiency, the figure comes from dp.simulate_figure(), which
    # replaced the app's own simulation of the pixels.
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    _selectbox(at, "Simulate vision").select("deutan").run()
    assert not at.exception
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


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("utf-8", CSV_TEXT),
        ("cp1252", CSV_TEXT),
        # Byte 0x81 has no character in cp1252, so only latin-1, which gives
        # every byte one, can read this file.
        ("latin-1", CSV_TEXT.replace("b\n", "b\x81\n")),
    ],
    ids=["utf-8", "cp1252", "latin-1"],
)
def test_an_upload_loads_in_its_encoding_and_the_call_names_it(encoding, text):
    # Regression: the app read every upload as UTF-8, outside the error
    # handler, so a cp1252 file ended the page in a UnicodeDecodeError.
    at = _upload(text.encode(encoding))
    assert not at.exception
    assert not at.error
    assert "área" in at.dataframe[0].value.columns
    assert len(_images(at)) == 1
    # pandas reads UTF-8 by default, so that call needs no argument.
    call = ("pd.read_csv(your_file)" if encoding == "utf-8"
            else f"pd.read_csv(your_file, encoding={encoding!r})")
    assert call in at.code[0].value


@pytest.mark.parametrize(
    "payload", [b"", b'a,b\n1,2\n3,4,5,6\n"unterminated\n'], ids=["empty", "ragged"]
)
def test_an_unreadable_upload_gets_an_error_not_a_traceback(payload):
    at = _upload(payload)
    assert not at.exception
    assert "Could not read data.csv as a CSV" in at.error[0].value
    assert not at.dataframe
    assert not _images(at)


def test_an_upload_with_no_numeric_column_gets_a_note_not_none():
    # The pane the app opens on needs a numeric column. Given no options, a
    # selectbox returns None, and the plot failed on data[None] with the
    # message "None".
    at = _upload(b"name,colour\nx,red\ny,blue\nz,red\n")
    assert not at.exception
    assert not at.error
    assert "numeric column" in at.info[0].value
    assert not _images(at)


@pytest.mark.parametrize(
    ("dataset", "family", "stage", "needs"),
    [
        # treatment is crop_yield's one two-level column, so it cannot be both
        # the category and the group, and wellbeing_survey has none.
        ("crop_yield", "Exploratory", "Dumbbell", "two-level column"),
        ("wellbeing_survey", "Exploratory", "Dumbbell", "two-level column"),
        ("crop_yield", "Classification", "ROC curve", "coded 0/1"),
        ("wellbeing_survey", "Classification", "Calibration", "coded 0/1"),
        ("wellbeing_survey", "Classification", "Confusion matrix", "two-level"),
        # The two predictors chosen by default are factors, so there is no
        # numeric predictor to vary.
        ("lexical_decision", "Models (fits an OLS)", "Effect of a predictor",
         "numeric predictor"),
    ],
)
def test_a_pane_with_nothing_to_offer_says_what_it_needs(dataset, family, stage, needs):
    # Regression: these panes showed "Could not draw that plot for this data:
    # None", or the refusal of a plot handed a column it cannot use.
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    _selectbox(at, "Dataset").select(dataset).run()
    _selectbox(at, "Family").select(family).run()
    if _selectbox(at, "Plot").value != stage:
        _selectbox(at, "Plot").select(stage).run()
    assert not at.exception
    assert not at.error
    assert needs in at.info[0].value
    assert not _images(at)


def test_the_app_uses_the_width_api_its_streamlit_floor_provides():
    # use_container_width is deprecated, and Streamlit logs on every rerun that
    # it will be removed. Its replacement, width="stretch", reached st.image in
    # Streamlit 1.49, so the extras that install Streamlit for the app must not
    # allow an older release.
    assert "use_container_width" not in APP.read_text(encoding="utf-8")
    pyproject = (APP.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    floors = re.findall(r'"streamlit>=([0-9.]+)"', pyproject)
    assert len(floors) == 2  # the app and all extras
    for floor in floors:
        assert tuple(int(part) for part in floor.split(".")[:2]) >= (1, 49)
