"""Every pane of the gallery app works on every bundled dataset.

The app is driven headless through Streamlit's AppTest with each pane's default
choices, as a visitor first meets it. A pane draws its figure, or says with
``st.info`` what the dataset lacks, or shows the plot's own refusal through the
handler at the foot of the app. The sweep is slow (a few minutes), so it carries
the ``slow`` marker: it runs with the full suite, and ``-m "not slow"`` leaves it
out of a quick run.
"""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

import depictr as dp  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

# Panes that draw nothing and say why, since the dataset has no column that a
# required choice can take. crop_yield and wellbeing_survey have no 0/1 outcome
# for the classification curves, no event column for the Kaplan-Meier plot and
# no two-level column besides the dumbbell's category, and wellbeing_survey has
# none at all for the confusion matrix. The two predictors the model panes
# choose by default are factors in wellbeing_survey and lexical_decision, so
# there is no numeric predictor to vary.
NOTHING_TO_OFFER = {
    "crop_yield": {"Dumbbell", "ROC curve", "PR curve", "Gains", "Lift",
                   "Calibration", "Threshold", "Kaplan-Meier"},
    "wellbeing_survey": {"Dumbbell", "ROC curve", "PR curve", "Gains", "Lift",
                         "Calibration", "Confusion matrix", "Threshold",
                         "Kaplan-Meier", "Effect of a predictor"},
    "lexical_decision": {"Effect of a predictor"},
    "clinical_trial": set(),
}

# Panes whose default choices meet the plot's own refusal: the calibration plot
# needs a probability as its score, and the first numeric column, which the app
# offers by default, is not one. Every pane in neither table must draw.
DEFAULT_CHOICE_GAPS = {
    "crop_yield": set(),
    "wellbeing_survey": set(),
    "lexical_decision": {"Calibration"},
    "clinical_trial": {"Calibration"},
}


def _selectbox(at, label):
    """The selectbox whose label starts with ``label``."""
    return next(box for box in at.selectbox if box.label.startswith(label))


def _images(at):
    """The images the app has drawn, under the element name of any release."""
    return [*at.get("image"), *at.get("imgs")]


@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(dp.DATASETS))
def test_every_pane_draws_for_every_dataset(dataset):
    pytest.importorskip("lifelines")
    pytest.importorskip("sklearn")
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    _selectbox(at, "Dataset").select(dataset).run()
    crashed, failed, noted, blank = [], {}, set(), []
    for family in _selectbox(at, "Family").options:
        _selectbox(at, "Family").select(family).run()
        for stage in _selectbox(at, "Plot").options:
            # Choosing the family has already drawn its first pane.
            if _selectbox(at, "Plot").value != stage:
                _selectbox(at, "Plot").select(stage).run()
            if at.exception:
                crashed.append(f"{family} / {stage}: {at.exception[0].value}")
            elif at.error:
                failed[stage] = at.error[0].value
            elif at.info:
                noted.add(stage)
            elif not _images(at):
                blank.append(stage)
    assert not crashed, "\n".join(crashed)
    # An empty selectbox gives None, and data[None] used to surface as the
    # message "None", which tells a visitor nothing.
    bare = {stage: msg for stage, msg in failed.items() if msg.endswith(": None")}
    assert not bare, f"{dataset}: {bare}"
    assert set(failed) == DEFAULT_CHOICE_GAPS[dataset], f"{dataset}: {failed}"
    assert noted == NOTHING_TO_OFFER[dataset], dataset
    assert not blank, f"{dataset}: nothing shown in {blank}"


def test_identifiers_are_not_offered_as_groups():
    # participant (24 levels) and item (40) in lexical_decision, like the id
    # columns of the other datasets, would make a legend of dozens of entries.
    # Only columns with 2 to 12 distinct values are offered as categories.
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    _selectbox(at, "Dataset").select("lexical_decision").run()
    _selectbox(at, "Family").select("Exploratory").run()
    _selectbox(at, "Plot").select("Categorical").run()
    assert _selectbox(at, "Category").options == ["condition", "modality"]
    _selectbox(at, "Family").select("Models (fits an OLS)").run()
    predictors = next(m for m in at.multiselect if m.label == "Predictors")
    assert "participant" not in predictors.options
    assert "item" not in predictors.options
