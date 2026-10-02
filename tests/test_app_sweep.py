"""Every pane of the gallery app works on every bundled dataset.

The app is driven headless through Streamlit's AppTest with each pane's default
choices, as a visitor first meets it. A pane that cannot draw shows an error
through the handler at the foot of the app. The sweep is slow (a few minutes,
most of it in the ridgeline and pairs panes), so it carries the ``slow`` marker:
it runs with the full suite, and ``-m "not slow"`` leaves it out of a quick run.
"""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

import depictr as dp  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

# Panes whose default choices do not suit a dataset, whatever its values: no
# 0/1 column to offer as the classification outcome (crop_yield,
# wellbeing_survey), a score column that is not a probability for the
# calibration plot, no second two-level column for the dumbbell, or no numeric
# predictor among the two offered by default. The gaps lie in how the app
# chooses its defaults, and the simulated data these datasets replaced showed
# the same errors in the same panes. Any other pane must draw.
DEFAULT_CHOICE_GAPS = {
    "crop_yield": {"Dumbbell", "ROC curve", "PR curve", "Gains", "Lift",
                   "Calibration", "Threshold"},
    "wellbeing_survey": {"Dumbbell", "ROC curve", "PR curve", "Gains", "Lift",
                         "Calibration", "Confusion matrix", "Threshold",
                         "Effect of a predictor"},
    "lexical_decision": {"Calibration", "Effect of a predictor"},
    "clinical_trial": {"Calibration"},
}


def _selectbox(at, label):
    """The selectbox whose label starts with ``label``."""
    return next(box for box in at.selectbox if box.label.startswith(label))


@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(dp.DATASETS))
def test_every_pane_draws_for_every_dataset(dataset):
    pytest.importorskip("lifelines")
    pytest.importorskip("sklearn")
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    _selectbox(at, "Dataset").select(dataset).run()
    crashed, failed = [], []
    for family in _selectbox(at, "Family").options:
        _selectbox(at, "Family").select(family).run()
        for stage in _selectbox(at, "Plot").options:
            # Choosing the family has already drawn its first pane.
            if _selectbox(at, "Plot").value != stage:
                _selectbox(at, "Plot").select(stage).run()
            if at.exception:
                crashed.append(f"{family} / {stage}: {at.exception[0].value}")
            elif at.error:
                failed.append(stage)
    assert not crashed, "\n".join(crashed)
    unexpected = set(failed) - DEFAULT_CHOICE_GAPS[dataset]
    assert not unexpected, f"{dataset}: no figure in {sorted(unexpected)}"


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
