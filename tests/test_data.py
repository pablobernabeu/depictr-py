"""The bundled datasets are the R package's own data, shipped as CSV files.

The CSVs and levels.json under src/depictr/data are written by
dev/export_r_datasets.R from the R twin's data/*.rda. The values pinned below
that come from R are printed by dev/goldens/datasets.R, run against the R twin
at the commit the files were exported from (depictr 0.3.0).
"""

from __future__ import annotations

import hashlib
import warnings
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

import depictr as dp
from depictr import data

# SHA-256 of each shipped CSV. A change here means the data changed: rerun the
# export script, check the R-derived values below and update both together.
CSV_SHA256 = {
    "crop_yield": "08a5dc314bbd115a0c9c14d968e58653fd11ea013077076683d5af0b1ca389b9",
    "wellbeing_survey": "cd3c49d9c38f0c65f2b079c85b106af8f0a97ccc329776c7e9b8bc9a34c223d6",
    "lexical_decision": "ef3d945449648c487b7f5f026b9fbbb16a6802017b6c2988f1ce1f19f544c9b2",
    "clinical_trial": "8eddda06435e56b4cd75b60eb982f8785682a44b313800b8bca92ae250a3f2de",
    "monthly_sales": "90ccf0260382e220e9a38f7476ff2c6d38ce78b63c6a5e328180dcfb304f96cd",
}

LOADERS = {
    "crop_yield": dp.crop_yield,
    "wellbeing_survey": dp.wellbeing_survey,
    "lexical_decision": dp.lexical_decision,
    "clinical_trial": dp.clinical_trial,
    "monthly_sales": dp.monthly_sales,
}

# Column names in R's order, and the shapes R reports.
COLUMNS = {
    "crop_yield": ["field", "treatment", "rainfall", "fertiliser", "soil_ph", "yield"],
    "wellbeing_survey": ["id", "region", "age", "education", "income", "stress",
                         "sleep_hours", "exercise_days", "life_satisfaction"],
    "lexical_decision": ["participant", "item", "condition", "modality",
                         "word_frequency", "RT", "accuracy"],
    "clinical_trial": ["patient", "arm", "age", "biomarker", "time", "event",
                       "adverse_event"],
    "monthly_sales": ["date", "series", "sales"],
}
SHAPES = {"crop_yield": (200, 6), "wellbeing_survey": (300, 9),
          "lexical_decision": (960, 7), "clinical_trial": (300, 7),
          "monthly_sales": (144, 3)}

# Factor levels in R's order (levels() in R).
LEVELS = {
    ("crop_yield", "treatment"): ["standard", "enhanced"],
    ("wellbeing_survey", "region"): ["East", "North", "South", "West"],
    ("wellbeing_survey", "education"): ["secondary", "undergraduate", "postgraduate"],
    ("lexical_decision", "participant"): [f"P{i:02d}" for i in range(1, 25)],
    ("lexical_decision", "item"): [f"I{i:02d}" for i in range(1, 41)],
    ("lexical_decision", "condition"): ["related", "unrelated"],
    ("lexical_decision", "modality"): ["visual", "auditory"],
    ("clinical_trial", "arm"): ["placebo", "treatment"],
    ("monthly_sales", "series"): ["indoor", "outdoor"],
}

# Column sums of the numeric columns and missing-value counts, from R.
SUMS = {
    "crop_yield": {"rainfall": 103416, "fertiliser": 15235, "soil_ph": 1292.86,
                   "yield": 594.98},
    "wellbeing_survey": {"age": 12739, "income": 8682214, "stress": 1215.4,
                         "sleep_hours": 1963.8, "exercise_days": 901,
                         "life_satisfaction": 1265.6},
    "lexical_decision": {"word_frequency": 3247.92, "RT": 637397, "accuracy": 902},
    "clinical_trial": {"age": 17816, "biomarker": 4.13, "time": 6526.8, "event": 136,
                       "adverse_event": 31},
    "monthly_sales": {"sales": 34783},
}
MISSING = {"income": 23, "sleep_hours": 18, "exercise_days": 12,
           "life_satisfaction": 7}


@pytest.mark.parametrize("name", sorted(CSV_SHA256))
def test_shipped_csv_is_the_exported_file(name):
    raw = (files("depictr") / "data" / f"{name}.csv").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CSV_SHA256[name]


@pytest.mark.parametrize("name", sorted(LOADERS))
def test_shape_and_columns_match_r(name):
    df = LOADERS[name]()
    assert df.shape == SHAPES[name]
    assert list(df.columns) == COLUMNS[name]


@pytest.mark.parametrize(("dataset", "column"), sorted(LEVELS))
def test_factors_are_categorical_in_r_level_order(dataset, column):
    series = LOADERS[dataset]()[column]
    assert isinstance(series.dtype, pd.CategoricalDtype)
    assert list(series.cat.categories) == LEVELS[(dataset, column)]
    assert series.cat.ordered == (column == "education")
    assert series.notna().all()


def test_education_is_ordered_secondary_to_postgraduate():
    edu = dp.wellbeing_survey()["education"]
    assert edu.cat.ordered
    assert (edu.min(), edu.max()) == ("secondary", "postgraduate")


@pytest.mark.parametrize(("dataset", "column"), [
    ("crop_yield", "field"), ("wellbeing_survey", "id"), ("clinical_trial", "patient"),
])
def test_identifiers_are_strings(dataset, column):
    series = LOADERS[dataset]()[column]
    assert not isinstance(series.dtype, pd.CategoricalDtype)
    assert all(isinstance(v, str) for v in series)
    assert series.is_unique


def test_numeric_dtypes_follow_r_storage():
    # R doubles are float64. R integers are int64, or float64 when a value is
    # missing (exercise_days), since int64 has no missing value.
    ld = dp.lexical_decision()
    assert ld["accuracy"].dtype == np.int64
    assert ld["RT"].dtype == np.float64
    ct = dp.clinical_trial()
    assert ct["event"].dtype == np.int64
    assert ct["adverse_event"].dtype == np.int64
    assert ct["age"].dtype == np.float64
    wb = dp.wellbeing_survey()
    assert wb["exercise_days"].dtype == np.float64
    assert wb["age"].dtype == np.float64


def test_monthly_sales_dates_are_datetimes():
    ms = dp.monthly_sales()
    assert pd.api.types.is_datetime64_any_dtype(ms["date"])
    assert ms["date"].min() == pd.Timestamp("2018-01-01")
    assert ms["date"].nunique() == 72


@pytest.mark.parametrize("name", sorted(SUMS))
def test_values_match_r(name):
    df = LOADERS[name]()
    for column, total in SUMS[name].items():
        assert df[column].sum() == pytest.approx(total, abs=1e-6), column


def test_missing_values_match_r():
    counts = dp.wellbeing_survey().isna().sum()
    assert counts[counts > 0].to_dict() == MISSING


def test_no_negative_yield():
    # The simulated crop_yield this replaces held two impossible negative
    # yields (minimum -1.25). R's minimum is 0.20.
    assert dp.crop_yield()["yield"].min() == pytest.approx(0.20)


def test_clinical_trial_log_rank_matches_r():
    # R: survival::survdiff(Surv(time, event) ~ arm) gives 39.7917 on 1 df.
    lifelines = pytest.importorskip("lifelines.statistics")
    ct = dp.clinical_trial()
    assert ct["event"].sum() == 136
    pla, trt = (ct[ct["arm"] == a] for a in ("placebo", "treatment"))
    res = lifelines.logrank_test(pla["time"], trt["time"], pla["event"], trt["event"])
    assert res.test_statistic == pytest.approx(39.7917, abs=1e-3)


def test_each_call_returns_a_fresh_frame():
    first = dp.crop_yield()
    first.loc[0, "yield"] = -99.0
    first["extra"] = 1
    again = dp.crop_yield()
    assert again.loc[0, "yield"] == pytest.approx(1.96)
    assert "extra" not in again.columns


def test_value_missing_from_levels_json_is_refused(monkeypatch):
    # pandas would turn every "enhanced" into a missing value if levels.json
    # lost that level. The loader refuses the file and keeps the data intact.
    schema = {name: {col: dict(spec) for col, spec in cols.items()}
              for name, cols in data._schema().items()}
    schema["crop_yield"]["treatment"]["levels"] = ["standard"]
    data._read.cache_clear()
    monkeypatch.setattr(data, "_schema", lambda: schema)
    try:
        with pytest.raises(RuntimeError, match="crop_yield.treatment"):
            dp.crop_yield()
    finally:
        data._read.cache_clear()


@pytest.mark.parametrize("name", sorted(LOADERS))
def test_seed_is_deprecated_and_ignored(name):
    loader = LOADERS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = loader()
    with pytest.warns(FutureWarning, match="seed"):
        seeded = loader(seed=123)
    pd.testing.assert_frame_equal(seeded, plain)


def test_datasets_mapping_holds_the_cross_sectional_four():
    assert list(dp.DATASETS) == ["crop_yield", "wellbeing_survey",
                                 "lexical_decision", "clinical_trial"]
    assert "monthly_sales" in dp.__all__
    assert data.monthly_sales is dp.monthly_sales
