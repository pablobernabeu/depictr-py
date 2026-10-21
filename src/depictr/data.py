"""The datasets of the depictr R package, shipped as files.

Each loader returns a fresh :class:`pandas.DataFrame` holding the same data as
the R package's dataset of the same name, so an example gives the same numbers
in both twins and every column an R example uses exists here. The data are read
from CSV files under ``depictr/data``. The repository's
``dev/export_r_datasets.R`` writes them from the R package's ``data/*.rda``,
with a ``levels.json`` that records what a CSV cannot carry: each factor's levels
and whether it is ordered, and the storage type of every column.

The columns take the pandas types closest to R's. A factor becomes a
:class:`pandas.Categorical` with R's levels in R's order (``education`` is
ordered), so plots and tables list the levels as the R twin does. A date becomes
``datetime64``, an identifier (``field``, ``id``, ``patient``) a string, a double
``float64`` and an integer ``int64``, or ``float64`` when it has missing values.

The datasets are chosen to exercise different plot families: ``crop_yield`` for
regression and interactions, ``wellbeing_survey`` for correlations and
missingness, ``lexical_decision`` for mixed models, grouped distributions and
classification, ``clinical_trial`` for survival and imbalanced outcomes, and
``monthly_sales`` for time series. All five are synthetic, simulated with a fixed
seed by the R package's ``data-raw/generate_datasets.R``, and describe no real
people.
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from importlib.resources import files

import pandas as pd


@cache
def _schema() -> dict:
    """The column types recorded by the export script, keyed by dataset."""
    text = (files("depictr") / "data" / "levels.json").read_text(encoding="utf-8")
    return json.loads(text)["datasets"]


@cache
def _read(name: str) -> pd.DataFrame:
    """Read one shipped CSV and give each column its R type, once per session."""
    spec = _schema()[name]
    # Every column is read as text first, and only an empty field counts as
    # missing, so that no value is reinterpreted before its type is applied.
    with (files("depictr") / "data" / f"{name}.csv").open("rb") as fh:
        raw = pd.read_csv(fh, dtype=str, keep_default_na=False, na_values=[""],
                          encoding="utf-8")
    if list(raw.columns) != list(spec):
        raise RuntimeError(f"{name}.csv does not match levels.json. Rerun the export.")
    out = {}
    for column, col in spec.items():
        values = raw[column]
        kind = col["type"]
        if kind == "factor":
            factor = pd.Categorical(values, categories=col["levels"],
                                    ordered=col["ordered"])
            # pandas turns a value outside the categories into a missing value
            # without a word, so a CSV and levels.json that drifted apart would
            # lose data silently.
            if factor.isna().sum() != values.isna().sum():
                raise RuntimeError(f"{name}.{column} holds a value that levels.json "
                                   "does not list. Rerun the export.")
            out[column] = factor
        elif kind == "Date":
            out[column] = pd.to_datetime(values, format="%Y-%m-%d")
        elif kind == "character":
            out[column] = values.astype(object)
        elif kind == "integer":
            numbers = pd.to_numeric(values).astype("float64")
            out[column] = numbers if numbers.isna().any() else numbers.astype("int64")
        elif kind == "double":
            out[column] = pd.to_numeric(values).astype("float64")
        else:  # pragma: no cover - the export script writes only the types above
            raise RuntimeError(f"Unknown column type {kind!r} in levels.json.")
    return pd.DataFrame(out)


def _load(name: str, seed) -> pd.DataFrame:
    if seed is not None:
        warnings.warn(
            f"{name}(): the `seed` argument is deprecated and ignored. The "
            "datasets are no longer simulated: they are the R package's data, "
            "read from files shipped with depictr.",
            FutureWarning, stacklevel=3,
        )
    return _read(name).copy(deep=True)


def crop_yield(seed: int | None = None) -> pd.DataFrame:
    """A field trial in which the fertiliser effect depends on the treatment.

    200 fields: ``field`` (identifier), ``treatment`` (standard, enhanced),
    ``rainfall`` (mm), ``fertiliser`` (kg/ha), ``soil_ph`` and ``yield``
    (t/ha). The R package's ``crop_yield``.

    Parameters
    ----------
    seed : None
        Deprecated and ignored. Passing a value raises a ``FutureWarning``.

    Examples
    --------
    >>> import depictr as dp
    >>> dp.crop_yield().shape
    (200, 6)
    """
    return _load("crop_yield", seed)


def wellbeing_survey(seed: int | None = None) -> pd.DataFrame:
    """A cross-sectional survey with regional contrasts and informative missingness.

    300 respondents: ``id``, ``region`` (East, North, South, West), ``age``,
    ``education`` (ordered: secondary < undergraduate < postgraduate),
    ``income``, ``stress``, ``sleep_hours``, ``exercise_days`` and
    ``life_satisfaction``, with values missing in income, sleep hours, exercise
    days and life satisfaction. The R package's ``wellbeing_survey``.

    Parameters
    ----------
    seed : None
        Deprecated and ignored. Passing a value raises a ``FutureWarning``.

    Examples
    --------
    >>> import depictr as dp
    >>> dp.wellbeing_survey().shape
    (300, 9)
    """
    return _load("wellbeing_survey", seed)


def lexical_decision(seed: int | None = None) -> pd.DataFrame:
    """A priming experiment with crossed participants and items.

    960 trials: 24 ``participant`` levels crossed with 40 ``item`` levels,
    ``condition`` (related, unrelated), ``modality`` (visual, auditory),
    ``word_frequency``, ``RT`` (ms) and ``accuracy`` (0/1). The R package's
    ``lexical_decision``.

    Parameters
    ----------
    seed : None
        Deprecated and ignored. Passing a value raises a ``FutureWarning``.

    Examples
    --------
    >>> import depictr as dp
    >>> dp.lexical_decision().shape
    (960, 7)
    """
    return _load("lexical_decision", seed)


def clinical_trial(seed: int | None = None) -> pd.DataFrame:
    """A two-arm trial with separating survival curves and a rare adverse event.

    300 patients: ``patient`` (identifier), ``arm`` (placebo, treatment),
    ``age``, ``biomarker``, follow-up ``time`` (months), ``event`` (1 = event,
    0 = censored) and ``adverse_event`` (0/1). The R package's
    ``clinical_trial``.

    Parameters
    ----------
    seed : None
        Deprecated and ignored. Passing a value raises a ``FutureWarning``.

    Examples
    --------
    >>> import depictr as dp
    >>> dp.clinical_trial().shape
    (300, 7)
    """
    return _load("clinical_trial", seed)


def monthly_sales(seed: int | None = None) -> pd.DataFrame:
    """Two monthly sales series with a trend and a yearly season.

    144 rows in long format: ``date`` (the first of each month, 2018 to 2023),
    ``series`` (indoor, outdoor) and ``sales``. The R package's
    ``monthly_sales``. It is not in [`DATASETS`][depictr.DATASETS], which
    lists the cross-sectional datasets the web app offers.

    Parameters
    ----------
    seed : None
        Deprecated and ignored. Passing a value raises a ``FutureWarning``.

    Examples
    --------
    >>> import depictr as dp
    >>> dp.monthly_sales().shape
    (144, 3)
    """
    return _load("monthly_sales", seed)


DATASETS = {
    "crop_yield": crop_yield,
    "wellbeing_survey": wellbeing_survey,
    "lexical_decision": lexical_decision,
    "clinical_trial": clinical_trial,
}
"""The cross-sectional datasets, by name: a dict from each name to its loader,
so ``DATASETS["crop_yield"]()`` returns that frame. The web app offers these
four and the documentation uses them. ``monthly_sales``, a time series, has its
own loader and is not included."""
