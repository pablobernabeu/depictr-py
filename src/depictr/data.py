"""Small, reproducibly simulated datasets used across examples and the web app.

Each loader returns a fresh :class:`pandas.DataFrame` generated from a fixed
seed, so results are stable without shipping data files. They mirror the datasets
in the depictr R package and are chosen to exercise different plot families:
``crop_yield`` for regression and interactions, ``wellbeing_survey`` for
correlations and missingness, ``lexical_decision`` for grouped distributions and
classification, and ``clinical_trial`` for survival and imbalanced outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crop_yield(seed: int = 1) -> pd.DataFrame:
    """A field trial with a genuine fertiliser-by-treatment interaction."""
    rng = np.random.default_rng(seed)
    n = 200
    treatment = rng.choice(["standard", "enhanced"], n)
    fertiliser = rng.uniform(0, 150, n)
    rainfall = rng.normal(500, 80, n)
    soil_ph = rng.normal(6.5, 0.6, n)
    slope = np.where(treatment == "enhanced", 0.014, 0.007)
    yield_ = (
        1.5
        + slope * fertiliser
        + 0.002 * (rainfall - 500)
        + 0.3 * (soil_ph - 6.5)
        + rng.normal(0, 0.8, n)
    )
    return pd.DataFrame({
        "fertiliser": fertiliser, "rainfall": rainfall, "soil_ph": soil_ph,
        "treatment": treatment, "yield": yield_,
    })


def wellbeing_survey(seed: int = 2) -> pd.DataFrame:
    """A cross-sectional survey with informative missingness in income."""
    rng = np.random.default_rng(seed)
    n = 300
    region = rng.choice(["North", "South", "East", "West"], n)
    age = rng.integers(18, 80, n)
    education = rng.choice(["secondary", "undergraduate", "postgraduate"], n,
                           p=[0.45, 0.4, 0.15])
    stress = rng.normal(5, 2, n).clip(0, 10)
    sleep_hours = (8 - 0.2 * stress + rng.normal(0, 0.8, n)).clip(3, 11)
    income = (30000 + 800 * age - 1500 * stress + rng.normal(0, 6000, n)).clip(0, None)
    life_satisfaction = (5 - 0.3 * stress + 0.2 * sleep_hours
                         + rng.normal(0, 0.5, n)).clip(1, 7)
    df = pd.DataFrame({
        "region": region, "age": age, "education": education, "income": income,
        "stress": stress, "sleep_hours": sleep_hours,
        "life_satisfaction": life_satisfaction,
    })
    # Informative missingness: income is missing more often at higher stress.
    p_missing = (0.02 + 0.02 * (df["stress"] > 6)).to_numpy()
    df.loc[rng.random(n) < p_missing, "income"] = np.nan
    df.loc[rng.random(n) < 0.06, "sleep_hours"] = np.nan
    df.loc[rng.random(n) < 0.04, "life_satisfaction"] = np.nan
    return df


def lexical_decision(seed: int = 3) -> pd.DataFrame:
    """A priming reaction-time/accuracy experiment."""
    rng = np.random.default_rng(seed)
    n = 600
    condition = rng.choice(["related", "unrelated"], n)
    modality = rng.choice(["visual", "auditory"], n)
    word_frequency = rng.uniform(1, 6, n)
    rt = (
        650
        + 30 * (condition == "unrelated")
        + 25 * (modality == "auditory")
        - 20 * word_frequency
        + rng.gamma(2, 35, n)
    )
    p_correct = 1 / (1 + np.exp(-(2 + 0.3 * word_frequency - 0.4 * (condition == "unrelated"))))
    accuracy = (rng.random(n) < p_correct).astype(int)
    return pd.DataFrame({
        "condition": condition, "modality": modality,
        "word_frequency": word_frequency, "RT": rt, "accuracy": accuracy,
    })


def clinical_trial(seed: int = 4) -> pd.DataFrame:
    """A two-arm trial with separating survival curves and a rare adverse event."""
    rng = np.random.default_rng(seed)
    n = 300
    arm = rng.choice(["control", "treatment"], n)
    biomarker = rng.normal(0, 1, n)
    age = rng.integers(40, 85, n)
    hazard = np.where(arm == "treatment", 0.05, 0.1) * np.exp(0.2 * biomarker)
    true_time = rng.exponential(1 / hazard)
    censor = rng.uniform(0, 36, n)
    time = np.minimum(true_time, censor)
    event = (true_time <= censor).astype(int)
    p_adverse = 1 / (1 + np.exp(-(-2.5 + 0.8 * biomarker + 0.02 * (age - 60))))
    adverse_event = (rng.random(n) < p_adverse).astype(int)
    return pd.DataFrame({
        "time": time, "event": event, "arm": arm, "biomarker": biomarker,
        "age": age, "adverse_event": adverse_event,
    })


#: The datasets available, by name, for the web app and documentation.
DATASETS = {
    "crop_yield": crop_yield,
    "wellbeing_survey": wellbeing_survey,
    "lexical_decision": lexical_decision,
    "clinical_trial": clinical_trial,
}
