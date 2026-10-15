"""format_terms(): the R twin's label rules, and the names patsy gives terms."""

import pandas as pd
import pytest

import depictr as dp
from depictr import format_terms


def test_format_terms_is_exported():
    assert "format_terms" in dp.__all__


def test_format_terms_follows_the_r_rules():
    # The same inputs and outputs as the R twin's test-helpers.R and
    # test-t2-foundational.R.
    assert format_terms("(Intercept)") == "Intercept"
    assert format_terms("b_conditionB") == "conditionB"
    assert format_terms("a:b") == "a × b"
    assert format_terms("a:b", interaction="asterisk") == "a * b"
    assert format_terms("a:b", interaction="colon") == "a:b"
    assert format_terms("a:b", interaction="space") == "a b"
    assert format_terms(["(Intercept)", None, "b_x:b_y"]) == [
        "Intercept", None, "x × y"]
    assert format_terms("b_a:bs_b:b_c", interaction="colon") == "a:b:c"
    x = ["b_x", "bs_x", "b_b_x", "ab_x", "x_b_y", "bx", "b_", "b_x:b_y",
         "b_x:bs_y:b_z", "x:b_y", ":b_x", "x::b_y", "", None]
    assert format_terms(x, interaction="colon") == [
        "x", "x", "b x", "ab x", "x b y", "bx", "", "x:y", "x:y:z", "x:y",
        ":x", "x::y", "", None]
    assert format_terms(x, strip_prefix=False, interaction="colon") == [
        None if s is None else s.replace("_", " ") for s in x]
    assert format_terms([]) == []
    assert format_terms("(Intercept)", tidy_intercept=False) == "(Intercept)"
    assert format_terms(["x_y:z", "Intercept"], tidy_intercept=False,
                        interaction="asterisk") == ["x y * z", "Intercept"]


def test_format_terms_wraps_where_r_wraps():
    # Values from the R twin's format_terms(), whose strwrap() keeps each line
    # shorter than `wrap` characters.
    assert format_terms("alpha:beta:gamma:delta", wrap=8) == (
        "alpha ×\nbeta ×\ngamma ×\ndelta")
    assert format_terms([None, "some long label"], wrap=4) == [
        None, "some\nlong\nlabel"]
    assert format_terms(["word_frequency:condition", "a", ""], wrap=10) == [
        "word\nfrequency\n×\ncondition", "a", ""]


def test_format_terms_keeps_missing_values_missing():
    # R keeps NA as NA, never the text "NA". A pandas column holds NaN or None.
    assert format_terms([float("nan"), "x", pd.NA]) == [None, "x", None]
    assert format_terms(pd.Series(["b_x", None])) == ["x", None]
    assert format_terms(None) is None


def test_format_terms_reads_patsy_names():
    # A factor level becomes "variable: level", and the Q() and C() wrappers
    # that a formula needs are dropped, so a label never shows formula syntax.
    assert format_terms("treatment[T.enhanced]") == "treatment: enhanced"
    assert format_terms("C(treatment)[T.enhanced]") == "treatment: enhanced"
    assert format_terms(
        'C(treatment, Treatment(reference="enhanced"))[T.standard]'
    ) == "treatment: standard"
    assert format_terms('Q("region")[T.North]') == "region: North"
    assert format_terms("Q('region')[T.North]") == "region: North"  # formulaic
    assert format_terms('Q("stress")') == "stress"
    assert format_terms('C(Q("soil type"))[T.clay]') == "soil type: clay"
    assert format_terms("treatment[enhanced]") == "treatment: enhanced"
    assert format_terms("rainfall:treatment[T.enhanced]") == (
        "rainfall × treatment: enhanced")
    assert format_terms("rainfall:treatment[T.enhanced]",
                        interaction="colon") == "rainfall:treatment: enhanced"
    assert format_terms("soil_ph") == "soil ph"
    assert format_terms(["Intercept", "const"]) == ["Intercept", "Intercept"]
    assert format_terms("const", tidy_intercept=False) == "const"
    # A colon inside a level or a call is not an interaction.
    assert format_terms("band[T.10:20]") == "band: 10:20"
    assert format_terms('Q("a:b")') == "a:b"


def test_format_terms_validates_its_arguments():
    with pytest.raises(ValueError, match="`interaction` must be one of"):
        format_terms("a:b", interaction="cross")
    with pytest.raises(ValueError, match="`wrap` must be a positive"):
        format_terms("a", wrap=0)
    with pytest.raises(TypeError, match="`x` must be"):
        format_terms(3)
