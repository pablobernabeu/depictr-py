"""arrange_plots(): the panel tags the R twin's patchwork annotation draws."""

import copy

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import pytest
from plotnine import ggplot
from plotnine.composition import Compose, plot_spacer

import depictr as dp

LD = dp.lexical_decision()
CY = dp.crop_yield()


def _plot(base_size=11):
    return dp.ecdf_plot(LD, "RT") + dp.theme_depictr(base_size=base_size)


def _panels(plot):
    """The panels of a plot or composition in reading order, spacers left out."""
    if isinstance(plot, Compose):
        for item in plot:
            yield from _panels(item)
    elif not isinstance(plot, plot_spacer):
        yield plot


def _tags(plot):
    return [panel.labels.tag for panel in _panels(plot)]


def _drawn_tags(plot, tags):
    """The point size of each drawn text whose string is one of ``tags``."""
    built = copy.deepcopy(plot)
    figure = built.draw(show=False)
    try:
        return {t.get_text(): float(t.get_fontsize())
                for t in figure.findobj(mtext.Text) if t.get_text() in tags}
    finally:
        plt.close(figure)


def test_panels_are_tagged_a_b_c_by_default_as_in_r():
    composed = dp.arrange_plots(_plot(), _plot(), _plot())
    assert _tags(composed) == ["A", "B", "C"]
    # Each tag is drawn, at the theme's plot_tag size: 1.2 times the base
    # size, which is also ggplot2's plot.tag default.
    assert _drawn_tags(composed, {"A", "B", "C"}) == {
        "A": pytest.approx(13.2), "B": pytest.approx(13.2),
        "C": pytest.approx(13.2)}


@pytest.mark.parametrize(("levels", "expected"), [
    ("A", ["A", "B", "C", "D"]),
    ("a", ["a", "b", "c", "d"]),
    ("1", ["1", "2", "3", "4"]),
    ("I", ["I", "II", "III", "IV"]),
    ("i", ["i", "ii", "iii", "iv"]),
    (["left", "right", "under", "over"], ["left", "right", "under", "over"]),
    (("w", "x", "y", "z", "spare"), ["w", "x", "y", "z"]),
])
def test_tag_levels_take_patchworks_styles_and_custom_tags(levels, expected):
    composed = dp.arrange_plots(*[_plot() for _ in range(4)],
                                tag_levels=levels)
    assert _tags(composed) == expected


def test_tag_levels_none_leaves_the_panels_untagged():
    composed = dp.arrange_plots(_plot(), _plot(), tag_levels=None)
    assert _tags(composed) == [None, None]


def test_a_lone_plot_is_tagged_as_patchwork_tags_one():
    lone = dp.arrange_plots(_plot())
    assert isinstance(lone, ggplot)
    assert lone.labels.tag == "A"
    assert dp.arrange_plots(_plot(), tag_levels=None).labels.tag is None


def test_tags_run_on_through_a_nested_composition_and_skip_spacers():
    # patchwork continues one sequence through a nested patchwork and gives a
    # spacer no tag, as check_figure() gives one no panel number.
    nested = dp.arrange_plots(_plot() | plot_spacer() | _plot(), _plot(),
                              ncol=1)
    assert _tags(nested) == ["A", "B", "C"]


def test_tag_size_sets_the_point_size_of_every_tag():
    composed = dp.arrange_plots(_plot(), _plot(base_size=5), tag_size=8)
    assert _drawn_tags(composed, {"A", "B"}) == {
        "A": pytest.approx(8), "B": pytest.approx(8)}


def test_the_callers_plots_are_left_untagged():
    first, second = _plot(), _plot()
    inner = first | second
    dp.arrange_plots(first, inner, _plot())
    assert first.labels.tag is None and second.labels.tag is None
    assert _tags(inner) == [None, None]


@pytest.mark.parametrize("levels", ["B", "AA", "", 1, ["x", 2]])
def test_tag_levels_it_does_not_know_are_refused(levels):
    with pytest.raises(ValueError) as excinfo:
        dp.arrange_plots(_plot(), _plot(), tag_levels=levels)
    assert str(excinfo.value) == (
        "`tag_levels` must be 'A', 'a', '1', 'I' or 'i', a sequence of tags, "
        "or None.")


def test_too_few_custom_tags_are_refused():
    with pytest.raises(ValueError) as excinfo:
        dp.arrange_plots(_plot(), _plot(), _plot(), tag_levels=["x", "y"])
    assert str(excinfo.value) == (
        "`tag_levels` gives 2 tags for 3 panels.")


def test_letters_run_out_after_26_panels():
    with pytest.raises(ValueError) as excinfo:
        dp.arrange_plots(*[_plot() for _ in range(27)])
    assert str(excinfo.value) == (
        "`tag_levels = 'A'` has 26 letters for 27 panels. Pass '1' or a "
        "sequence of tags.")


@pytest.mark.parametrize("size", [0, -1, "large", True, float("nan")])
def test_a_tag_size_that_is_not_a_positive_number_is_refused(size):
    with pytest.raises(ValueError) as excinfo:
        dp.arrange_plots(_plot(), _plot(), tag_size=size)
    assert str(excinfo.value) == "`tag_size` must be a single positive number."


def test_the_package_composites_are_tagged_where_the_r_twin_tags_them():
    smf = pytest.importorskip("statsmodels.formula.api")
    fit = smf.ols('Q("yield") ~ rainfall + fertiliser + soil_ph', CY).fit()
    # The R twin tags model_report() alone; its other composites call
    # patchwork directly and draw no tags.
    assert _tags(dp.model_report(fit)) == ["A", "B", "C"]
    assert _tags(dp.residual_diagnostics_plot(fit)) == [None] * 4
    assert set(_tags(dp.explore_pairs(CY, ["yield", "rainfall"]))) == {None}
    assert _tags(dp.estimation_plot(LD, "RT", "condition", two_panel=True,
                                    seed=1)) == [None, None]
