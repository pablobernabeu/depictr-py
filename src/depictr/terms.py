"""Readable labels for the terms of a fitted model.

A model names its coefficients for the computer: patsy writes a factor level as
``treatment[T.enhanced]``, an interaction as ``rainfall:treatment[T.enhanced]``
and a quoted column as ``Q("region")[T.North]``. :func:`format_terms` turns
such names into labels a reader can take in, following the R twin's rules, and
the helpers below give :func:`depictr.coefficient_plot` and
:func:`depictr.compare_models` their default labels and their ``labels`` and
``interaction`` arguments.

A factor's coefficient is labelled with its variable and level, as
``treatment: enhanced``, whatever the number of levels. The row then says
which contrast it shows, here enhanced against the reference level, standard.
"""

from __future__ import annotations

import itertools
import numbers
import re
import warnings
from collections.abc import Iterable, Mapping

import pandas as pd

_SEPARATORS = {"times": " × ", "asterisk": " * ", "colon": ":", "space": " "}

# The names an intercept takes: R's, patsy's and that of statsmodels'
# add_constant().
_INTERCEPTS = ("(Intercept)", "Intercept", "const")

# The terms the forest plots drop with the intercept, brms's b_Intercept
# among them, as in the R twin.
_INTERCEPT_PATTERN = r"(?i)(?:b_)?\(?intercept\)?|const"

# The prefix that patsy and formulaic put before a level for their built-in
# contrasts: Treatment, Sum, Helmert and Diff.
_CONTRAST_PREFIX = re.compile(r"^[TSHD]\.")

# A Q("...") wrapper, which a formula needs for a name that is not a Python
# identifier. patsy writes it with double quotes and formulaic with single.
_QUOTED = re.compile(r"""^Q\(\s*(["'])(.*)\1\s*\)$""", re.DOTALL)


def interaction_separator(interaction: str) -> str:
    """The text that joins the components of an interaction label."""
    try:
        return _SEPARATORS[interaction]
    except (KeyError, TypeError):
        raise ValueError(
            "`interaction` must be one of 'times', 'asterisk', 'colon' or 'space'."
        ) from None


def _is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split ``text`` at each ``sep`` outside brackets, parentheses and quotes.

    A colon inside ``Q("a:b")``, inside a call or inside the brackets around a
    level, as in ``band[T.10:20]``, does not join two components.
    """
    parts, depth, quote, start = [], 0, None, 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _closing_paren(text: str, open_at: int) -> int:
    """Index of the parenthesis that closes the one at ``open_at``, or -1."""
    depth, quote = 0, None
    for i in range(open_at, len(text)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _variable_name(code: str) -> str:
    """The variable a formula factor names, without its Q() or C() wrapper.

    ``Q("region")`` gives ``region`` and ``C(treatment, Treatment("b"))`` gives
    ``treatment``. Any other code, such as ``np.log(rainfall)``, is kept.
    """
    code = code.strip()
    quoted = _QUOTED.match(code)
    if quoted:
        return quoted.group(2)
    if code.startswith("C(") and _closing_paren(code, 1) == len(code) - 1:
        first = _split_top_level(code[2:-1], ",")[0]
        return _variable_name(first)
    return code


def _level_start(component: str) -> int:
    """Index of the bracket that opens a level suffix, or -1 without one.

    The suffix is the bracketed text at the end of a component, such as
    ``[T.enhanced]``. A bracket inside a call or inside quotes, as in
    ``C(x, levels=["a", "b"])``, does not open one.
    """
    if not component.endswith("]"):
        return -1
    depth, quote = 0, None
    for i, ch in enumerate(component):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif ch == "[" and depth == 0 and i > 0:
            return i
    return -1


def _component_label(component: str) -> str:
    """Label one component of a term: ``var[T.b]`` becomes ``var: b``."""
    start = _level_start(component)
    if start < 0:
        return _variable_name(component)
    level = _CONTRAST_PREFIX.sub("", component[start + 1:-1])
    return f"{_variable_name(component[:start])}: {level}"


def _wrap(text: str, width: int) -> str:
    """Break ``text`` into lines shorter than ``width``, as R's strwrap() does.

    A word longer than the width keeps a line of its own.
    """
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) < width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return "\n".join(lines)


def _format_one(term: str, sep: str, strip_prefix: bool, tidy_intercept: bool,
                wrap: int | None) -> str:
    if strip_prefix:
        # A prefix counts at the start of the term or straight after a colon,
        # as in the R twin.
        term = re.sub(r"(^|:)bs?_", r"\1", term)
    if tidy_intercept and term in _INTERCEPTS:
        term = "Intercept"
    else:
        term = sep.join(_component_label(c) for c in _split_top_level(term, ":"))
    term = term.replace("_", " ")
    return term if wrap is None else _wrap(term, wrap)


def format_terms(x, interaction: str = "times", strip_prefix: bool = True,
                 tidy_intercept: bool = True, wrap: int | None = None):
    """Tidy raw coefficient names for display.

    Cleans up the term names that modelling functions produce so that they read
    well on a plot, following the rules of the R twin's ``format_terms()``. The
    intercept is renamed ``Intercept``, a ``b_`` or ``bs_`` prefix (as brms adds)
    is stripped from each component of an interaction, interaction colons are
    written with a chosen symbol, and underscores are shown as spaces, so
    ``word_frequency`` becomes ``word frequency``.

    The names patsy and formulaic give a formula's terms are read as well. A
    factor level becomes ``variable: level``, as ``treatment[T.enhanced]`` and
    ``C(treatment)[T.enhanced]`` become ``treatment: enhanced``, and a
    ``Q("...")`` wrapper is dropped, so ``Q("region")[T.North]`` becomes
    ``region: North``. A colon inside a level, a call or a quoted name does not
    count as an interaction.

    Parameters
    ----------
    x : str or iterable of str
        The term names. Missing values (``None``, ``NaN``) are kept missing.
    interaction : {"times", "asterisk", "colon", "space"}
        How to write an interaction: a multiplication sign (the default), an
        asterisk, the colon unchanged or a space.
    strip_prefix : bool
        Whether to remove a leading ``b_`` or ``bs_`` from each interaction
        component.
    tidy_intercept : bool
        Whether to replace ``(Intercept)``, ``Intercept`` and ``const`` with
        ``Intercept``.
    wrap : int, optional
        Width at which to wrap long labels onto new lines, each line kept
        shorter than ``wrap`` characters, as R's ``strwrap()`` keeps them. The
        default leaves labels unwrapped.

    Returns
    -------
    str or list
        A string for a string, otherwise a list as long as ``x``, with
        ``None`` where ``x`` held a missing value.

    Examples
    --------
    >>> import depictr as dp
    >>> dp.format_terms(["Intercept", "treatment[T.enhanced]", "soil_ph"])
    ['Intercept', 'treatment: enhanced', 'soil ph']
    >>> dp.format_terms("rainfall:treatment[T.enhanced]", interaction="asterisk")
    'rainfall * treatment: enhanced'
    >>> dp.format_terms('Q("region")[T.North]')
    'region: North'
    """
    sep = interaction_separator(interaction)
    if wrap is not None and (isinstance(wrap, bool)
                             or not isinstance(wrap, numbers.Integral) or wrap < 1):
        raise ValueError("`wrap` must be a positive whole number or None.")

    def one(term):
        if _is_missing(term):
            return None
        return _format_one(str(term), sep, strip_prefix, tidy_intercept, wrap)

    if x is None or isinstance(x, str):
        return one(x)
    if not isinstance(x, Iterable):
        raise TypeError("`x` must be a string or an iterable of strings.")
    return [one(term) for term in x]


# ---- default labels for the forest plots ----------------------------------


def _design_info(model):
    """The patsy DesignInfo of a formula fit, or None.

    statsmodels up to 0.14 keeps it as ``data.design_info`` and 0.15 as
    ``data.model_spec``, which holds a formulaic ModelSpec instead when the
    formula engine is set to formulaic. Only patsy's records how each column
    name was built, so a formulaic fit is labelled from its names.
    """
    if isinstance(model, pd.DataFrame):  # a table of estimates has no design
        return None
    data = getattr(getattr(model, "model", None), "data", None)
    for attribute in ("model_spec", "design_info"):
        info = getattr(data, attribute, None)
        if info is not None and hasattr(info, "term_codings"):
            return info
    return None


def _level_of(suffix: str, categories: list[str]) -> str:
    """The level a patsy contrast suffix names: ``[T.b]`` gives ``b``."""
    if suffix.startswith("[") and suffix.endswith("]"):
        inner = suffix[1:-1]
        if inner not in categories and _CONTRAST_PREFIX.match(inner):
            return inner[2:]
        return inner
    return suffix.lstrip(".")  # a polynomial contrast's .Linear, .Quadratic


def _subterm_columns(info, subterm):
    """Each column a patsy subterm fills, with its label pieces.

    Rebuilds the column names as patsy builds them, the leftmost factor
    varying fastest, so each name comes with the variable and level of every
    factor in it. Yields ``(name, pieces)``, where ``pieces`` is None when no
    factor in the column names a level.
    """
    per_factor = []
    for factor in subterm.factors:
        code = factor.name()
        variable = _variable_name(code)
        factor_info = info.factor_infos[factor]
        if factor_info.type == "categorical":
            categories = [str(c) for c in factor_info.categories]
            suffixes = subterm.contrast_matrices[factor].column_suffixes
            per_factor.append([
                (code + s, f"{variable}: {_level_of(s, categories)}", True)
                for s in suffixes])
        elif factor_info.num_columns > 1:
            per_factor.append([(f"{code}[{i}]", f"{variable}: {i}", True)
                               for i in range(factor_info.num_columns)])
        else:
            per_factor.append([(code, variable, False)])
    if not per_factor:
        yield "Intercept", None
        return
    for combination in itertools.product(*reversed(per_factor)):
        combination = combination[::-1]
        name = ":".join(piece[0] for piece in combination)
        named = any(piece[2] for piece in combination)
        yield name, [piece[1] for piece in combination] if named else None


def _design_labels(model, interaction: str) -> dict:
    """Labels for the factor columns of a patsy formula fit, by column name.

    The design records where each factor's name ends and its level begins, so
    a level holding a colon or a bracket is labelled whole. The R twin builds
    the same map from the model matrix. Each label is also keyed under a
    ``b_`` prefix, so a brms-style table lines up with the fit. A column that
    names no level, and any fit without a patsy design, is left to
    :func:`format_terms`. If the rebuilt names ever differ from the design's
    own, the map is dropped, so a change in patsy cannot mislabel a column.
    """
    info = _design_info(model)
    if info is None:
        return {}
    sep = interaction_separator(interaction)
    names, labels = [], {}
    try:
        for subterms in info.term_codings.values():
            for subterm in subterms:
                for name, pieces in _subterm_columns(info, subterm):
                    names.append(name)
                    if pieces is not None:
                        labels[name] = sep.join(pieces).replace("_", " ")
    except (AttributeError, KeyError, TypeError):
        return {}
    if names != list(info.column_names):
        return {}
    labels.update({f"b_{name}": label for name, label in list(labels.items())})
    return labels


def _is_raw(labels) -> bool:
    return isinstance(labels, str) and labels == "raw"


def _check_labels(labels) -> None:
    """Refuse a ``labels`` argument of a kind the plots cannot read."""
    if labels is None or _is_raw(labels) or isinstance(labels, (Mapping, list, tuple)):
        return
    raise TypeError(
        "`labels` must be a dict from term to label, a list with one label per "
        "term, or 'raw'."
    )


def _warn_unused_labels(terms, labels, stacklevel: int = 3) -> None:
    """Warn when a key of a ``labels`` dict matches no term.

    A mistyped key would otherwise leave the raw name on the plot with nothing
    to say that the relabelling never happened. The sentence is the one
    posterior_plot() and the R twin give.
    """
    if not isinstance(labels, Mapping):
        return
    terms = list(dict.fromkeys(str(t) for t in terms))
    unused = sorted(set(map(str, labels)) - set(terms))
    if unused:
        warnings.warn(
            f"`labels` keys not found among the parameters: {', '.join(unused)}. "
            f"The parameters are {', '.join(terms)}.",
            stacklevel=stacklevel,
        )


def _check_label_count(labels, n: int, owner: str, read_in: str,
                       intercept_dropped: bool) -> None:
    """Refuse a ``labels`` list that is not one label per term, as R does."""
    if len(labels) == n:
        return
    dropped = ", after dropping the intercept" if intercept_dropped else ""
    raise ValueError(
        f"`labels` has length {len(labels)} but {owner} {n} "
        f"{'term' if n == 1 else 'terms'} to label ({read_in}{dropped})."
    )


def _term_labels(terms, labels, auto: dict, interaction: str) -> list[str]:
    """Labels for ``terms``: the user's dict first, then the defaults.

    ``labels="raw"`` keeps the names as the model gives them. Otherwise, a term
    takes its label from the ``labels`` dict, then from ``auto``, the factor
    labels read from a model's design, then from :func:`format_terms`.
    """
    terms = [str(t) for t in terms]
    if _is_raw(labels):
        return terms
    given = labels if isinstance(labels, Mapping) else {}
    return [
        str(given[t]) if t in given
        else auto[t] if t in auto
        else format_terms(t, interaction=interaction)
        for t in terms
    ]


def _check_distinct(terms, labels) -> None:
    """Refuse two terms drawn under one label, which would share one row."""
    seen = {}
    for term, label in zip(terms, labels, strict=True):
        if label in seen and seen[label] != term:
            raise ValueError(
                f"Terms {seen[label]} and {term} share the label {label!r}; "
                "rename one with `labels`."
            )
        seen.setdefault(label, term)
