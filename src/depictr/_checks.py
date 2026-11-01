"""Argument checks that several modules share, worded as the R twin words them."""

from __future__ import annotations

import numbers


def _check_conf_level(conf_level, allow_none: bool = False) -> None:
    """Refuse a confidence level outside (0, 1).

    A level given as a percentage, such as 95, has no normal or t quantile, so
    the intervals would come out NaN or the call fail far from the argument,
    and a level of 0 or 1 would draw a zero-width or an unbounded interval.
    Every function that takes a ``conf_level`` calls this first, so each
    refuses the same values with the R twin's message. ``allow_none`` admits
    ``None`` where it leaves the limits out, as ``NA`` does in R.
    """
    if allow_none and conf_level is None:
        return
    if (isinstance(conf_level, bool) or not isinstance(conf_level, numbers.Real)
            or not 0 < conf_level < 1):
        raise ValueError(
            "`conf_level` must be a single number strictly between 0 and 1"
            + (", or None to leave the limits out." if allow_none else ".")
        )
