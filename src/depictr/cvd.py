"""Colour-vision-deficiency simulation and a palette safety check.

The simulator follows the physiologically-based model of Machado, Oliveira and
Fernandes (2009), which maps a colour to the colour a person with a given form
and severity of colour-vision deficiency would perceive. The transformation is
defined on linear-light RGB, so a colour is decoded from sRGB to linear RGB (the
IEC 61966-2-1 transfer functions), transformed, then re-encoded.

This drives two things: the live colourblind-vision toggle in the web app, and
:func:`palette_safety`, which checks that the colours in a palette stay far
enough apart for each deficiency that they remain distinguishable. depictr uses
it to validate its default palette rather than merely assert that it is safe.

Machado, G. M., Oliveira, M. M., & Fernandes, L. A. F. (2009). A
physiologically-based model for simulation of color vision deficiency. IEEE
Transactions on Visualization and Computer Graphics, 15(6), 1291-1298.
https://doi.org/10.1109/TVCG.2009.113
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np

# Machado et al. (2009) transformation matrices at full severity (1.0), applied
# to linear RGB. Partial severities are approximated by interpolating towards the
# identity; the safety check below uses full severity, the worst case, exactly.
_MACHADO_1 = {
    "protan": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    "deutan": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),
    "tritan": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ]),
}

DEFICIENCIES = tuple(_MACHADO_1)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Decode sRGB channel values in [0, 1] to linear light (IEC 61966-2-1)."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Encode linear-light channel values in [0, 1] back to sRGB."""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def _to_rgb01(colours: list[str]) -> np.ndarray:
    return np.array([mcolors.to_rgb(c) for c in colours])


def simulate_cvd(colours: list[str], deficiency: str, severity: float = 1.0) -> list[str]:
    """Simulate how a palette appears under a colour-vision deficiency.

    Parameters
    ----------
    colours : list of str
        Colours (any matplotlib-readable form) to transform.
    deficiency : {"protan", "deutan", "tritan"}
        The deficiency to simulate (red-, green- or blue-weak vision).
    severity : float
        Severity in [0, 1]; 0 leaves the colours unchanged and 1 is the full
        deficiency. Intermediate values interpolate the transform towards the
        identity, an approximation to Machado et al.'s per-severity matrices.

    Returns
    -------
    list of str
        The simulated colours as hex strings.
    """
    if deficiency not in _MACHADO_1:
        raise ValueError(f"`deficiency` must be one of {DEFICIENCIES}.")
    if not 0 <= severity <= 1:
        raise ValueError("`severity` must lie in [0, 1].")
    matrix = (1 - severity) * np.eye(3) + severity * _MACHADO_1[deficiency]
    linear = _srgb_to_linear(_to_rgb01(colours))
    simulated = _linear_to_srgb((linear @ matrix.T))
    return [mcolors.to_hex(np.clip(row, 0, 1)) for row in simulated]


def _rgb_to_lab(colours: list[str]) -> np.ndarray:
    """Convert sRGB colours to CIE L*a*b* (D65)."""
    linear = _srgb_to_linear(_to_rgb01(colours))
    m = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ])
    xyz = linear @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=1)


def _min_pairwise_delta_e(colours: list[str]) -> tuple[float, tuple[int, int]]:
    """Smallest CIE76 colour difference among all pairs, and that pair's indices."""
    lab = _rgb_to_lab(colours)
    best, pair = np.inf, (0, 0)
    for i in range(len(lab)):
        for j in range(i + 1, len(lab)):
            d = float(np.linalg.norm(lab[i] - lab[j]))
            if d < best:
                best, pair = d, (i, j)
    return best, pair


def palette_safety(colours: list[str] | None = None, threshold: float = 5.0) -> dict:
    """Check that a palette stays distinguishable under each deficiency.

    For normal vision and each deficiency at full severity, the palette's colours
    are converted to CIE L*a*b* and the smallest pairwise colour difference
    (CIE76 Delta-E) is found. The lower this minimum, the more likely two
    categories are to be confused. A palette is reported safe when the minimum
    across all conditions is at least ``threshold``.

    The default ``threshold`` of 5.0 is calibrated against the reference
    colourblind-safe palette: the Okabe-Ito set's tightest pair (reddish-purple
    against grey) sits at Delta-E 7.4 under full deuteranopia, so the cut must lie
    below that to pass the recommended palette, while still flagging colours that
    become near-identical under a deficiency. The difference includes lightness,
    which survives colour-vision deficiency, so two colours that share a hue but
    differ in lightness are correctly treated as distinguishable. Full severity
    is the worst case; most colour-vision deficiency is milder.

    Parameters
    ----------
    colours : list of str, optional
        The palette to test. Defaults to the depictr qualitative palette.
    threshold : float
        The smallest acceptable Delta-E.

    Returns
    -------
    dict
        ``min_delta_e`` (the worst case across conditions), ``by_condition``
        (the minimum Delta-E for normal vision and each deficiency),
        ``worst_condition`` and ``worst_pair`` (the closest colours and where),
        ``safe`` (whether ``min_delta_e`` meets ``threshold``) and ``threshold``.
    """
    from .palette import depictr_palette

    colours = colours or depictr_palette()
    by_condition, pairs = {}, {}
    by_condition["normal"], pairs["normal"] = _min_pairwise_delta_e(colours)
    for deficiency in DEFICIENCIES:
        sim = simulate_cvd(colours, deficiency, severity=1.0)
        by_condition[deficiency], pairs[deficiency] = _min_pairwise_delta_e(sim)
    worst_condition = min(by_condition, key=by_condition.get)
    min_delta_e = by_condition[worst_condition]
    i, j = pairs[worst_condition]
    return {
        "min_delta_e": round(min_delta_e, 2),
        "by_condition": {k: round(v, 2) for k, v in by_condition.items()},
        "worst_condition": worst_condition,
        "worst_pair": (colours[i], colours[j]),
        "safe": bool(min_delta_e >= threshold),
        "threshold": threshold,
    }
