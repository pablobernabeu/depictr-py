"""Colour-vision-deficiency simulation and a palette safety check.

The simulator follows the physiologically-based model of Machado, Oliveira and
Fernandes (2009), which maps a colour to the colour a person with a given form
and severity of colour-vision deficiency would perceive. The transformation is
defined on linear-light RGB, so a colour is decoded from sRGB to linear RGB (the
IEC 61966-2-1 transfer functions), transformed, then re-encoded.

This drives three things: :func:`simulate_figure`, which shows a whole figure
as a reader with a deficiency sees it and backs the colour-vision toggle in the
web app, :func:`palette_safety`, which checks that the colours in a palette stay
far enough apart for each deficiency that they remain distinguishable, and the
colour-separability rows of :func:`depictr.audit.check_figure`, which asks the
same question of a finished figure. depictr uses them to validate its default
palette rather than merely assert that it is safe. How far apart two colours
are is measured with the CIEDE2000 formula, as Sharma, Wu and Dalal (2005) set
it out.

Machado, G. M., Oliveira, M. M., & Fernandes, L. A. F. (2009). A
physiologically-based model for simulation of color vision deficiency. IEEE
Transactions on Visualization and Computer Graphics, 15(6), 1291-1298.
https://doi.org/10.1109/TVCG.2009.113

Sharma, G., Wu, W., & Dalal, E. N. (2005). The CIEDE2000 color-difference
formula: Implementation notes, supplementary test data, and mathematical
observations. Color Research & Application, 30(1), 21-30.
https://doi.org/10.1002/col.20070
"""

from __future__ import annotations

import math
import numbers

import matplotlib.colors as mcolors
import numpy as np

# Machado et al. (2009) transformation matrices, applied to linear RGB, for
# protanomaly, deuteranomaly and tritanomaly at severities 0.0 to 1.0 in steps
# of 0.1, as the authors publish them on the article's supplementary page
# (https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html).
# They are identical, entry for entry, to the copies in colorspacious 1.1.2 and
# R's colorspace 2.1.2. ``_MACHADO[d][k]`` is the matrix for severity k / 10.
# The key order fixes the order deficiencies are reported in, and matches the
# R twin's ``.cvd_matrices_by_severity``.
_MACHADO = {
    "protan": np.array([
        # 0.0
        [[1.000000, 0.000000, 0.000000],
         [0.000000, 1.000000, 0.000000],
         [0.000000, 0.000000, 1.000000]],
        # 0.1
        [[0.856167, 0.182038, -0.038205],
         [0.029342, 0.955115, 0.015544],
         [-0.002880, -0.001563, 1.004443]],
        # 0.2
        [[0.734766, 0.334872, -0.069637],
         [0.051840, 0.919198, 0.028963],
         [-0.004928, -0.004209, 1.009137]],
        # 0.3
        [[0.630323, 0.465641, -0.095964],
         [0.069181, 0.890046, 0.040773],
         [-0.006308, -0.007724, 1.014032]],
        # 0.4
        [[0.539009, 0.579343, -0.118352],
         [0.082546, 0.866121, 0.051332],
         [-0.007136, -0.011959, 1.019095]],
        # 0.5
        [[0.458064, 0.679578, -0.137642],
         [0.092785, 0.846313, 0.060902],
         [-0.007494, -0.016807, 1.024301]],
        # 0.6
        [[0.385450, 0.769005, -0.154455],
         [0.100526, 0.829802, 0.069673],
         [-0.007442, -0.022190, 1.029632]],
        # 0.7
        [[0.319627, 0.849633, -0.169261],
         [0.106241, 0.815969, 0.077790],
         [-0.007025, -0.028051, 1.035076]],
        # 0.8
        [[0.259411, 0.923008, -0.182420],
         [0.110296, 0.804340, 0.085364],
         [-0.006276, -0.034346, 1.040622]],
        # 0.9
        [[0.203876, 0.990338, -0.194214],
         [0.112975, 0.794542, 0.092483],
         [-0.005222, -0.041043, 1.046265]],
        # 1.0
        [[0.152286, 1.052583, -0.204868],
         [0.114503, 0.786281, 0.099216],
         [-0.003882, -0.048116, 1.051998]],
    ]),
    "deutan": np.array([
        # 0.0
        [[1.000000, 0.000000, 0.000000],
         [0.000000, 1.000000, 0.000000],
         [0.000000, 0.000000, 1.000000]],
        # 0.1
        [[0.866435, 0.177704, -0.044139],
         [0.049567, 0.939063, 0.011370],
         [-0.003453, 0.007233, 0.996220]],
        # 0.2
        [[0.760729, 0.319078, -0.079807],
         [0.090568, 0.889315, 0.020117],
         [-0.006027, 0.013325, 0.992702]],
        # 0.3
        [[0.675425, 0.433850, -0.109275],
         [0.125303, 0.847755, 0.026942],
         [-0.007950, 0.018572, 0.989378]],
        # 0.4
        [[0.605511, 0.528560, -0.134071],
         [0.155318, 0.812366, 0.032316],
         [-0.009376, 0.023176, 0.986200]],
        # 0.5
        [[0.547494, 0.607765, -0.155259],
         [0.181692, 0.781742, 0.036566],
         [-0.010410, 0.027275, 0.983136]],
        # 0.6
        [[0.498864, 0.674741, -0.173604],
         [0.205199, 0.754872, 0.039929],
         [-0.011131, 0.030969, 0.980162]],
        # 0.7
        [[0.457771, 0.731899, -0.189670],
         [0.226409, 0.731012, 0.042579],
         [-0.011595, 0.034333, 0.977261]],
        # 0.8
        [[0.422823, 0.781057, -0.203881],
         [0.245752, 0.709602, 0.044646],
         [-0.011843, 0.037423, 0.974421]],
        # 0.9
        [[0.392952, 0.823610, -0.216562],
         [0.263559, 0.690210, 0.046232],
         [-0.011910, 0.040281, 0.971630]],
        # 1.0
        [[0.367322, 0.860646, -0.227968],
         [0.280085, 0.672501, 0.047413],
         [-0.011820, 0.042940, 0.968881]],
    ]),
    "tritan": np.array([
        # 0.0
        [[1.000000, 0.000000, 0.000000],
         [0.000000, 1.000000, 0.000000],
         [0.000000, 0.000000, 1.000000]],
        # 0.1
        [[0.926670, 0.092514, -0.019184],
         [0.021191, 0.964503, 0.014306],
         [0.008437, 0.054813, 0.936750]],
        # 0.2
        [[0.895720, 0.133330, -0.029050],
         [0.029997, 0.945400, 0.024603],
         [0.013027, 0.104707, 0.882266]],
        # 0.3
        [[0.905871, 0.127791, -0.033662],
         [0.026856, 0.941251, 0.031893],
         [0.013410, 0.148296, 0.838294]],
        # 0.4
        [[0.948035, 0.089490, -0.037526],
         [0.014364, 0.946792, 0.038844],
         [0.010853, 0.193991, 0.795156]],
        # 0.5
        [[1.017277, 0.027029, -0.044306],
         [-0.006113, 0.958479, 0.047634],
         [0.006379, 0.248708, 0.744913]],
        # 0.6
        [[1.104996, -0.046633, -0.058363],
         [-0.032137, 0.971635, 0.060503],
         [0.001336, 0.317922, 0.680742]],
        # 0.7
        [[1.193214, -0.109812, -0.083402],
         [-0.058496, 0.979410, 0.079086],
         [-0.002346, 0.403492, 0.598854]],
        # 0.8
        [[1.257728, -0.139648, -0.118081],
         [-0.078003, 0.975409, 0.102594],
         [-0.003316, 0.501214, 0.502102]],
        # 0.9
        [[1.278864, -0.125333, -0.153531],
         [-0.084748, 0.957674, 0.127074],
         [-0.000989, 0.601151, 0.399838]],
        # 1.0
        [[1.255528, -0.076749, -0.178779],
         [-0.078411, 0.930809, 0.147602],
         [0.004733, 0.691367, 0.303900]],
    ]),
}

# The full-severity matrices, which palette_safety() and check_figure() use.
_MACHADO_1 = {deficiency: table[10] for deficiency, table in _MACHADO.items()}

DEFICIENCIES = tuple(_MACHADO)

# The colour differences palette_safety() and check_figure() can measure.
_METRICS = ("ciede2000", "cie76")

# Degrees to radians and back, the products that math.radians() and
# math.degrees() compute, written out so that the R twin evaluates the same
# products and the twins round alike.
_RADIANS_PER_DEGREE = math.pi / 180
_DEGREES_PER_RADIAN = 180 / math.pi


def _check_metric(metric) -> None:
    """Refuse a colour difference that the package does not measure."""
    if not isinstance(metric, str) or metric not in _METRICS:
        raise ValueError("`metric` must be 'ciede2000' or 'cie76'.")


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Decode sRGB channel values in [0, 1] to linear light (IEC 61966-2-1)."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Encode linear-light channel values in [0, 1] back to sRGB."""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def _to_rgb01(colours: list[str]) -> np.ndarray:
    return np.array([mcolors.to_rgb(c) for c in colours])


def _as_colour_list(colours) -> list:
    """A list of colours, reading a single string as one colour.

    A string is iterable, so ``list("#005b96")`` would split it into
    characters: a hex code would fail on ``"#"``, and ``"rgb"`` would become
    three colours.
    """
    return [colours] if isinstance(colours, str) else list(colours)


def _machado_matrix(deficiency: str, severity: float) -> np.ndarray:
    """The Machado transform for one deficiency and severity.

    Between two tabulated severities, the matrix is interpolated linearly, as
    the supplementary page suggests and as colorspacious and R's
    ``colorspace::interpolate_cvd_transform()`` do, so 0.25 lies halfway
    between the matrices for 0.2 and 0.3. Severities 0 and 1 take their
    matrices exactly as published, which keeps full-severity results bit for
    bit. The R twin's ``.cvd_matrix()`` computes the same expressions.
    """
    table = _MACHADO[deficiency]
    scaled = 10 * severity
    i = math.floor(scaled)
    f = scaled - i
    if f == 0:
        return table[i]
    return (1 - f) * table[i] + f * table[i + 1]


def _check_vision(deficiency, severity) -> None:
    """Refuse a deficiency or severity the simulation does not cover."""
    if deficiency not in _MACHADO:
        raise ValueError(
            "`deficiency` must be one of 'protan', 'deutan' or 'tritan'."
        )
    if not 0 <= severity <= 1:
        raise ValueError("`severity` must lie in [0, 1].")


def simulate_cvd(colours: str | list[str], deficiency: str,
                 severity: float = 1.0) -> list[str]:
    """Simulate how a palette appears under a colour-vision deficiency.

    Parameters
    ----------
    colours : str or list of str
        Colours (any matplotlib-readable form) to transform. A single string
        is one colour, and an empty list gives an empty list back, as in the
        R twin.
    deficiency : {"protan", "deutan", "tritan"}
        The deficiency to simulate (red-, green- or blue-weak vision).
    severity : float
        Severity in [0, 1]. Zero leaves the colours unchanged and one is the
        full deficiency. Machado, Oliveira and Fernandes (2009) publish a
        matrix for each tenth of severity, which is used as it stands at 0.1,
        0.2 and so on. A severity between two tenths takes the linear
        interpolation of their matrices, as the authors suggest and as
        colorspacious does.

    Returns
    -------
    list of str
        The simulated colours as hex strings.

    References
    ----------
    Machado, G. M., Oliveira, M. M., & Fernandes, L. A. F. (2009). A
    physiologically-based model for simulation of color vision deficiency. IEEE
    Transactions on Visualization and Computer Graphics, 15(6), 1291-1298.
    https://doi.org/10.1109/TVCG.2009.113

    Examples
    --------
    >>> import depictr as dp
    >>> dp.simulate_cvd(['#005b96', '#e69f00'], 'deutan')
    ['#275295', '#cab411']

    See Also
    --------
    depictr.simulate_figure : the same simulation applied to a whole figure.
    """
    _check_vision(deficiency, severity)
    colours = _as_colour_list(colours)
    if not colours:
        return []
    matrix = _machado_matrix(deficiency, severity)
    linear = _srgb_to_linear(_to_rgb01(colours))
    simulated = _linear_to_srgb(linear @ matrix.T)
    return [mcolors.to_hex(np.clip(row, 0, 1)) for row in simulated]


def _simulate_pixels(rgba: np.ndarray, deficiency: str, severity: float) -> np.ndarray:
    """A copy of an RGBA image in [0, 1], its colours as a deficiency sees them.

    Each pixel takes the transform that :func:`simulate_cvd` applies to a
    colour, without rounding to eight bits, and keeps its alpha.
    """
    seen = np.array(rgba, dtype=float)
    matrix = _machado_matrix(deficiency, severity)
    simulated = _linear_to_srgb(_srgb_to_linear(seen[..., :3]) @ matrix.T)
    seen[..., :3] = np.clip(simulated, 0, 1)
    return seen


def _drawn(plot) -> bool:
    """Whether plotnine has drawn a plot, or any plot of a composition.

    Drawing attaches the matplotlib figure to the plot, and the composition
    attaches its figure to each plot it holds.
    """
    if "figure" in vars(plot):
        return True
    return any(_drawn(item) for item in getattr(plot, "items", ())
               if item is not None)


def simulate_figure(plot, deficiency: str, severity: float = 1.0,
                    dpi: float = 150) -> np.ndarray:
    """Show a finished figure as a reader with a colour-vision deficiency sees it.

    [`simulate_cvd`][depictr.simulate_cvd] simulates a list of colours, and
    this function the figure they end up in, legend, colourbar, text and
    background included. A copy of the plot is drawn and rasterised with
    matplotlib's Agg renderer. Every pixel then goes through the same
    transform: it is decoded from sRGB to linear light, multiplied by the
    matrix of Machado, Oliveira and Fernandes (2009) for the deficiency and
    severity and encoded again. The alpha channel is kept as drawn.

    The R twin's ``simulate_figure()`` applies the same matrices to the
    figure's vector elements and returns a gtable to draw or save. A solid
    region comes out in the same colour in both, to within the rounding of an
    eight-bit channel.

    Parameters
    ----------
    plot : plotnine.ggplot or plotnine.composition.Compose
        The plot, or a composition of several, as it would be saved. It is
        left as it was, since a copy is drawn. A plot whose ``draw()`` method
        has already been called is refused, because plotnine does not draw it
        correctly a second time.
    deficiency : {"protan", "deutan", "tritan"}
        The deficiency to simulate (red-, green- or blue-weak vision).
    severity : float
        Severity in [0, 1], as for [`simulate_cvd`][depictr.simulate_cvd].
        Zero returns the pixels as drawn, which is the figure under normal
        vision.
    dpi : float
        The resolution to rasterise at, in dots per inch. The image is the
        figure's width and height in inches times ``dpi`` pixels.

    Returns
    -------
    numpy.ndarray
        The image as floats in [0, 1], of shape (height, width, 4), with the
        channels red, green, blue and alpha. Matplotlib's ``imshow()`` and
        Streamlit's ``st.image()`` take it as it is.

    Raises
    ------
    TypeError
        When ``plot`` is not a plotnine plot or composition.
    ValueError
        When ``deficiency``, ``severity`` or ``dpi`` is out of range, or when
        ``plot`` has already been drawn.

    References
    ----------
    Machado, G. M., Oliveira, M. M., & Fernandes, L. A. F. (2009). A
    physiologically-based model for simulation of color vision deficiency. IEEE
    Transactions on Visualization and Computer Graphics, 15(6), 1291-1298.
    https://doi.org/10.1109/TVCG.2009.113

    Examples
    --------
    >>> import depictr as dp
    >>> p = dp.ecdf_plot(dp.lexical_decision(), "RT", group="condition")
    >>> seen = dp.simulate_figure(p, "deutan", dpi=72)
    >>> seen.shape[2], bool(seen.min() >= 0 and seen.max() <= 1)
    (4, True)

    Under normal vision, for a side-by-side comparison:

    >>> as_drawn = dp.simulate_figure(p, "deutan", severity=0, dpi=72)
    >>> as_drawn.shape == seen.shape
    True
    """
    import copy

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from plotnine import ggplot
    from plotnine.composition import Compose

    _check_vision(deficiency, severity)
    if not isinstance(plot, (ggplot, Compose)):
        raise TypeError("`plot` must be a plotnine ggplot or composition.")
    if (isinstance(dpi, bool) or not isinstance(dpi, numbers.Real)
            or not math.isfinite(dpi) or dpi <= 0):
        raise ValueError("`dpi` must be a positive number.")

    # Drawing builds a plot's layers and attaches the figure to it in place,
    # so a copy is drawn and the caller's plot is left as it was. A plot drawn
    # before cannot be drawn again: plotnine's copy shares its figure and its
    # built state, and the second drawing came out with the panel missing.
    if _drawn(plot):
        raise ValueError(
            "`plot` has already been drawn, and plotnine does not draw a plot "
            "correctly a second time. Pass the plot before it is drawn, or "
            "build it again.")
    figure = copy.deepcopy(plot).draw(show=False)
    try:
        figure.set_dpi(dpi)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=float) / 255
    finally:
        plt.close(figure)
    if severity == 0:
        return rgba
    return _simulate_pixels(rgba, deficiency, severity)


def _rgb_to_lab(colours: list[str]) -> np.ndarray:
    """Convert sRGB colours to CIE L*a*b* (D65)."""
    linear = _srgb_to_linear(_to_rgb01(colours))
    # The sRGB D65 primaries to seven decimals. The rounded four-decimal form
    # shifts a Delta-E by a few thousandths, which is enough to round a reported
    # distance differently from the R twin on the same colours.
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=1)


def _ciede2000(lab1, lab2) -> float:
    """CIEDE2000 colour difference between two CIE L*a*b* colours.

    The formula as Sharma, Wu and Dalal (2005) set it out, with the parametric
    weights kL, kC and kH at 1. Where the definition leaves a choice, theirs is
    followed. A neutral colour has a hue angle of zero. A pair in which either
    colour is neutral has no hue difference, and its mean hue is the sum of the
    two hue angles. The R twin's ``.ciede2000()`` evaluates the same
    expressions in the same order. The 34 pairs the authors published are in
    ``tests/data/sharma2005.csv``.
    """
    l1, a1, b1 = (float(v) for v in lab1)
    l2, a2, b2 = (float(v) for v in lab2)

    # a* is scaled up for colours of low chroma, the formula's correction for
    # greys, before chroma and hue are taken.
    c_bar = (math.sqrt(a1 * a1 + b1 * b1) + math.sqrt(a2 * a2 + b2 * b2)) / 2
    g = 0.5 * (1 - math.sqrt(c_bar ** 7 / (c_bar ** 7 + 25 ** 7)))
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = math.sqrt(a1p * a1p + b1 * b1)
    c2p = math.sqrt(a2p * a2p + b2 * b2)

    def hue(b: float, a: float) -> float:
        h = 0.0 if b == 0 and a == 0 else math.atan2(b, a) * _DEGREES_PER_RADIAN
        return h + 360 if h < 0 else h

    h1p = hue(b1, a1p)
    h2p = hue(b2, a2p)

    # The differences in lightness, chroma and hue, the hue difference taken
    # the short way round the circle.
    neutral = c1p * c2p == 0
    dlp = l2 - l1
    dcp = c2p - c1p
    dhp = h2p - h1p
    if neutral:
        dhp = 0.0
    elif dhp > 180:
        dhp = dhp - 360
    elif dhp < -180:
        dhp = dhp + 360
    dbig_hp = 2 * math.sqrt(c1p * c2p) * math.sin(dhp / 2 * _RADIANS_PER_DEGREE)

    # The weights for lightness, chroma and hue, taken at the pair's means. The
    # term that couples the chroma and hue differences is the formula's
    # correction for blues.
    lbp = (l1 + l2) / 2
    cbp = (c1p + c2p) / 2
    if neutral:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p - 360) / 2
    t_term = (1 - 0.17 * math.cos((hbp - 30) * _RADIANS_PER_DEGREE)
              + 0.24 * math.cos(2 * hbp * _RADIANS_PER_DEGREE)
              + 0.32 * math.cos((3 * hbp + 6) * _RADIANS_PER_DEGREE)
              - 0.20 * math.cos((4 * hbp - 63) * _RADIANS_PER_DEGREE))
    z = (hbp - 275) / 25
    d_theta = 30 * math.exp(-(z * z))
    r_c = 2 * math.sqrt(cbp ** 7 / (cbp ** 7 + 25 ** 7))
    l50 = lbp - 50
    s_l = 1 + 0.015 * (l50 * l50) / math.sqrt(20 + l50 * l50)
    s_c = 1 + 0.045 * cbp
    s_h = 1 + 0.015 * cbp * t_term
    r_t = -math.sin(2 * d_theta * _RADIANS_PER_DEGREE) * r_c

    lightness = dlp / s_l
    chroma = dcp / s_c
    hue_term = dbig_hp / s_h
    return math.sqrt(lightness * lightness + chroma * chroma
                     + hue_term * hue_term + r_t * chroma * hue_term)


def _min_pairwise_delta_e(colours: list[str], groups=None,
                          metric: str = "ciede2000"
                          ) -> tuple[float, tuple[int, int]]:
    """Smallest colour difference among all pairs, and that pair's indices.

    The difference is CIEDE2000, or with ``metric="cie76"`` the Euclidean
    distance in CIELAB. ``groups``, when given, labels each colour, and a pair
    whose labels are equal is skipped: ``check_figure`` labels each drawn
    colour with the colour it was declared as, so that one group drawn in two
    ways is not set against itself.
    """
    lab = _rgb_to_lab(colours)
    best, pair = np.inf, (0, 0)
    for i in range(len(lab)):
        for j in range(i + 1, len(lab)):
            if groups is not None and groups[i] == groups[j]:
                continue
            if metric == "cie76":
                d = float(np.linalg.norm(lab[i] - lab[j]))
            else:
                d = _ciede2000(lab[i], lab[j])
            if d < best:
                best, pair = d, (i, j)
    return best, pair


def palette_safety(colours: str | list[str] | None = None, threshold: float = 5.0,
                   metric: str = "ciede2000") -> dict:
    """Check that a palette stays distinguishable under each deficiency.

    For normal vision and each deficiency at full severity, the palette's
    colours are converted to CIELAB and the smallest colour difference between
    any two of them is found. The difference is CIEDE2000 (Sharma, Wu & Dalal,
    2005) unless ``metric`` asks for the older measure. The lower this minimum,
    the more likely two categories are to be confused. A palette is reported
    safe when the minimum across all conditions is at least ``threshold``.

    CIE76, the Euclidean distance in CIELAB, overstates the difference between
    saturated colours: the blues ``#005b96`` and ``#0060a8`` are 8.20 apart by
    CIE76 and 2.69 by CIEDE2000. ggplot2's default palette of five colours
    passes the check under CIE76, although under deuteranopia its sky blue and
    its pink come within 2.74 of each other in CIEDE2000.

    The default ``threshold`` of 5.0 is calibrated against the built-in palette
    of [`depictr.depictr_palette`][], which is based on the Okabe-Ito set. That
    palette's tightest pair (reddish-purple against grey) sits at 6.43 under full
    deuteranopia, so the cut must lie below that to pass the built-in palette,
    while still flagging colours that become near-identical under a deficiency.
    The difference includes lightness, which survives colour-vision deficiency,
    so two colours that share a hue but differ in lightness are correctly
    treated as distinguishable. Full severity is the worst case; most
    colour-vision deficiency is milder.

    Parameters
    ----------
    colours : str or list of str, optional
        The palette to test. Defaults to the depictr qualitative palette. Needs
        at least two colours: a pairwise distance over fewer than two has no
        value, not an infinitely safe one. A single string is one colour, so
        it raises with that message.
    threshold : float
        The smallest acceptable colour difference, on the scale of ``metric``.
    metric : {"ciede2000", "cie76"}
        The colour difference to measure: CIEDE2000, the default, or CIE76,
        the Euclidean distance in CIELAB. ``"cie76"`` is kept so that results
        computed with it can be reproduced, and a later release will remove it.

    Returns
    -------
    dict
        ``min_delta_e`` (the worst case across conditions), ``by_condition``
        (the minimum colour difference for normal vision and each deficiency),
        ``worst_condition`` and ``worst_pair`` (the closest colours and where),
        ``safe`` (whether ``min_delta_e`` meets ``threshold``) and ``threshold``.

    See Also
    --------
    depictr.check_figure : the same question asked of a finished figure, which
        uses only as many colours as it has groups and has text and a background
        besides.

    References
    ----------
    Sharma, G., Wu, W., & Dalal, E. N. (2005). The CIEDE2000 color-difference
    formula: Implementation notes, supplementary test data, and mathematical
    observations. Color Research & Application, 30(1), 21-30.
    https://doi.org/10.1002/col.20070

    Examples
    --------
    >>> import depictr as dp
    >>> dp.palette_safety()['safe']
    True

    ggplot2's default palette of five colours, which passes under CIE76.

    >>> hue5 = ["#F8766D", "#A3A500", "#00BF7D", "#00B0F6", "#E76BF3"]
    >>> dp.palette_safety(hue5)['safe'], dp.palette_safety(hue5, metric="cie76")['safe']
    (False, True)
    """
    from .palette import depictr_palette

    _check_metric(metric)

    # `colours or ...` would have swapped in the default palette for an empty
    # list, reporting on eight colours the caller never passed.
    colours = depictr_palette() if colours is None else _as_colour_list(colours)
    if len(colours) < 2:
        # Otherwise the no-pair sentinel below survives to the result, which
        # then claims safe=True at an infinite distance and names one colour as
        # both halves of the worst pair. inf is also not JSON-serialisable, and
        # this dict gets printed straight into the published docs.
        raise ValueError(
            "`colours` needs at least two colours to have a pairwise distance."
        )
    by_condition, pairs = {}, {}
    by_condition["normal"], pairs["normal"] = _min_pairwise_delta_e(
        colours, metric=metric)
    for deficiency in DEFICIENCIES:
        sim = simulate_cvd(colours, deficiency, severity=1.0)
        by_condition[deficiency], pairs[deficiency] = _min_pairwise_delta_e(
            sim, metric=metric)
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
