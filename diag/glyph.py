"""Diagnose 'failed to load glyph' on macOS (temporary, never merged)."""

import json
import platform
import subprocess
import sys
import traceback

MODE = sys.argv[1] if len(sys.argv) > 1 else "main"


def bad():
    import depictr as dp
    from plotnine import aes, geom_point, ggplot, scale_colour_manual
    cy = dp.crop_yield()
    return (ggplot(cy, aes("rainfall", "yield", colour="treatment"))
            + geom_point()
            + scale_colour_manual(values=["#d62728", "#309208"])
            + dp.theme_depictr(base_size=5))


def spy():
    """Report the font, size and dpi of every failing text measurement."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import findfont
    real = RendererAgg.get_text_width_height_descent

    def wrapped(self, s, prop, ismath):
        try:
            return real(self, s, prop, ismath)
        except RuntimeError as e:
            print("FAIL measure", repr(s), "size_pt", prop.get_size_in_points(),
                  "dpi", self.dpi, "family", prop.get_family(),
                  "file", findfont(prop), "err", e, flush=True)
            raise
    RendererAgg.get_text_width_height_descent = wrapped


def run(label, fn):
    try:
        fn()
        print("OK  ", label, flush=True)
    except Exception as e:  # noqa: BLE001
        print("ERR ", label, type(e).__name__, e, flush=True)
        traceback.print_exc(limit=3)


if MODE == "main":
    import matplotlib
    import matplotlib.ft2font as ft2font
    import plotnine
    print(sys.version)
    print(platform.platform())
    print("matplotlib", matplotlib.__version__,
          "freetype", getattr(ft2font, "__freetype_version__", "?"),
          "build", getattr(ft2font, "__freetype_build_type__", "?"))
    print("plotnine", plotnine.__version__, "backend", matplotlib.get_backend())
    print("text.hinting", matplotlib.rcParams["text.hinting"],
          "hinting_factor", matplotlib.rcParams["text.hinting_factor"])
    from matplotlib.font_manager import FontProperties, findfont, get_font
    from matplotlib.ft2font import LoadFlags
    print("plotnine base_family", plotnine.options.base_family)
    for fam in ["sans-serif", "DejaVu Sans", plotnine.options.base_family]:
        try:
            print("findfont", fam, "->", findfont(FontProperties(family=[fam])))
        except Exception as e:  # noqa: BLE001
            print("findfont", fam, "ERR", e)
    path = findfont(FontProperties(family=["sans-serif"]))
    flags = {n: getattr(LoadFlags, n) for n in
             ["DEFAULT", "NO_HINTING", "FORCE_AUTOHINT", "NO_AUTOHINT",
              "TARGET_LIGHT"] if hasattr(LoadFlags, n)}
    fails = []
    for size in [1, 2, 3, 3.5, 4, 4.5, 5, 5.75, 6, 8]:
        for dpi in [72, 100, 300]:
            for name, flag in flags.items():
                font = get_font(path)
                try:
                    font.set_size(size, dpi)
                    font.set_text("yield", 0.0, flags=flag)
                except RuntimeError as e:
                    fails.append((size, dpi, name, str(e)))
    print("direct FT2Font failures (size, dpi, flag):", json.dumps(fails))
    # Each scenario in a fresh process, so no state is shared.
    for scenario in ["plotnine72", "plotnine100", "save72", "save100",
                     "check_then_plotnine72", "audit_only", "save72_check",
                     "rc_diff"]:
        print(f"--- {scenario}", flush=True)
        subprocess.run([sys.executable, __file__, scenario], check=False)
    sys.exit(0)

spy()
import os  # noqa: E402
import tempfile  # noqa: E402

import depictr as dp  # noqa: E402

tmp = tempfile.mkdtemp()
out = os.path.join(tmp, "f.png")
if MODE == "plotnine72":
    run("plotnine save 72", lambda: bad().save(out, width=8.9, height=6,
                                                units="cm", dpi=72,
                                                verbose=False))
elif MODE == "plotnine100":
    run("plotnine save 100", lambda: bad().save(out, width=8.9, height=6,
                                                 units="cm", dpi=100,
                                                 verbose=False))
elif MODE == "save72":
    run("save_plot 72", lambda: dp.save_plot(bad(), out, width=8.9, height=6,
                                             units="cm", dpi=72))
elif MODE == "save100":
    run("save_plot 100", lambda: dp.save_plot(bad(), out, width=8.9, height=6,
                                              units="cm", dpi=100))
elif MODE == "check_then_plotnine72":
    run("check_figure", lambda: dp.check_figure(bad(), width_cm=8.9,
                                                render_width_cm=8.9))
    run("then plotnine save 72", lambda: bad().save(out, width=8.9, height=6,
                                                     units="cm", dpi=72,
                                                     verbose=False))
elif MODE == "audit_only":
    run("draw at theme size", lambda: bad().draw(show=False))
elif MODE == "save72_check":
    run("save_plot 72 check", lambda: dp.save_plot(
        bad(), out, width=8.9, height=6, units="cm", dpi=72, check=True))
elif MODE == "rc_diff":
    import matplotlib
    before = dict(matplotlib.rcParams)
    dp.check_figure(bad(), width_cm=8.9, render_width_cm=8.9)
    after = dict(matplotlib.rcParams)
    print("rcParams changed by check_figure:",
          {k: (before[k], after[k]) for k in before if before[k] != after[k]})
