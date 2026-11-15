"""matplotlib-only reproduction of 'failed to load glyph' (temporary, never merged)."""

import os
import platform
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.ft2font as ft2font  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont, get_font  # noqa: E402

print("python", sys.version.split()[0], platform.platform())
print("matplotlib", matplotlib.__version__,
      "freetype", getattr(ft2font, "__freetype_version__", "?"),
      getattr(ft2font, "__freetype_build_type__", "?"))
print("rcParams text.hinting:", matplotlib.rcParams["text.hinting"],
      "| text.hinting_factor:", matplotlib.rcParams["text.hinting_factor"])
print("findfont Helvetica ->", findfont(FontProperties(family=["Helvetica"])))
tmp = tempfile.mkdtemp()


def save(family, size, dpi, rc=None, text="yield"):
    try:
        with matplotlib.rc_context(rc or {}):
            fig = plt.figure(figsize=(2, 1))
            fig.text(0.1, 0.5, text, fontfamily=family, fontsize=size)
            try:
                fig.savefig(os.path.join(tmp, "x.png"), dpi=dpi)
            finally:
                plt.close(fig)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


print("\n== savefig, default rcParams")
for fam in ["Helvetica", "Helvetica Neue", "Arial", "Times", "DejaVu Sans"]:
    for size, dpi in [(5, 72), (5, 100), (7, 72), (8, 72), (10, 72), (5, 300)]:
        print(f"{fam:15s} {size:>2}pt {dpi:>3}dpi ({size * dpi / 72:5.2f} px/em):",
              save(fam, size, dpi))

print("\n== savefig Helvetica 5pt 72dpi under each text.hinting value")
for h in ["default", "no_autohint", "force_autohint", "no_hinting",
          "auto", "native", "either", "none"]:
    print(f"text.hinting={h!r}:", save("Helvetica", 5, 72, {"text.hinting": h}))

print("\n== savefig Helvetica 5pt 72dpi, single characters")
bad = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
       if save("Helvetica", 5, 72, text=c) != "ok"]
print("characters that fail:", "".join(bad) or "none")

print("\n== FT2Font.set_text on Helvetica.ttc, failing (px/em, flag)")
path = "/System/Library/Fonts/Helvetica.ttc"
flags = {}
if hasattr(ft2font, "LoadFlags"):
    flags = {n: getattr(ft2font.LoadFlags, n) for n in
             ["DEFAULT", "NO_HINTING", "FORCE_AUTOHINT", "NO_AUTOHINT",
              "TARGET_LIGHT"] if hasattr(ft2font.LoadFlags, n)}
fails = []
for ppem in range(1, 16):
    for name, flag in flags.items():
        font = get_font(path)
        try:
            font.set_size(ppem, 72)
            font.set_text("yield", 0.0, flags=flag)
        except RuntimeError:
            fails.append((ppem, name))
print(fails)
