"""Identify Helvetica.ttc and test other output formats with matplotlib 3.11 (temporary)."""

import hashlib
import os
import platform
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fontTools.ttLib import TTCollection  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402

PATH = "/System/Library/Fonts/Helvetica.ttc"
print("matplotlib", matplotlib.__version__, "macOS", platform.mac_ver()[0])
found = findfont(FontProperties(family=["Helvetica"]))
print("findfont:", str(found), "face_index:", getattr(found, "face_index", "n/a"))
print("sha256:", hashlib.sha256(open(PATH, "rb").read()).hexdigest())
for i, face in enumerate(TTCollection(PATH, lazy=True).fonts):
    name = face["name"]
    tables = {t: len(face.reader[t]) for t in ("fpgm", "prep", "cvt ") if t in face}
    print(f"face {i}: {name.getDebugName(1)!r} {name.getDebugName(2)!r},"
          f" version {name.getDebugName(5)!r}, head.fontRevision {face['head'].fontRevision:.4f},"
          f" gasp {getattr(face.get('gasp'), 'gaspRange', None)}, bytes {tables}")

tmp = tempfile.mkdtemp()
for ext in ("png", "pdf", "svg", "ps", "eps"):
    fig = plt.figure(figsize=(2, 1))
    fig.text(0.1, 0.5, "yield", fontfamily="Helvetica", fontsize=5)
    try:
        fig.savefig(os.path.join(tmp, "out." + ext), dpi=72)
        print(f"savefig .{ext} at 72 dpi: ok")
    except Exception as e:  # noqa: BLE001
        print(f"savefig .{ext} at 72 dpi: {type(e).__name__}: {e}")
    finally:
        plt.close(fig)
