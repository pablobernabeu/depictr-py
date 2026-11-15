"""Does the Helvetica.ttc failure follow the FreeType version or the hinting factor? (temporary)

matplotlib 3.10 sets sizes with a horizontal resolution of dpi * hinting_factor (8 by default),
matplotlib 3.11 with dpi alone. Both are tested at native hinting (LoadFlags.DEFAULT).
"""

import ctypes
import os
import shutil
import subprocess
import sys

FONT = os.environ.get("DIAG_FONT", "/System/Library/Fonts/Helvetica.ttc")
CHAR = "y"


def mpl_section():
    import matplotlib
    import matplotlib.ft2font as ft

    print("matplotlib", matplotlib.__version__, "freetype", ft.__freetype_version__)
    factors = (1, 8) if matplotlib.__version__.startswith("3.10") else (None,)
    for hf in factors:
        font = ft.FT2Font(FONT) if hf is None else ft.FT2Font(FONT, hf)
        fails = []
        for ppem in range(1, 16):
            font.set_size(ppem, 72)
            gid = font.get_char_index(ord(CHAR))
            try:
                font.load_glyph(gid, flags=ft.LoadFlags.DEFAULT)
            except RuntimeError as e:
                fails.append(f"{ppem}: {e}")
        label = "default (no factor)" if hf is None else hf
        print(f"  hinting_factor {label}, face {font.family_name!r} {font.style_name!r}:",
              fails or "no failures")


def lib_section(path):
    lib = ctypes.CDLL(path)
    vp = ctypes.c_void_p
    lib.FT_Init_FreeType.argtypes = [ctypes.POINTER(vp)]
    lib.FT_Library_Version.argtypes = [vp] + [ctypes.POINTER(ctypes.c_int)] * 3
    lib.FT_New_Face.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, ctypes.POINTER(vp)]
    lib.FT_Set_Char_Size.argtypes = [vp, ctypes.c_long, ctypes.c_long,
                                     ctypes.c_uint, ctypes.c_uint]
    lib.FT_Get_Char_Index.argtypes = [vp, ctypes.c_ulong]
    lib.FT_Get_Char_Index.restype = ctypes.c_uint
    lib.FT_Load_Glyph.argtypes = [vp, ctypes.c_uint, ctypes.c_int32]
    library = vp()
    assert lib.FT_Init_FreeType(ctypes.byref(library)) == 0
    v = [ctypes.c_int() for _ in range(3)]
    lib.FT_Library_Version(library, *[ctypes.byref(x) for x in v])
    face = vp()
    assert lib.FT_New_Face(library, FONT.encode(), 0, ctypes.byref(face)) == 0
    for hres in (72, 576):
        fails = []
        for ppem in range(1, 16):
            assert lib.FT_Set_Char_Size(face, ppem * 64, 0, hres, 72) == 0
            err = lib.FT_Load_Glyph(face, lib.FT_Get_Char_Index(face, ord(CHAR)), 0)
            if err:
                fails.append(f"{ppem}:0x{err:02x}")
        print(f"  FreeType {v[0].value}.{v[1].value}.{v[2].value}, FT_LOAD_DEFAULT,"
              f" horizontal resolution {hres} dpi, vertical 72:", fails or "no failures")


def main():
    print("python", sys.version.split()[0])
    print("\n== matplotlib FT2Font, 'y' in face 0, ppem 1 to 15, LoadFlags.DEFAULT")
    mpl_section()
    print("\n== FreeType through ctypes, 'y' in face 0, ppem 1 to 15; failures as ppem:error")
    libs = []
    try:
        import freetype
        pkg = os.path.dirname(freetype.__file__)
        libs.append(os.path.join(pkg, [f for f in os.listdir(pkg)
                                       if f.startswith("libfreetype")][0]))
    except ImportError:
        print("  freetype-py not installed")
    brew = shutil.which("brew") and subprocess.run(
        ["brew", "--prefix", "freetype"], capture_output=True, text=True)
    if brew and brew.returncode == 0:
        libs.append(os.path.join(brew.stdout.strip(), "lib", "libfreetype.dylib"))
    for path in libs:
        out = subprocess.run([sys.executable, __file__, "lib", path],
                             capture_output=True, text=True)
        print(out.stdout + out.stderr, end="")


if __name__ == "__main__":
    if sys.argv[1:2] == ["lib"]:
        lib_section(sys.argv[2])
    else:
        main()
