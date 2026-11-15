"""FreeType error code and interpreter version for the Helvetica.ttc failure (temporary).

Usage: python diag/ft_versions.py            run every section
       python diag/ft_versions.py mpl        matplotlib section only (used in subprocesses)
       python diag/ft_versions.py lib PATH   call the FreeType library at PATH through ctypes
"""

import ctypes
import os
import shutil
import subprocess
import sys

FONT = os.environ.get("DIAG_FONT", "/System/Library/Fonts/Helvetica.ttc")
CHAR = "y"
FLAGS = {"DEFAULT": 0x0, "NO_HINTING": 0x2, "FORCE_AUTOHINT": 0x20,
         "NO_AUTOHINT": 0x8000, "TARGET_LIGHT": 0x10000}


def mpl_section():
    import matplotlib
    import matplotlib.ft2font as ft

    print("matplotlib", matplotlib.__version__, "freetype", ft.__freetype_version__,
          "| FREETYPE_PROPERTIES =", os.environ.get("FREETYPE_PROPERTIES"))
    for face_index in range(64):
        try:
            font = ft.FT2Font(FONT, face_index=face_index)
        except Exception:  # noqa: BLE001
            break
        res = []
        for ppem in (5, 7, 8):
            font.set_size(ppem, 72)
            gid = font.get_char_index(ord(CHAR))
            try:
                font.load_glyph(gid, flags=ft.LoadFlags.DEFAULT)
                res.append(f"{ppem}px ok")
            except Exception as e:  # noqa: BLE001
                res.append(f"{ppem}px {type(e).__name__}: {e}")
        print(f"  face {face_index} {font.family_name!r} {font.style_name!r}:",
              " | ".join(res))


def lib_section(path):
    lib = ctypes.CDLL(path)
    vp = ctypes.c_void_p
    lib.FT_Init_FreeType.argtypes = [ctypes.POINTER(vp)]
    lib.FT_Library_Version.argtypes = [vp] + [ctypes.POINTER(ctypes.c_int)] * 3
    lib.FT_Property_Set.argtypes = [vp, ctypes.c_char_p, ctypes.c_char_p, vp]
    lib.FT_New_Face.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, ctypes.POINTER(vp)]
    lib.FT_Set_Char_Size.argtypes = [vp, ctypes.c_long, ctypes.c_long,
                                     ctypes.c_uint, ctypes.c_uint]
    lib.FT_Get_Char_Index.argtypes = [vp, ctypes.c_ulong]
    lib.FT_Get_Char_Index.restype = ctypes.c_uint
    lib.FT_Load_Glyph.argtypes = [vp, ctypes.c_uint, ctypes.c_int32]
    lib.FT_Done_Face.argtypes = [vp]
    lib.FT_Done_FreeType.argtypes = [vp]

    for interp in (None, 35, 40):
        library = vp()
        assert lib.FT_Init_FreeType(ctypes.byref(library)) == 0
        v = [ctypes.c_int() for _ in range(3)]
        lib.FT_Library_Version(library, *[ctypes.byref(x) for x in v])
        prop = "default"
        if interp is not None:
            val = ctypes.c_uint(interp)
            rc = lib.FT_Property_Set(library, b"truetype", b"interpreter-version",
                                     ctypes.cast(ctypes.byref(val), vp))
            prop = f"v{interp} (FT_Property_Set rc=0x{rc:02x})"
        face = vp()
        assert lib.FT_New_Face(library, FONT.encode(), 0, ctypes.byref(face)) == 0
        fails = []
        for ppem in range(1, 16):
            assert lib.FT_Set_Char_Size(face, 0, ppem * 64, 72, 72) == 0
            gid = lib.FT_Get_Char_Index(face, ord(CHAR))
            for name, flag in FLAGS.items():
                err = lib.FT_Load_Glyph(face, gid, flag)
                if err:
                    fails.append(f"{ppem}:{name}:0x{err:02x}")
        print(f"  FreeType {v[0].value}.{v[1].value}.{v[2].value}, interpreter {prop}:",
              fails or "no failures")
        lib.FT_Done_Face(face)
        lib.FT_Done_FreeType(library)


def main():
    print("python", sys.version.split()[0])
    print("\n== matplotlib FT2Font.load_glyph('y'), LoadFlags.DEFAULT, each face of Helvetica.ttc")
    for props in (None, "truetype:interpreter-version=35",
                  "truetype:interpreter-version=40"):
        env = dict(os.environ)
        env.pop("FREETYPE_PROPERTIES", None)
        if props:
            env["FREETYPE_PROPERTIES"] = props
        out = subprocess.run([sys.executable, __file__, "mpl"], env=env,
                             capture_output=True, text=True)
        print(out.stdout + out.stderr)

    print("== FreeType called directly through ctypes, 'y' in face 0, ppem 1 to 15;"
          " failures as ppem:flag:error")
    libs = []
    try:
        from importlib.metadata import version

        import freetype
        pkg = os.path.dirname(freetype.__file__)
        bundled = [f for f in os.listdir(pkg) if f.startswith("libfreetype")]
        libs.append(("freetype-py " + version("freetype-py"), os.path.join(pkg, bundled[0])))
    except ImportError:
        print("  freetype-py not installed")
    brew = shutil.which("brew") and subprocess.run(
        ["brew", "--prefix", "freetype"], capture_output=True, text=True)
    if brew and brew.returncode == 0:
        libs.append(("Homebrew", os.path.join(brew.stdout.strip(), "lib", "libfreetype.dylib")))
    for label, path in libs:
        print(f" {label}: {path}")
        out = subprocess.run([sys.executable, __file__, "lib", path],
                             capture_output=True, text=True)
        print(out.stdout + out.stderr)


if __name__ == "__main__":
    if sys.argv[1:2] == ["mpl"]:
        mpl_section()
    elif sys.argv[1:2] == ["lib"]:
        lib_section(sys.argv[2])
    else:
        main()
