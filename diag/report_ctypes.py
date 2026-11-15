import ctypes

# FreeType from Homebrew (brew install freetype); any libfreetype build will do.
lib = ctypes.CDLL("/opt/homebrew/lib/libfreetype.dylib")
vp = ctypes.c_void_p
lib.FT_Init_FreeType.argtypes = [ctypes.POINTER(vp)]
lib.FT_Library_Version.argtypes = [vp] + [ctypes.POINTER(ctypes.c_int)] * 3
lib.FT_New_Face.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, ctypes.POINTER(vp)]
lib.FT_Set_Char_Size.argtypes = [vp, ctypes.c_long, ctypes.c_long, ctypes.c_uint, ctypes.c_uint]
lib.FT_Get_Char_Index.argtypes = [vp, ctypes.c_ulong]
lib.FT_Get_Char_Index.restype = ctypes.c_uint
lib.FT_Load_Glyph.argtypes = [vp, ctypes.c_uint, ctypes.c_int32]

library, face = vp(), vp()
lib.FT_Init_FreeType(ctypes.byref(library))
v = [ctypes.c_int() for _ in range(3)]
lib.FT_Library_Version(library, *[ctypes.byref(x) for x in v])
print("FreeType", ".".join(str(x.value) for x in v))
lib.FT_New_Face(library, b"/System/Library/Fonts/Helvetica.ttc", 0, ctypes.byref(face))
for hres in (72, 576):
    for ppem in (5, 7, 8):
        lib.FT_Set_Char_Size(face, ppem * 64, 0, hres, 72)
        err = lib.FT_Load_Glyph(face, lib.FT_Get_Char_Index(face, ord("y")), 0)  # FT_LOAD_DEFAULT
        print(f"horizontal {hres} dpi, vertical 72 dpi, {ppem} px/em: error 0x{err:02x}")
