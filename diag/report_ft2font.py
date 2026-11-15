from matplotlib.ft2font import FT2Font, LoadFlags

font = FT2Font("/System/Library/Fonts/Helvetica.ttc")
font.set_size(5, 72)
font.load_glyph(font.get_char_index(ord("y")), flags=LoadFlags.DEFAULT)
