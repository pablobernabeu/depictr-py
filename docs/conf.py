"""Sphinx configuration for the depictr documentation site.

Builds three things, pkgdown-style: an API reference generated from the
docstrings (autosummary + napoleon), a gallery of worked examples whose plots
are rendered at build time (sphinx-gallery with a custom plotnine scraper), and
the narrative pages written in Markdown (myst-parser).
"""

import warnings

import matplotlib

matplotlib.use("Agg")  # render figures headlessly during the build

# Keep the benign "Removed N rows containing missing values" notices (the bundled
# data has deliberate missingness) out of the rendered gallery output.
from plotnine.exceptions import PlotnineWarning  # noqa: E402

warnings.filterwarnings("ignore", category=PlotnineWarning)

from depictr import __version__ as release  # noqa: E402  (single source of truth)

project = "depictr"
author = "Pablo Bernabeu"
copyright = "2026, Pablo Bernabeu"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

# Copy buttons on code blocks; do not copy prompts or example output.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

autosummary_generate = True
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
myst_enable_extensions = ["colon_fence"]

html_theme = "pydata_sphinx_theme"
html_title = "depictr"
html_theme_options = {
    "github_url": "https://github.com/pablobernabeu/depictr-py",
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.ico"

# No "Show Source" link or /_sources directory of raw .md/.rst files -- not
# useful for a docs site with no external contributors editing pages directly.
html_copy_source = False
html_show_sourcelink = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}


def _plotnine_scraper(block, block_vars, gallery_conf):
    """Capture plotnine plots and compositions created in an example block.

    plotnine does not register its figures with pyplot, so the stock matplotlib
    scraper misses them. This saves any plotnine object that appears in the
    example's namespace and has not been captured yet, in creation order.
    """
    from plotnine import ggplot
    from plotnine.composition import Compose
    from sphinx_gallery.scrapers import figure_rst

    image_path_iterator = block_vars["image_path_iterator"]
    paths = []
    for value in list(block_vars["example_globals"].values()):
        if isinstance(value, (ggplot, Compose)) and not getattr(
            value, "_depictr_scraped", False
        ):
            path = next(image_path_iterator)
            try:
                value.save(path, width=8, height=5, dpi=110, verbose=False)
            except Exception:  # noqa: BLE001 - skip anything that will not draw
                continue
            try:
                value._depictr_scraped = True
            except Exception:  # noqa: BLE001
                pass
            paths.append(path)
    return figure_rst(paths, gallery_conf["src_dir"])


sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "image_scrapers": (_plotnine_scraper,),
    # A bare plot object on the last line of a cell otherwise prints its
    # uninformative repr ("<plotnine.ggplot object at 0x...>") as output. The
    # figure itself comes from the image scraper; explicit print() output is
    # still shown. So suppress the last-expression repr capture.
    "capture_repr": (),
    "remove_config_comments": True,
    "matplotlib_animations": False,
    "download_all_examples": False,
}
