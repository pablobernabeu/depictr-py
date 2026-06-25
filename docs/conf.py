"""Sphinx configuration for the depictr documentation site.

Builds three things, pkgdown-style: an API reference generated from the
docstrings (autosummary + napoleon), a gallery of worked examples whose plots
are rendered at build time (sphinx-gallery with a custom plotnine scraper), and
the narrative pages written in Markdown (myst-parser).
"""

import matplotlib

matplotlib.use("Agg")  # render figures headlessly during the build

project = "depictr"
author = "Pablo Bernabeu"
copyright = "2026, Pablo Bernabeu"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]

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
}

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
    "remove_config_comments": True,
    "matplotlib_animations": False,
    "download_all_examples": False,
}
