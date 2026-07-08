# Contributing to depictr

Thank you for considering a contribution. Issues and pull requests are both
welcome, whether they fix a bug, improve the documentation or add a plot.

## Reporting a problem or suggesting a feature

Please open an issue at <https://github.com/pablobernabeu/depictr-py/issues>. A
small, self-contained example helps a great deal, and for a plotting bug a
screenshot of the result is useful too.

## Setting up for development

Clone the repository and install the package in editable mode with the back-ends
and the test and docs tools:

```bash
pip install -e ".[all,test,docs]"
```

```bash
pytest                    # run the test suite
mkdocs build --strict     # build the docs (runs the gallery examples)
```

## Conventions

Every plotting function returns a [plotnine](https://plotnine.org) object (or a
composition for multi-panel figures), so a plot can be extended with the usual
`+` syntax. New or changed plots should keep the shared theme (`theme_depictr`)
and the colourblind-safe Okabe-Ito palette, and where a specialist package
computes a quantity well, depictr delegates to it and redraws the result rather
than re-implementing it. The gallery guides in `docs/gallery/` render their plots
at build time, so a new family is easiest to review with a guide added there.

## Submitting a pull request

Base your work on `main`, keep the change focused, and add or update tests and
documentation alongside the code. Running `pytest` and `mkdocs build --strict`
before opening the pull request saves a round trip.

By contributing you agree that your contribution is licensed under the same MIT
licence as the package, and that you will follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
