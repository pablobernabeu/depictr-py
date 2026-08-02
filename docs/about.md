# About

## Citing depictr

<!-- The citation is built when the site is built, from the version the package
     itself reports, so it cannot drift behind the code the way a hard-coded
     string does. The rendered reference, the BibTeX entry and the Download .bib
     link all come from one string, and the link is that same string
     percent-encoded into a data URI, so the download needs no .bib file shipped
     beside the site and cannot disagree with the entry shown above it. Two
     copies of the version stay out of reach here and still have to be bumped by
     hand on release, since neither file can execute code. One is
     `extra.version` in mkdocs.yml, which fills the version chip in the header,
     and the other is the `version` field of CITATION.cff. -->

If depictr helps your work, a citation is appreciated:

```` python exec="1"
from urllib.parse import quote

from depictr import __version__

entry = f"""@Manual{{depictr-py,
  title  = {{depictr: A unified, colourblind-safe toolkit for publication-ready statistical visualisation (Python)}},
  author = {{Pablo Bernabeu}},
  year   = {{2026}},
  note   = {{Python package version {__version__}}},
  doi    = {{10.5281/zenodo.21266311}},
  url    = {{https://doi.org/10.5281/zenodo.21266311}},
}}"""

print("> Bernabeu, P. (2026). *depictr: A unified, colourblind-safe toolkit for")
print(f"> publication-ready statistical visualisation* (Python). Version {__version__}.")
print("> <https://doi.org/10.5281/zenodo.21266311>")
print()

# This block opens on four backticks so that the three printed here close the
# BibTeX listing and nothing else. The listing is printed as Markdown rather
# than as ready-made HTML because that is what earns it syntax highlighting and
# a copy button from the theme.
print("```bibtex")
print(entry)
print("```")
print()

print('<p><a download="depictr-py.bib" href="data:application/x-bibtex;'
      f'charset=utf-8,{quote(entry, safe="")}">Download .bib</a></p>')
````

The repository's
[`CITATION.cff`](https://github.com/pablobernabeu/depictr-py/blob/main/CITATION.cff)
carries the same metadata in machine-readable form.

## The developer

depictr is developed by **Pablo Bernabeu**, a researcher in the Department of
Education at the University of Oxford. His work spans cognitive psychology and
neuroscience, linguistics, education and digital technologies, drawing on a
range of methods that include behavioural and EEG experiments, corpus analysis
and computational modelling. He is a Fellow of the Software Sustainability
Institute (2020), recognised for his work on R-based tools for data
presentation, and holds a PhD in Psychology from Lancaster University.

More about his work is at [pablobernabeu.github.io](https://pablobernabeu.github.io),
on [GitHub](https://github.com/pablobernabeu) and via
[ORCID 0000-0003-1083-2460](https://orcid.org/0000-0003-1083-2460).

depictr has a sibling [R package](https://pablobernabeu.github.io/depictr/) that
shares the same design.

## Licence

depictr is released under the MIT licence, whose full text is on this site's
[licence page](licence.md) and in the repository's
[`LICENSE`](https://github.com/pablobernabeu/depictr-py/blob/main/LICENSE) file.
The licence permits use, modification and redistribution, commercially or not,
provided the copyright and permission notice travel with the code.

## Versioning and archival

Releases are tagged on
[GitHub](https://github.com/pablobernabeu/depictr-py/releases) and archived on
Zenodo. The concept DOI,
[10.5281/zenodo.21266311](https://doi.org/10.5281/zenodo.21266311), always
resolves to the latest archived version, so a citation that uses it does not go
stale. The [changelog](changelog.md) records what changed in each release.

## Contributing and support

Bug reports and feature requests are welcome on the
[issues page](https://github.com/pablobernabeu/depictr-py/issues), where a
small, self-contained example (and, for a plotting bug, a screenshot) makes a
problem much easier to act on. The
[contributing guide](https://github.com/pablobernabeu/depictr-py/blob/main/.github/CONTRIBUTING.md)
explains how to set up a development install and what conventions a pull
request should follow.
