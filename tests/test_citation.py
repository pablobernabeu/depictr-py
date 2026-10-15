"""The citation metadata that GitHub, Zenodo and reference managers read from CITATION.cff.

The author entry gave no ORCID iD, so Zenodo's record of 0.3.0 carries none, while the R
twin's CITATION.cff and its Zenodo record give it. A version is dated by the day it first
became public on any channel, a PyPI upload or a GitHub release, and the comment above the
date records each channel's day. The changelog heading of the cited release gives the same
date. These tests read the repository around the package and skip when a file is not in the
tree.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CITATION = ROOT / "CITATION.cff"
CHANGELOG = ROOT / "CHANGELOG.md"
ORCID = "https://orcid.org/0000-0003-1083-2460"
DAY = re.compile(r"\d{4}-\d{2}-\d{2}")

# The earliest day each released version was public on any channel.
FIRST_PUBLIC = {"0.3.0": "2026-09-24"}


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        pytest.skip(f"{path.name} is not in this tree")
    return path.read_text(encoding="utf-8").splitlines()


def _field(lines: list[str], name: str) -> str:
    values = [line.split(":", 1)[1].strip().strip('"') for line in lines
              if line.startswith(f"{name}:")]
    assert len(values) == 1, f"CITATION.cff has {len(values)} {name} fields"
    return values[0]


def _comment_above(lines: list[str], name: str) -> str:
    """The comment lines directly above a field, joined into one string."""
    at = next(i for i, line in enumerate(lines) if line.startswith(f"{name}:"))
    comment: list[str] = []
    for line in reversed(lines[:at]):
        if not line.startswith("#"):
            break
        comment.insert(0, line)
    return " ".join(comment)


def test_the_author_carries_the_orcid_id():
    lines = _lines(CITATION)
    start = lines.index("authors:")
    entry = []
    for line in lines[start + 1:]:
        if not line.startswith(" "):
            break
        entry.append(line.strip())
    assert f'orcid: "{ORCID}"' in entry, entry


def test_the_release_is_dated_by_the_day_it_first_became_public():
    lines = _lines(CITATION)
    version = _field(lines, "version")
    released = _field(lines, "date-released")
    comment = _comment_above(lines, "date-released")
    assert "GitHub release" in comment and "PyPI" in comment, comment
    days = sorted(DAY.findall(comment))
    assert days, f"the comment above date-released records no day: {comment!r}"
    assert released == days[0], f"date-released is {released}, the earliest day recorded {days[0]}"
    if version in FIRST_PUBLIC:
        assert released == FIRST_PUBLIC[version]


def test_the_changelog_dates_the_cited_release_as_citation_cff_does():
    lines = _lines(CITATION)
    version = _field(lines, "version")
    released = _field(lines, "date-released")
    text = "\n".join(_lines(CHANGELOG))
    heading = re.search(rf"^## \[{re.escape(version)}\] - (\S+)[ \t]*$", text, flags=re.M)
    assert heading, f"CHANGELOG.md has no dated heading for {version}"
    assert heading.group(1) == released, (
        f"CHANGELOG.md dates {version} {heading.group(1)}, CITATION.cff {released}"
    )
