"""Documentation files that nothing else in the suite reads.

The README's table of exported functions, the version in the site header, the
forked header template of the site and the comments in its stylesheet are
written by hand, so an export left out of the table, a header naming the
release before, a template fallen behind the mkdocs-material it was taken from,
or a contrast ratio quoted from a sibling package's colours would pass every
other check.
"""

import hashlib
import re
from pathlib import Path

import pytest

import depictr

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
BRAND_CSS = REPO / "docs" / "stylesheets" / "brand.css"
HEADER = REPO / "overrides" / "partials" / "header.html"
DOCS_WORKFLOW = REPO / ".github" / "workflows" / "docs.yml"
MKDOCS = REPO / "mkdocs.yml"
FORK_GUARD = "Check the header override against the mkdocs-material it forked"


def _repository_file(path):
    # The documentation and the workflows travel with the repository, not with
    # an installed distribution.
    if not path.exists():
        pytest.skip(f"{path.name} is not part of an installed distribution")
    return path.read_text(encoding="utf-8")


def test_the_readme_table_of_exported_functions_lists_every_export():
    # The table is introduced as the exported functions, and it had left out
    # the dataset loaders, DATASETS, depictr_brand, depictr_accent and the
    # scale_color_depictr spelling.
    text = _repository_file(README)
    start = text.index("## Function families")
    table = text[start:text.index("## Status", start)]
    listed = set(re.findall(r"`(\w+)`", table))
    exported = {name for name in depictr.__all__ if name != "__version__"}
    assert not exported - listed, f"exported but unlisted: {sorted(exported - listed)}"
    assert not listed - exported, f"listed but not exported: {sorted(listed - exported)}"


def test_the_site_header_names_the_version_the_package_reports():
    # extra.version in mkdocs.yml fills the version chip beside the title. It
    # is bumped by hand on release, where the About page takes its version from
    # the package, so a missed bump would put the release before in the header.
    # Read as text, so the suite needs no YAML parser.
    chip = re.search(r'^extra:\n  version: "([^"]+)"$', _repository_file(MKDOCS), re.M)
    assert chip, "mkdocs.yml has no extra.version"
    assert chip.group(1) == depictr.__version__


def _fork_record():
    """The mkdocs-material version and upstream hash recorded for the fork.

    tomllib arrived in Python 3.11 and the suite runs from 3.10, so the record
    is read as text here. The deploy workflow parses it with tomllib.
    """
    text = _repository_file(REPO / "pyproject.toml")
    start = text.find("[tool.depictr.mkdocs-material-fork]")
    assert start != -1, "pyproject.toml has no [tool.depictr.mkdocs-material-fork] table"
    section = re.split(r"\n(?=\[)", text[start:], maxsplit=1)[0]
    version = re.search(r'^version = "([^"]+)"$', section, re.M)
    digest = re.search(r'^partials_header_sha256 = "([0-9a-f]{64})"$', section, re.M)
    assert version, "the fork records no mkdocs-material version"
    assert digest, "the fork records no hash of the upstream partial"
    return version.group(1), digest.group(1)


def test_the_header_fork_records_the_version_its_comment_names():
    version, _ = _fork_record()
    assert f"mkdocs-material {version}'s partials/header.html" in _repository_file(HEADER)


def test_the_docs_workflow_checks_the_fork_before_it_builds():
    # The record is of use only when the deploy compares it with the template
    # it installs, before the site is built. Read as text, so the suite needs
    # no YAML parser.
    workflow = _repository_file(DOCS_WORKFLOW)
    guard = workflow.find(FORK_GUARD)
    assert guard != -1, f"the docs workflow has no '{FORK_GUARD}' step"
    build = workflow.find("python -m mkdocs build --strict")
    assert build != -1, "the docs workflow no longer builds the site"
    assert guard < build, "the fork is checked only after the site is built"
    end = workflow.find("\n      - name:", guard)
    step = workflow[guard:end if end != -1 else len(workflow)]
    assert '["tool"]["depictr"]["mkdocs-material-fork"]' in step
    assert "templates/partials/header.html" in step


def test_the_recorded_hash_matches_the_installed_partial():
    # Runs only where mkdocs-material is installed, as in the docs job and
    # with the docs extra. The test jobs of CI install neither.
    material = pytest.importorskip("material")
    version, digest = _fork_record()
    upstream = Path(material.__file__).parent / "templates" / "partials" / "header.html"
    actual = hashlib.sha256(upstream.read_bytes()).hexdigest()
    assert actual == digest, f"partials/header.html has changed since mkdocs-material {version}"


def _luminance(hex_colour):
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def test_the_dark_mode_button_comment_quotes_this_site_s_contrast():
    # The comment came with the shared stylesheet and quoted the 2.9:1 of the
    # library's teal. depictr's primary blue on its own navy page is 2.4:1.
    css = _repository_file(BRAND_CSS)
    quoted = float(re.search(r"too dark on the navy page \(~([\d.]+):1\)", css).group(1))
    primary = re.search(r"--brand-primary:\s*#([0-9A-Fa-f]{6})", css).group(1)
    slate = re.search(r'\[data-md-color-scheme="slate"\]\s*\{(.*?)\n\}', css, re.S).group(1)
    page = re.search(r"--md-default-bg-color:\s*#([0-9A-Fa-f]{6})", slate).group(1)
    lighter, darker = sorted((_luminance(primary), _luminance(page)), reverse=True)
    assert round((lighter + 0.05) / (darker + 0.05), 1) == quoted
