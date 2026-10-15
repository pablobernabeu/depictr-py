"""The packaging declarations that shape the core metadata PyPI shows."""

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, where pytest brings tomli
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject():
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def test_the_build_backend_carries_no_ceiling(pyproject):
    # hatchling was held below 1.32 while twine could not read the Metadata-Version
    # 2.5 that 1.32 emits. twine 7 reads it, and a ceiling left in place only
    # freezes the backend and invites a pull request to move it each release.
    requires = pyproject["build-system"]["requires"]
    hatchling = [r for r in requires if re.match(r"hatchling\b", r)]
    assert hatchling == ["hatchling"]


def test_the_licence_is_an_spdx_expression_with_its_file_named(pyproject):
    # A table pointing at the file wrote the whole licence text into the
    # metadata's License field. The SPDX expression is written as
    # License-Expression and the file is shipped as a licence file.
    project = pyproject["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert (ROOT / "LICENSE").is_file()


def test_no_licence_classifier_repeats_the_expression(pyproject):
    # PEP 639 deprecates licence classifiers and lets a build backend refuse
    # one beside a License-Expression.
    classifiers = pyproject["project"]["classifiers"]
    assert [c for c in classifiers if c.startswith("License ::")] == []
