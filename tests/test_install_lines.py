"""Install commands that paste into any shell.

Readers copy the install commands from the README, the guides and the messages the package
raises, and paste them into whatever shell is open. zsh, the default shell for new accounts
on macOS since 10.15, reads ``depictr[all]`` as a filename pattern and, finding no file to
match, stops with "no matches found" before pip runs. Double quotes keep the brackets
literal in zsh, bash, fish, PowerShell and cmd alike, whereas cmd passes single quotes on to
pip. An interactive zsh also leaves comments off by default, so a pasted
``pip install depictr  # core`` hands ``#`` and the words after it to pip, which stops with
"Invalid requirement: '#'". An install block therefore holds bare commands, and the prose
before it explains them.

The first two tests pin the two rules on examples. The third applies them to the README,
the contributing guide, the documentation, the sources and the app.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# `pip install`, `pip3 install` and `python -m pip install` alike.
PIP_INSTALL = re.compile(r"\bpip3?\s+install\b")
# One shell word: a double-quoted or single-quoted string, or a run of characters holding
# no quote and no space. A backtick closes inline code, and with it the command.
WORD = re.compile(r"""[ \t]*("[^"\n]*"|'[^'\n]*'|[^\s"'`]+)""")
# The opening line of a fenced block and the first word of its info string.
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*\{?\.?([\w-]*)")
SHELLS = {"bash", "sh", "shell", "console", "zsh"}


def unquoted_extras(line: str) -> list[str]:
    """The words of each pip install command in a line that hold a bracket outside double quotes.

    A command ends at the end of the line, at a backtick, a shell operator or a comment. It
    also ends after a word closing with a comma, a semicolon or a parenthesis, as a command
    quoted in prose does.
    """
    found = []
    for command in PIP_INSTALL.finditer(line):
        pos = command.end()
        while word := WORD.match(line, pos):
            text, pos = word.group(1), word.end()
            if text in {"&&", "||", "|", ";"} or text.startswith("#"):
                break
            if "[" in text and not text.startswith('"'):
                found.append(text)
            if text[-1] in ",;)":
                break
    return found


def commented_install_lines(markdown: str) -> list[tuple[int, str]]:
    """The pip install lines in a page's shell blocks that carry a ``#`` comment, numbered."""
    found = []
    fence, shell = "", False
    for number, line in enumerate(markdown.splitlines(), start=1):
        if not fence:
            opened = FENCE.match(line)
            if opened:
                fence, shell = opened.group(1), opened.group(2).lower() in SHELLS
            continue
        closing = line.strip()
        if closing and set(closing) == {fence[0]} and len(closing) >= len(fence):
            fence = ""
        elif shell and PIP_INSTALL.search(line) and re.search(r"\s#", line):
            found.append((number, line.strip()))
    return found


def _sources() -> list[Path]:
    files = [ROOT / "README.md", ROOT / ".github" / "CONTRIBUTING.md"]
    files += sorted((ROOT / "docs").rglob("*.md"))
    files += sorted((ROOT / "src").rglob("*.py"))
    files += sorted((ROOT / "app").rglob("*.py"))
    return [path for path in files if path.is_file()]


@pytest.mark.parametrize(
    ("line", "words"),
    [
        ("pip install depictr", []),
        ("pip install depictr[all]", ["depictr[all]"]),
        ('pip install "depictr[all]"', []),
        ("pip install 'depictr[survival]'", ["'depictr[survival]'"]),
        ("python -m pip install -e .[app]", [".[app]"]),
        ('pip install -e ".[all,test,docs]"', []),
        ("pip install git+https://github.com/pablobernabeu/depictr-py", []),
        ('pip install "depictr[app]" && streamlit run app.py --x "a[b]"', []),
        ("pip install depictr  # add [all] for the back-ends", []),
        ("Install it with ``pip install depictr[survival]``.", ["depictr[survival]"]),
        ('"`pip install depictr[models]`."', ["depictr[models]"]),
        ("'`pip install \"depictr[models]\"`.'", []),
        ('which `pip install "depictr[classification]"` installs, as [the guide](a.md) says', []),
        ("an editable install (pip install -e '.[app]').", ["'.[app]'"]),
        ('an editable install (pip install -e ".[app]").', []),
        ("after ``pip install`` and with no clone", []),
    ],
)
def test_an_extra_outside_double_quotes_is_found(line, words):
    assert unquoted_extras(line) == words


def test_a_comment_on_an_install_line_in_a_shell_block_is_found():
    page = "\n".join([
        "Prose may say `pip install depictr  # anything`, since it is not in a block.",
        "```bash",
        "pip install depictr",
        "pip install depictr            # core (plotnine, pandas, numpy, matplotlib, scipy)",
        "# A comment on a line of its own reaches no pip command.",
        "```",
        "```python",
        "x = 1  # pip install lines in Python code are not pasted into a shell",
        "```",
        "```console",
        '$ pip install "depictr[all]" # the prompt\'s comment',
        "```",
        "    ```sh",
        '    pip install -e ".[app]"',
        "    ```",
        "````markdown",
        "```bash",
        "pip install depictr  # quoted inside a Markdown example",
        "```",
        "````",
    ])
    assert [number for number, _ in commented_install_lines(page)] == [4, 11]


def test_install_lines_paste_into_any_shell():
    sources = _sources()
    assert ROOT / "README.md" in sources
    assert ROOT / "docs" / "getting-started.md" in sources
    assert ROOT / "src" / "depictr" / "survival.py" in sources
    offences = []
    for path in sources:
        name = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            offences += [f"{name}:{number}: {word} needs double quotes"
                         for word in unquoted_extras(line)]
        if path.suffix == ".md":
            offences += [f"{name}:{number}: a comment on the line reaches pip: {line}"
                         for number, line in commented_install_lines(text)]
    assert offences == []
