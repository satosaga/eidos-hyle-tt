"""
Guards against the class of doc-rot bug a 2026-09 documentation audit
found repeatedly by hand: a file-path reference in prose that no longer
points anywhere, because the referenced file was renamed, moved, or
deleted after the prose was written.

Two checks, one per direction code<->doc cross-references are allowed
to go (see docs/ARCHITECTURE.md's "Where documentation lives"):
docstrings/comments in src/ may never reference a docs/*.md file at
all, in either direction of granularity -- that direction is banned
outright, not just kept accurate, so this test looks for any such
reference rather than checking whether it still resolves. docs/*.md
files (and README.md/CONTRIBUTING.md/SECURITY.md) may freely reference
source/config file paths by name, but those paths must still exist.

Matches by path suffix rather than requiring an exact repo-relative
path, since prose already writes the same file at different lengths
depending on context (e.g. `core/schema.py` and `src/core/schema.py`
both appear across these files for the same file) -- a suffix match
accepts either without hardcoding which form a given doc uses.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DOC_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "SECURITY.md",
    *sorted((REPO_ROOT / "docs").glob("*.md")),
]

_DOCS_MD_PATTERN = re.compile(r"docs/\s*[A-Za-z0-9_-]+\.md")

_PATH_BACKTICK_PATTERN = re.compile(
    r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|md|sh|json|toml|ya?ml|html|cfg|ini))`"
)

# Looks like a file path but is not one -- an HTTP route served by
# hyle.apps.fit2gpx_converter's local server (see docs/ARCHITECTURE.md's
# "FIT parsing happens once, in Python"), not a file on disk.
_NON_PATH_BACKTICK_STRINGS = {"/parsed.json"}

_EXCLUDED_DIR_NAMES = {"__pycache__", "node_modules", "temp"}


def _is_excluded(path: Path) -> bool:
    for part in path.relative_to(REPO_ROOT).parts:
        if part.startswith(".") or part in _EXCLUDED_DIR_NAMES or part.startswith("env3") or part.endswith(".egg-info"):
            return True
    return False


def _strip_comment_markers(text: str) -> str:
    """Join `# `-prefixed comment lines into one continuous string so a
    docs/*.md reference split across a line-wrapped comment (e.g. "...docs/\n#
    ARCHITECTURE.md...") is still found -- a plain regex over the raw text
    would miss it, since the comment marker sits between "docs/" and the
    filename after the line break."""
    return "\n".join(re.sub(r"^\s*#\s?", "", line) for line in text.splitlines())


def _all_repo_file_suffixes() -> list[str]:
    return [
        p.relative_to(REPO_ROOT).as_posix()
        for p in REPO_ROOT.rglob("*")
        if p.is_file() and not _is_excluded(p)
    ]


def test_no_docs_md_references_in_src():
    """Docstrings/comments in src/ must never reference a docs/*.md file --
    see docs/ARCHITECTURE.md's "Where documentation lives". Seven such
    references were found and removed during a 2026-09 documentation
    audit; this test keeps them from creeping back in."""
    offenders = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        if _DOCS_MD_PATTERN.search(_strip_comment_markers(path.read_text())):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, (
        f"{offenders} reference a docs/*.md file -- docstrings/comments "
        "must not point into docs/, see docs/ARCHITECTURE.md's "
        '"Where documentation lives".'
    )


def test_doc_file_path_references_exist():
    """Every backtick-quoted, slash-containing file-path reference inside
    README.md/CONTRIBUTING.md/SECURITY.md/docs/*.md must resolve to a
    real file somewhere in the repo. Bare filenames with no directory
    component (e.g. `module.py`) are skipped -- those are frequently a
    generic illustrative example in prose, not a reference to one
    specific file, and checking them would mean matching against every
    same-named file in the repo (including third-party ones) rather than
    a real path."""
    all_suffixes = _all_repo_file_suffixes()

    broken = set()
    for doc in DOC_FILES:
        text = doc.read_text()
        for match in _PATH_BACKTICK_PATTERN.finditer(text):
            candidate = match.group(1)
            if candidate in _NON_PATH_BACKTICK_STRINGS or "/" not in candidate:
                continue
            candidate = candidate.lstrip("/")
            if not any(s == candidate or s.endswith("/" + candidate) for s in all_suffixes):
                broken.add(f"{doc.relative_to(REPO_ROOT)}: `{candidate}`")

    assert not broken, "Broken file-path reference(s) found:\n" + "\n".join(sorted(broken))
