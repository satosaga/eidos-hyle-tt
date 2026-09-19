"""
Smoke tests for console_scripts wiring (pyproject.toml [project.scripts]).

These tests parse pyproject.toml directly instead of hard-coding the
command list, so they can never silently drift from the actual entry-point
table: if a script is added, renamed, or its target moves, this test picks
it up automatically without needing a matching edit here.

This guards against the class of bug seen during the 2026 directory
refactor: a module gets split into a package or renamed, and pyproject.toml
(or some other lookup table) keeps pointing at the old target while
everything still *looks* fine because nothing actually tried to resolve it.

Requires the project's real runtime dependencies (PySide6, fit_tool, etc.)
to be installed, since importing an app module imports its GUI stack too.
Run inside the project's own venv:

    pip install -e ".[dev]"
    pytest
"""

import importlib
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - project requires >=3.12, kept as a fallback only
    import tomli as tomllib

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _load_scripts() -> dict[str, str]:
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    return data["project"]["scripts"]


SCRIPTS = _load_scripts()


@pytest.mark.parametrize("command, target", sorted(SCRIPTS.items()))
def test_entry_point_resolves(command, target):
    """Each `command = "module:attr"` entry must import and expose a callable."""
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    assert hasattr(module, attr), (
        f"{command}: {module_name!r} has no attribute {attr!r} "
        f"-- pyproject.toml entry point is stale"
    )
    func = getattr(module, attr)
    assert callable(func), f"{command}: {target} is not callable"


def test_expected_number_of_apps_present():
    """Guard against silently losing (or duplicating) an app during future edits."""
    assert len(SCRIPTS) == 13, (
        f"expected 13 console_scripts (8 eidos.* + 5 hyle.*), "
        f"found {len(SCRIPTS)}: {sorted(SCRIPTS)}"
    )
