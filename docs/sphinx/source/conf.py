"""Sphinx configuration for (EIDOS/HYLE)^TT's API reference.

Generates docs from the docstrings already in src/core, src/eidos,
src/hyle as-is -- see docs/ARCHITECTURE.md and docs/RUNBOOK.md for the
hand-written design/operational docs this complements (not replaces).

Build (after `pip install -e ".[docs]"`):
    sphinx-apidoc -f -e -o docs/sphinx/source/api src
    sphinx-build -b html docs/sphinx/source docs/sphinx/_build/html

Both commands, and why the two-step apidoc-then-build sequence exists
(the api/ stub files are regenerated on demand, not committed), are in
docs/RUNBOOK.md.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# src/ holds core/, eidos/, hyle/ as sibling top-level packages (same
# layout mypy_path/explicit_package_bases in pyproject.toml's [tool.mypy]
# already assumes) -- autodoc has to import these for real, so it needs
# the same path on sys.path that an editable install would put there.
SRC_DIR = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_DIR))

project = "(EIDOS/HYLE)^TT"
copyright = "2026, Sato SAGA"
author = "Sato SAGA"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # parses Google/NumPy-style docstring sections,
                             # for the few docstrings written that way --
                             # freeform prose docstrings (the majority)
                             # render fine without it
    "sphinx.ext.viewcode",  # adds "view source" links next to each entry
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
# Several dataclasses/NamedTuples here (e.g. core.schema.PowerBlocks,
# core.activity_parser.FlatFITData) document their fields in a
# Google-style "Attributes:" docstring section. Napoleon's default
# rendering of "Attributes:" turns each item into its own
# `.. py:attribute::` directive -- a second, independent definition of
# the exact same name autodoc's own dataclass-field introspection
# already documents (autodoc treats a type-annotated class attribute as
# listable regardless of undoc-members), so every such field was
# reported twice ("duplicate object description"). napoleon_use_ivar
# renders "Attributes:" as inline `:ivar:` field-list text attached to
# the class's own docstring instead -- readable, but not a second
# indexable object -- which resolves the clash without needing
# undoc-members off (which would have hidden other, unrelated
# individually-undocumented members elsewhere in the codebase).
napoleon_use_ivar = True
# Pull parameter/return types from actual type annotations rather than
# requiring them to be duplicated in docstring text (:type:/:rtype:
# fields) -- this codebase's ongoing mypy adoption (pyproject.toml's
# [tool.mypy]) is exactly the source of truth this wants to read from.
autodoc_typehints = "description"

# --- Mock unavailable heavy/hardware dependencies ------------------------
# autodoc *imports* every module it documents (it's not just scraping
# text), so building this in an environment that doesn't have PySide6,
# numba, an ANT+ stack, etc. installed would otherwise crash outright.
# Only mock what's actually missing right now, rather than
# unconditionally mocking this whole list: this project's real dev venv
# (env312_arm64) has everything installed, and mocking an import that's
# actually available would degrade its documentation for no reason
# (e.g. PySide6 signal/enum introspection would show as a generic Mock
# instead of the real thing).
_POSSIBLY_MISSING = [
    "PySide6", "numba", "scipy", "pandas", "numpy", "pyarrow",
    "pydantic", "OpenGL", "cv2", "pytesseract", "openant", "fit_tool",
    "reportlab", "pyautogui", "tkinterdnd2", "matplotlib",
]
autodoc_mock_imports = []
for _mod in _POSSIBLY_MISSING:
    try:
        importlib.import_module(_mod)
    except ImportError:
        autodoc_mock_imports.append(_mod)

# Even with PySide6 genuinely installed and NOT mocked (the normal case
# in the real dev venv), a handful of eidos.lib/eidos.apps modules do
# class-body arithmetic on a PySide6 enum at import time (e.g.
# `Qt.ItemDataRole.UserRole + 1` in eidos.lib.record_model) -- when
# PySide6 *is* mocked (sandbox/CI without it installed), that arithmetic
# runs against a Mock object and raises at import time, which autodoc
# reports as a failed-to-import warning for that whole module. Real,
# known, sandbox-only noise: it does not happen against the real
# PySide6. Nothing to fix here -- see docs/RUNBOOK.md's Sphinx section.

# Docstrings throughout this codebase use single backticks for inline
# code/names (`core.course_geometry.py`, `total_time_s`, etc.) -- ordinary
# Markdown habit, but in RST a single backtick is an "interpreted text
# role" that tries to resolve its contents as a cross-reference target,
# not literal text, and errors ("Unknown target name") when it can't.
# default_role makes a bare single-backtick span behave as inline
# literal/code text (RST's `` double-backtick `` meaning) instead,
# matching what these docstrings actually intended, without having to
# rewrite every one of them to double backticks.
default_role = "literal"

templates_path: list[str] = []
exclude_patterns: list[str] = []

html_theme = "alabaster"  # bundled with Sphinx, no extra dependency
html_static_path: list[str] = []


def _skip_dotted_field_names(app, what, name, obj, skip, options):
    """core.pydantic_mapper.ExperimentIndexModel's fields are named with
    literal dots (e.g. "output.results.kpis.total_time_s", taken
    straight from TYPE_MAP's dot-notation keys -- see that module's
    docstring). A real Python identifier can never contain a dot, so
    autodoc's member-listing misinterprets a dotted field name as a
    nested attribute *path* and fails to import it, producing one
    warning per field (~25 of them). Skip anything whose name contains
    a dot -- there's no legitimate case where that's a real class
    member rather than this specific artifact.
    """
    if "." in name:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", _skip_dotted_field_names)
