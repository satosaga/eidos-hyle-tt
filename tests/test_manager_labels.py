"""
SCRIPT_DISPLAY_MAP's keys must match Path(script).stem for
GENERATOR_SCRIPT/VIEWER_SCRIPT: format_script_name() looks the label up
via that stem, and a mismatch means it silently falls back to naive
title-casing ("Generator") instead of the intended branded label
("EIDOS^TT Generator") -- with no error raised, since this is a
display-only path that py_compile/pyflakes/import checks can't catch.
These tests pin the two things that must stay in sync.
"""

import importlib
from pathlib import Path

from eidos.apps.manager.constants import (
    GENERATOR_SCRIPT,
    SCRIPT_DISPLAY_MAP,
    SCRIPT_MODULE_MAP,
    VIEWER_SCRIPT,
)
from eidos.apps.manager.helpers import format_script_name


def test_script_display_map_keys_match_script_constants():
    for script in (GENERATOR_SCRIPT, VIEWER_SCRIPT):
        stem = Path(script).stem
        assert stem in SCRIPT_DISPLAY_MAP, (
            f"{script!r} (stem {stem!r}) is missing from SCRIPT_DISPLAY_MAP -- "
            f"format_script_name() will silently fall back to title-casing"
        )


def test_format_script_name_returns_intended_labels():
    assert format_script_name(GENERATOR_SCRIPT) == "EIDOS^TT Generator"
    assert format_script_name(VIEWER_SCRIPT) == "EIDOS^TT Viewer"


def test_script_module_map_targets_are_importable():
    """SCRIPT_MODULE_MAP drives `python -m <module>` subprocess launches from
    the manager; a stale target here would fail silently until a user
    actually clicked the button."""
    for script, module_name in SCRIPT_MODULE_MAP.items():
        module = importlib.import_module(module_name)
        assert hasattr(module, "main"), (
            f"{module_name} (launched for {script!r}) has no main()"
        )
