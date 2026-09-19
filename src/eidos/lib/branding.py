"""
eidos.lib.branding -- Single source of truth for EIDOS^TT window-title text.

Every EIDOS^TT GUI (Manager, Viewer, Designer, Analyzer, Navigator, Trainer)
built its top-level window title by hand-typing "EIDOS^TT <name>", which let
the prefix and the per-app name drift out of sync with each other (and with
README.md's naming) independently in each file. window_title() is the one
place that spelling lives now; entry points supply only their own display
name.
"""

from __future__ import annotations

PROJECT_TITLE = "EIDOS^TT"


def window_title(app_name: str) -> str:
    """Return the standard '{PROJECT_TITLE} {app_name}' window title."""
    return f"{PROJECT_TITLE} {app_name}"
