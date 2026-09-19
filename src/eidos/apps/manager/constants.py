"""
eidos.apps.manager.constants -- Display-label maps and subprocess-launch
identifiers shared across the manager package.
"""

from eidos.lib.branding import window_title

SCRIPT_DISPLAY_MAP = {
    "generator": window_title("Generator"),
    "viewer": window_title("Viewer"),
}
SECTION_MAP = {
    'run_set_id': 'Execution Metadata',
    'input.versions': 'Input: Versions',
    'input.git_state': 'Input: Git State',
    'input.settings.physiological': 'Input: Settings (Physiological)',
    'input.settings.physical': 'Input: Settings (Physical)',
    'input.settings.run': 'Input: Settings (Run)',
    'output.metadata': 'Output: Metadata',
    'output.results': 'Output: Results (KPIs)',
}

# --------------------------------------------------
# File path constants
# --------------------------------------------------
GENERATOR_SCRIPT = "generator.py"
VIEWER_SCRIPT = "viewer.py"

# GENERATOR_SCRIPT/VIEWER_SCRIPT above are display/dict-key identifiers
# only (format_script_name's SCRIPT_DISPLAY_MAP lookup, self.processes
# dict keys, the script_name == GENERATOR_SCRIPT routing checks below) --
# NOT filesystem paths. The actual subprocess target is looked up here,
# via `python -m <module>` (see start_external_process), rather than via a
# resolved sibling file path (formerly SCRIPT_DIR/script_name): that
# approach broke the moment eidos.apps.viewer became a package instead of
# a flat module (no viewer.py file exists to resolve a path to anymore).
# `-m` doesn't care whether the target is a flat module or a package with
# __init__.py, so this survives eidos.apps.generator being split the same
# way later too. See eidos/apps/viewer/dialogs.py's _launch_script
# docstring for the same fix applied to Viewer's own sibling launches.
SCRIPT_MODULE_MAP = {
    GENERATOR_SCRIPT: "eidos.apps.generator",
    VIEWER_SCRIPT: "eidos.apps.viewer",
}
