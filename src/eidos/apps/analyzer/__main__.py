"""
Enables `python -m eidos.apps.analyzer <StrategySetDir> <RunID> <Nseg> <Seed>`.

Without this file, `python -m eidos.apps.analyzer` fails with
"No module named eidos.apps.analyzer.__main__; 'eidos.apps.analyzer' is a
package and cannot be directly executed" -- a package's own __init__.py is
NOT run as __main__ by `python -m <package>` the way a flat module.py is;
only an explicit __main__.py submodule is. The eidos-analyzer console_scripts
wrapper (pyproject.toml: "eidos.apps.analyzer:main") is unaffected by this,
since it imports main directly rather than using -m -- this is only needed
for eidos/apps/viewer.py's "Launch Analyzer" button, which subprocess-launches
sibling apps via `python -m eidos.apps.<name>` (see viewer.py's
_launch_script docstring).
"""

from eidos.apps.analyzer import main

if __name__ == "__main__":
    main()
