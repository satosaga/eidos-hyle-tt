"""
Enables `python -m eidos.apps.viewer`.

See eidos/apps/analyzer/__main__.py's docstring for why this file has to
exist separately from __init__.py: a package's own __init__.py is not run
as __main__ by `python -m <package>` the way a flat module.py is.
"""

from eidos.apps.viewer import main

if __name__ == "__main__":
    main()
