"""
EIDOS^TT Manager

Central hub for creating/editing rider-environment-run configuration
files and filter files, then launching Generator (strategy optimization)
or Viewer (result browsing) as subprocesses on a chosen config.

Package layout:

    - constants.py -- display-label maps, subprocess-launch identifiers
    - validators.py -- small QValidator/QDoubleSpinBox subclasses
    - helpers.py -- small standalone helper functions
    - file_managers.py -- ConfigFileManager / FilterFileManager
    - editor_panes.py -- ConfigurationEditorPane / FilterEditorPane
    - manager_panes.py -- BaseManagerPane / GenerationManagerPane / ViewingManagerPane
    - window.py -- TTManagerGUI, the QMainWindow tying it all together

This __init__.py keeps the CLI entry point (main()), so the console_scripts
target in pyproject.toml ("eidos.apps.manager:main") is unchanged by the split.
"""

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from core.io_config import BASE_CONFIGS_DIR, BASE_FILTERS_DIR, BASE_STRATEGIES_DIR
from core.logging_setup import configure_logging
from eidos.apps.manager.window import TTManagerGUI


def main() -> None:
    # Manager doesn't log much of its own (mostly a GUI whose own status
    # is shown via the status bar / QMessageBox), but this still matters:
    # editor_panes.py's logger calls and anything from libraries it
    # imports should come out in the same timestamped format as every
    # other eidos.apps.* entry point when eidos-manager is run directly
    # from a terminal.
    configure_logging()

    # Ensure required directories exist
    Path(BASE_CONFIGS_DIR).mkdir(exist_ok=True)
    Path(os.path.join(BASE_CONFIGS_DIR, "templates")).mkdir(exist_ok=True)
    Path(BASE_FILTERS_DIR).mkdir(exist_ok=True)
    Path(BASE_STRATEGIES_DIR).mkdir(exist_ok=True)

    app = QApplication(sys.argv)
    try:
        window = TTManagerGUI()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", f"The application failed to start due to an error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
