"""
EIDOS^TT Viewer

Browses generated strategy records (strategy_*.json exports), overlays the
power/speed/W'-balance profile against the course, and is the launch point
for Navigator, Trainer, Analyzer, and Exporter on a selected FIT/strategy.

Package layout, following the same pattern as eidos.apps.analyzer:

    - helpers.py -- retrieve_nested_key / load_initial_records (no Qt)
    - widgets.py -- CourseMapWidget (QPainter-based course map)
    - dialogs.py -- RecordDetailDialog / ExportReportDialog / StrategySelectorDialog
    - window.py -- TTSimulatorViewer, the QMainWindow tying it all together

This __init__.py keeps the CLI entry point (main()), so the console_scripts
target in pyproject.toml ("eidos.apps.viewer:main") is unchanged by the split.
"""

import logging
import sys

from PySide6.QtWidgets import QApplication

from core.logging_setup import configure_logging, log_banner
from eidos.apps.viewer.window import TTSimulatorViewer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# V. Application entry point
# ----------------------------------------------------------------------
def main() -> None:
    configure_logging()
    app = QApplication(sys.argv)

    # No command-line arguments are passed to the viewer.
    log_banner(logger, "EIDOS^TT Viewer: Start Execution")
    viewer = TTSimulatorViewer()
    viewer.show()

    # 2. Start the event loop
    exit_code = app.exec()

    # 3. Exit
    log_banner(logger, "EIDOS^TT Viewer closed.")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()