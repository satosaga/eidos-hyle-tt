"""
eidos.apps.viewer.dialogs -- Small QDialog subclasses.

RecordDetailDialog (CLI-args viewer, shown via show() -- non-modal),
ExportReportDialog (export/config report viewer, modal, shown via
exec()), and StrategySelectorDialog (FIT-file picker that launches
Navigator/Trainer/Analyzer as subprocesses, modal, shown via exec()).
Exporter is launched separately, from TTSimulatorViewer.start_sync_export
in window.py, not from this dialog.
"""

import logging
import os
import platform
import re
import subprocess
import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from core.io_config import PROJECT_ROOT
from core.logging_setup import log_banner

logger = logging.getLogger(__name__)

# Friendly display names for the log_banner() shown just before launching
# a grandchild process in _launch_script, keyed by the same module_name
# string its callers already pass in.
_LAUNCH_DISPLAY_NAMES = {
    "eidos.apps.navigator": "Navigator",
    "eidos.apps.trainer": "Trainer",
    "eidos.apps.analyzer": "Analyzer",
}


# --------------------------
# III.  
# --------------------------
class RecordDetailDialog(QDialog):
    """Modal dialog that displays the CLI argument string for an optimization record."""
    def __init__(self, cli_args, parent=None):
        """Initialize the dialog with the given cli_args string and display it in a read-only text editor."""
        super().__init__(parent)
        self.setWindowTitle("CLI Arguments")
        self.resize(550, 150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        content = (
            "--- CLI Arguments ----------------------\n"
            f"{cli_args}\n"
            "-----------------------------------------"
        )

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(content)
        self.text_edit.setReadOnly(True)
        # Monaco font; slightly muted background for readability
        self.text_edit.setFont(QFont("Monaco", 11) if sys.platform == "darwin" else QFont("Consolas", 11))
        self.text_edit.setStyleSheet("QTextEdit { background-color: #f8f8f8; color: #333; border: none; }")
        
        layout.addWidget(self.text_edit)


class ExportReportDialog(QDialog):
    """Pop-up window for displaying export results and configuration reports."""
    def __init__(self, title, report_text, parent=None):
        """Initialize the dialog with title and report_text displayed in a read-only monospace editor."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(700, 500)
        layout = QVBoxLayout(self)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(report_text)
        
        font = QFont("Menlo", 13)
        self.text_edit.setFont(font)
        
        layout.addWidget(self.text_edit)

        # Only the export log contains an "Export directory: <dir>" line
        # (exporter.py's "Export directory: %s" % base_dir); other reports
        # shown in this same dialog class (e.g. "Record Configuration
        # Profile") won't match, so the Open Directory button is simply omitted then.
        match = re.search(r"Export directory: (\S+)", report_text)
        self.export_dir = match.group(1) if match else None

        btn_layout = QHBoxLayout()

        if self.export_dir:
            open_folder_button = QPushButton("Open Directory")
            open_folder_button.clicked.connect(self._open_export_dir)
            btn_layout.addWidget(open_folder_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        btn_layout.addWidget(close_button)

        layout.addLayout(btn_layout)

    def _open_export_dir(self):
        """Reveal self.export_dir in the OS's default file browser (Finder/Explorer/whatever the Linux DE provides). Only runs on button press, not automatically."""
        if not self.export_dir or not os.path.isdir(self.export_dir):
            return
        if platform.system() == "Darwin":
            subprocess.Popen(["open", self.export_dir])
        elif platform.system() == "Windows":
            os.startfile(self.export_dir)
        else:
            subprocess.Popen(["xdg-open", self.export_dir])


class StrategySelectorDialog(QDialog):
    """Dialog listing available FIT files and launching Navigator, Trainer, or Analyzer."""
    def __init__(self, fit_files, run_info, parent=None):
        """Initialize the dialog with fit_files list and run_info, and build the launcher UI."""
        super().__init__(parent)
        self.setWindowTitle("🏁 Launch Strategy")
        self.setFixedWidth(400)
        self.run_info = run_info 

        layout = QVBoxLayout(self)
        
        self.list_widget = QListWidget()
        self.list_widget.setFont(QFont("Menlo", 13))
        for f in sorted(fit_files):
            self.list_widget.addItem(f)
        layout.addWidget(self.list_widget)

        # Button layout
        btn_layout = QHBoxLayout()
        
        # --- Button styles ---
        # Launch button (orange)
        orange_style = """
            QPushButton {
                background-color: #FF8833; color: white; font-weight: bold;
                border-radius: 5px; padding: 8px;
            }
            QPushButton:hover { background-color: #FF6600; }
        """
        # Close button (blue)
        blue_style = """
            QPushButton {
                background-color: #0099FF; color: white; font-weight: bold;
                border-radius: 5px; padding: 8px;
            }
            QPushButton:hover { background-color: #0077CC; }
        """

        # Navigator launch button
        nav_btn = QPushButton("📣 Launch Navigator")
        nav_btn.setStyleSheet(orange_style)
        nav_btn.clicked.connect(lambda: self._launch_script('eidos.apps.navigator'))
        btn_layout.addWidget(nav_btn)

        # Trainer launch button
        train_btn = QPushButton("🚴 Launch Trainer")
        train_btn.setStyleSheet(orange_style)
        train_btn.clicked.connect(lambda: self._launch_script('eidos.apps.trainer'))
        btn_layout.addWidget(train_btn)

        layout.addLayout(btn_layout)

        # Analyzer launch button, on its own row below Navigator/Trainer --
        # the workflow is to generate an activity with one of those first,
        # then inspect it with the Analyzer, so it reads as a distinct next
        # step rather than a third peer alongside them. eidos.apps.analyzer
        # only reads sys.argv[1:5] (StrategySetDir/RunID/Nseg/Seed) and ignores
        # anything past that, so the same 5-arg command built for
        # Navigator/Trainer (which appends the intensity factor as a 5th
        # arg) works here unchanged. addWidget (not addLayout) gives it the
        # same full-width sizing as the Close button below.
        analyzer_btn = QPushButton("🔍 Launch Analyzer")
        analyzer_btn.setStyleSheet(orange_style)
        analyzer_btn.clicked.connect(lambda: self._launch_script('eidos.apps.analyzer'))
        layout.addWidget(analyzer_btn)

        # Close button
        close_button = QPushButton("Close")
        close_button.setStyleSheet(blue_style)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def _launch_script(self, module_name):
        """Launch module_name (e.g. 'eidos.apps.navigator') as a subprocess
        using the selected FIT file and run_info args.

        Invoked via `python -m <module_name>` rather than a resolved sibling
        file path (the pre-src-layout / pre-P3-split approach): that assumed
        every launched app was a single flat .py file living right next to
        this one, which broke the moment analyzer.py became a subpackage
        (eidos/apps/analyzer/) with no analyzer.py file at all. `-m` doesn't
        care whether the target is a flat module or a package with a
        __init__.py -- it resolves purely by import identity, via whatever
        made eidos.apps.viewer itself importable (the editable install) --
        so this keeps working across any future split the same way
        eidos.apps.analyzer's console_scripts entry already does.
        """
        item = self.list_widget.currentItem()
        if not item: return

        filename = item.text()
        # run_info contents: (strategy_set_dir, run_set_id, n_seg, seed)
        strategy_set_dir_name, run_set_id, n_seg, seed = self.run_info

        # Extract intensity factor (IF) from filename
        import re
        match = re.search(r"IF(\d+)", filename)
        if_val = f"{int(match.group(1))/100:.2f}" if match else "1.00"

        # Build the argument list identical to the CLI invocation.
        # -u (unbuffered stdio) still matters here even though Navigator/
        # Trainer/Analyzer now all go through logging.StreamHandler (which
        # auto-flushes every record on its own): stdout is block-buffered
        # whenever it's not a tty -- which it isn't, piped all the way up
        # through this Popen -> viewer -> the manager-launched viewer's
        # own stdio -- and anything that bypasses the handler (print(),
        # OpenGL/GLUT's own stderr chatter in Trainer, an uncaught
        # traceback) would otherwise still sit in Python's internal buffer
        # instead of reaching the log in real time. Matches the -u already
        # used for Generator/Viewer's own launch in
        # eidos.apps.manager.window.start_external_process.
        command = [
            sys.executable,
            "-u",
            "-m", module_name,
            strategy_set_dir_name,  # arg 1: strategy-set directory name
            run_set_id,    # arg 2: run set ID
            n_seg,         # arg 3: number of segments
            seed,          # arg 4: seed
            if_val         # arg 5: intensity factor
        ]

        try:
            # A log_banner() right before Popen marks this hand-off in the
            # Manager's Execution Log. This grandchild inherits stdio all
            # the way up to Manager (see the -u comment above) but Popen
            # here is fire-and-forget -- Manager gets no signal of its own
            # that a new process just started talking on the same pipe --
            # so without an explicit marker, the switch from Viewer's own
            # output to e.g. Trainer's would only be visible by reading
            # the text itself.
            display_name = _LAUNCH_DISPLAY_NAMES.get(module_name, module_name)
            log_banner(logger, f"Launching {display_name}")

            # cwd = true repo root (via core.io_config.PROJECT_ROOT) so the
            # launched app's own data-directory resolution (BASE_*_DIR, all
            # of which are already absolute -- see io_config.PROJECT_ROOT)
            # doesn't depend on where this GUI process itself happened to
            # be launched from.
            subprocess.Popen(command, cwd=PROJECT_ROOT)
            logger.info("Launched: %s", ' '.join(command))
        except Exception as e:
            logger.error("Failed to launch %s: %s", module_name, e)

